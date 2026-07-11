# adk-tinker

Kicking the tires on [ADK for Go 2.0](https://developers.googleblog.com/announcing-adk-go-20/)
(`google.golang.org/adk/v2`, released 2026-06-30).

Seventeen runnable programs, each a step up in ADK 2.0 features:

| Command | What it is |
|---------|------------|
| `go run .` | **hello agent** — one Gemini agent + one `add` tool, one hard-coded turn |
| `go run ./hitlgraph console` | **workflow graph + human-in-the-loop** — LLM drafts a post, a human approves/edits it, then it publishes |
| `go run ./durablehitl submit/resume` | **durable, search-grounded HITL** — the pause survives a full process restart (state in SQLite) |
| `go run ./fanout "topic"` | **fan-out / fan-in** — 3 grounded drafters run in parallel, an editor picks the best, then a human approves |
| `go run ./schemahitl console` | **schema-validated HITL** — the human's reply must match a JSON `ResponseSchema` (approve/reject/edit) |
| `go run ./serve web -port 8791 api` | **serve over HTTP** — the same graph as a REST API; pause/resume driven by `curl` |
| `go run ./a2a demo` | **A2A** — expose an agent over the agent-to-agent protocol and call it from a client |
| `go run ./claude console` | **Claude instead of Gemini** — the same HITL graph on `claude-opus-4-8` via a custom `model.LLM` |
| `go run ./claudetools` | **tool calling on Claude** — the `add` function tool, driven by Claude through the same adapter |
| `go run ./claudesearch "topic"` | **search-grounded Claude** — `geminitool.GoogleSearch{}` auto-mapped to Anthropic's `web_search` |
| `go run ./a2amesh "question"` | **cross-provider mesh** — a Claude orchestrator delegates to a Gemini specialist over A2A |
| `go run ./adk46 "peak"` | **ADK × ADK** 🏔️ — use the *Agent* Development Kit to plan a hike in the *Adirondacks* |
| `go run ./adk46er bag/list/next` | **durable 46er tracker** — log peaks (SQLite), see progress, ask a Claude mentor for your next |
| `go run ./rangerguide "question"` | **ranger↔guide mesh** — a Claude guide consults a Gemini "park ranger" over A2A |
| `go run ./whichpeak "profile"` | **Trailhead Oracle** — a search-grounded Claude pick for which peak to hike today |
| `go run ./eval` | **LLM-as-judge eval harness** — score an agent's answers against rubrics (the Dev UI "Evals" tab is a 501 stub in Go v2.0.0) |
| `go run ./loopcritic "peak"` | **self-critique loop** — a `LoopAgent` where a safety critic keeps sending the plan back until it passes, then `exit_loop`s |

## Hello agent

```bash
export GOOGLE_API_KEY=...   # free key: https://aistudio.google.com/apikey
go run .
```

Expected: the agent calls the `add` tool for `2 + 3` and reports `5`.

## Workflow graph + human-in-the-loop (`hitlgraph/`)

A two-node graph built on the v2 workflow engine:

```
Start ─▶ draft ─▶ review
        (LLM)     (HITL pause + decision)
```

- **draft** — `workflow.NewAgentNode(llmAgent, …)` drops a Gemini agent into the
  graph as a single-turn node; it turns your typed topic into a post draft.
- **review** — a re-entry node (`workflow.ResumeOrRequestInput` +
  `NodeConfig{RerunOnResume:&true}`). On the first pass it shows the draft and
  returns `workflow.ErrNodeInterrupted`, which **suspends the whole workflow**
  until you answer; then it re-runs with the draft still in hand plus your
  reply and acts on it — `yes`/Enter approves, `no` discards, anything else is
  your edit.

The graph is wired with `workflow.Chain(Start, draft, review)` and wrapped as an
agent via `workflowagent.New`. The console launcher (`full.NewLauncher()`) drives
the pause/resume and renders prompts.

```bash
export GOOGLE_API_KEY=...
go run ./hitlgraph console

# User  -> the launch of ADK for Go 2.0
# Agent -> (draft appears, workflow PAUSES)
# User  -> yes            # approve as-is — or type an edit, or "no" to discard
# Agent -> ✅ Published (approved as-is): ...
```

## Durable HITL — the pause survives a restart (`durablehitl/`)

Same `draft → review → publish` graph, but with two upgrades that compose:

1. **Search-grounded draft** — the `draft` agent gets `geminitool.GoogleSearch{}`,
   so it drafts from live info.
2. **Durable state** — sessions are stored in **SQLite** via
   `database.NewSessionService(sqlite.Open("adk_sessions.db"))`, so a paused
   workflow outlives the process that started it.

It runs as **two separate processes** sharing the DB file:

```bash
export GOOGLE_API_KEY=...

# Process 1: search + draft, then PAUSE and exit. State is saved to SQLite.
go run ./durablehitl submit "the launch of ADK for Go 2.0"

# Process 2 (brand new process): reload the paused session and finish.
go run ./durablehitl resume "ADK for Go 2.0 is here 🚀 ...your edited post..."
```

How resume works across processes (no launcher involved):

- The `review` node's `RequestInput` is recorded in the event log as a
  long-running `FunctionCall` named `workflow.WorkflowInputFunctionCallName`
  (`"adk_request_input"`) — persisted in SQLite.
- `resume` calls `sessionSvc.Get(...)`, scans the persisted events for that
  call with no matching `FunctionResponse` yet, and replies with a
  `genai.FunctionResponse{ID, Name, Response: {"payload": <text>}}`.
- `runner.Run` routes that response to the waiting node by ID and the workflow
  continues into `publish`. (This is exactly what the console launcher does
  internally in `cmd/launcher/console/hitl.go` — just split across two processes.)

Uses the pure-Go SQLite driver `github.com/glebarez/sqlite` (no cgo), the same
one ADK uses in its own tests. The `adk_sessions.db` file is gitignored.

### HITL pause patterns in ADK 2.0

| Pattern | Primitive | Upstream example |
|---------|-----------|------------------|
| Static chain handoff (`durablehitl`, `fanout`) | emit `RequestInput` + return `ErrNodeInterrupted`; reply → next node | `examples/workflow/hitl_simple` |
| Single-node re-entry (`hitlgraph`, `schemahitl`) | `workflow.ResumeOrRequestInput(...)` + `NodeConfig{RerunOnResume:&true}` | `examples/workflow/hitl_rerun` |
| Dynamic orchestrator | `workflow.RunNode(...)` + `ctx.ResumedInput(id)` | `examples/workflow/dynamic/hitl` |

## Fan-out / fan-in (`fanout/`)

Best-of-N with a human gate. Three search-grounded drafters (distinct voices)
run **in parallel**, a `JoinNode` barrier gathers them, an LLM editor picks the
best, then a human approves:

```
         ┌─▶ drafter_punchy    ─┐
Start ───┼─▶ drafter_technical ─┼─▶ gather ─▶ format ─▶ editor ─▶ review
         └─▶ drafter_concise   ─┘  (Join)    (func)    (LLM)    (HITL)
```

```bash
go run ./fanout "the AI walled gardens"     # or: go run ./fanout   (prompts for a topic)
```

- `workflow.NewEdgeBuilder()` with `AddFanOut(Start, …)` / `AddFanIn(join, …)` /
  `Add(a, b)` expresses the barrier (`workflow.Chain` can't).
- The `JoinNode` fires once after **all** predecessors finish and hands its
  successor a `map[nodeName]output` — the formatter looks drafts up by node name.
- A single-turn `AgentNode`'s output propagates via `Event.Output`.
- Unlike the `console` demos, `fanout` uses its **own runner + formatter** (not
  the generic launcher) so the three parallel drafts and the pick print as a
  clean labeled report instead of one concatenated blob. `review` is a re-entry
  node: `yes`/Enter approves, `no` discards, anything else is your edit.

## Schema-validated HITL (`schemahitl/`)

The pause carries a JSON `ResponseSchema`, so the human's reply must be a
structured object — `{"decision":"approve|reject|edit","text":"…"}` — validated
by the engine before resuming (a mismatch yields `workflow.ErrInvalidResumeResponse`
and keeps the node waiting for a corrected retry).

- Schema type: `*jsonschema.Schema` from `github.com/google/jsonschema-go/jsonschema`
  (set `Type:"object"`, `Properties`, `Enum []any`, `Required []string`).
- A validated reply is delivered as a **`map[string]any`** (a JSON object decoded
  into `any`) — read `m["decision"].(string)`, `m["text"].(string)`.
- Uses the **re-entry** pattern (`ResumeOrRequestInput` + `NodeConfig{RerunOnResume:&true}`)
  so the node keeps its draft input *and* receives the decision in one place.
- In `console`, type a full JSON object on one line; a bare word gets wrapped as
  `{"payload": …}` and fails the object schema.

## Serve over HTTP (`serve/`)

The same `full.NewLauncher()` that runs `console` also serves REST — the mode is
`web api` (there is **no** `rest` keyword). Flags are positional: `-port` is a
`web` flag (before `api`).

```bash
go run ./serve web -port 8791 api        # REST under http://localhost:8791/api

# create session → run (pauses) → resume with a FunctionResponse:
SID=$(curl -s -X POST .../api/apps/served_review/users/u/sessions -d '{}' | jq -r .id)
curl -s -X POST .../api/run -d '{"appName":"served_review","userId":"u","sessionId":"'$SID'",
  "newMessage":{"role":"user","parts":[{"text":"<topic>"}]}}'          # → requestedInput.interruptId
curl -s -X POST .../api/run -d '{"appName":"served_review","userId":"u","sessionId":"'$SID'",
  "newMessage":{"role":"user","parts":[{"functionResponse":{"id":"<interruptId>",
  "name":"adk_request_input","response":{"payload":"<approved text>"}}}]}}'   # → event.output
```

`POST /api/run` is non-streaming: it returns the whole turn's events as one JSON
array and returns *at* the pause, which is what makes HITL curl-drivable. Wire a
SQLite `SessionService` into `launcher.Config` (as `serve/` does) or the web
launcher defaults to in-memory (sessions vanish on restart).

## Agent-to-Agent, A2A (`a2a/`)

Expose an agent over the [A2A protocol](https://github.com/a2aproject/a2a-go) and
call it from a separate process.

```bash
export GOOGLE_API_KEY=...          # server runs the LLM; the client does not
go run ./a2a server 8792 &         # serves an agent card + JSON-RPC /invoke
go run ./a2a client http://127.0.0.1:8792 "one line about Go agents"
# or all-in-one:  go run ./a2a demo "…"
```

- **Server** (`server/adka2a/v2`, package `adka2a`): `adka2a.NewExecutor(ExecutorConfig{RunnerConfig: runner.Config{…}})`,
  then serve `a2asrv.NewStaticAgentCardHandler(card)` at `a2asrv.WellKnownAgentCardPath`
  and `a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor))` at `/invoke`.
- **Client** (`agent/remoteagent/v2`, package `remoteagent`): `remoteagent.NewA2A(A2AConfig{AgentCardProvider: remoteagent.NewAgentCardProvider(baseURL)})`
  returns a plain `agent.Agent` you run through `runner.New`/`Run` like any local agent.
- Note the import-path/package-name mismatch: use the `/v2` paths, whose packages
  are named `adka2a` / `remoteagent` (not `v2`). Needs `github.com/a2aproject/a2a-go/v2`.

## Claude instead of Gemini (`claude/` + `claudemodel/`)

ADK is Google's framework but it's **model-agnostic** — an agent takes any
`model.LLM`, and `model/gemini` is just one implementation. `claudemodel/` is a
second one, backed by the official
[`github.com/anthropics/anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go)
(defaults to `claude-opus-4-8`). `claude/` is the `hitlgraph` demo with exactly
one line changed:

```go
model := claudemodel.NewModel("")   // instead of gemini.NewModel(ctx, "...", ...)
```

Everything else — the graph, the `AgentNode`, the re-entry HITL review, the
console launcher — is untouched ADK.

```bash
export ANTHROPIC_API_KEY=...   # or `ant auth login` (the Go SDK reads the profile)
go run ./claude console
```

The adapter implements ADK's tiny model interface —
`GenerateContent(ctx, *LLMRequest, stream) iter.Seq2[*LLMResponse, error]` — by
translating the genai-shaped request into an Anthropic Messages call and shaping
the reply back into a `genai.Content`:

- genai `Config.SystemInstruction` → Anthropic `System`
- genai `Contents` (role `model`↔`assistant`) → Anthropic `Messages`
- Anthropic `TextBlock`s → text parts on a `model`-role `genai.Content`

**Tool calling works too** (`go run ./claudetools` — the `add` tool on Claude):

- genai function declarations (`ParametersJsonSchema`) → Anthropic tool defs
- a genai `FunctionCall` part → an Anthropic `tool_use` block, and a
  `FunctionResponse` part → a `tool_result` block, with the tool-use **ID
  threaded** so Anthropic pairs each result to its call across the loop
- an Anthropic `tool_use` in the reply → a genai `FunctionCall`, so the ADK
  runner executes the Go tool and loops until Claude produces the final text

**Search grounding works too** (`go run ./claudesearch "topic"`): the adapter
detects a `geminitool.GoogleSearch{}` tool in the request and maps it to
Anthropic's own `web_search` server tool (with `pause_turn` handling). So the
*same* agent — declaring Gemini's `GoogleSearch` — is web-grounded on either
provider; on Claude it runs live searches and answers with cited facts.

Scope: text, system instruction, function tools, and web-search grounding. Other
Gemini-specific server tools (code execution, Maps, …) have no Claude equivalent
and are ignored.

## Cross-provider A2A mesh (`a2amesh/`)

A **Claude orchestrator** delegates a factual question to a **Gemini specialist**
over the A2A protocol — two different providers' agents in one system:

```
Claude orchestrator (claude-opus-4-8)
        │  calls the "gemini_specialist" tool
        ▼
agenttool ──A2A/JSON-RPC──▶ Gemini specialist (gemini-3.5-flash), served over A2A
        ▲                            answers the sub-question
        └──────────── answer ────────┘   → Claude synthesizes the reply
```

```bash
export GOOGLE_API_KEY=...      # Gemini specialist  (+ Anthropic creds for Claude)
go run ./a2amesh "what is the tallest volcano in the solar system?"
```

It composes the whole toolkit: `claudemodel` + its **tool calling** (the
agent-as-tool uses the `Parameters` JSON-Schema path — the blocker fix), plus the
A2A `server`/`remoteagent` wiring. Claude emits a `tool_use` for `gemini_specialist`;
ADK routes it over A2A to the Gemini agent; the answer returns as a `tool_result`;
Claude synthesizes. (If the free-tier Gemini quota is exhausted, Claude degrades
gracefully — it reports the specialist was unreachable rather than failing.)

## ADK × ADK — plan a High Peaks hike (`adk46/`)

A little word-play: use the **A**gent **D**evelopment **K**it to bag the **ADK**
(the Adirondacks) — the [46 High Peaks](https://en.wikipedia.org/wiki/Adirondack_High_Peaks).
A team of Adirondack scout agents fans out to research a peak **in parallel**
(each grounded on live web search), a "head guide" agent synthesizes a trip
brief, and *you* — the hiker — approve, edit, or scrap it.

```
  🗺️ route_scout ─┐
Start ─ ⛅ sky_watcher ─┼─ gather ─ format ─ 🏔️ head_guide ─ review (you)
  🎒 pack_master  ─┘  (Join)  (func)     (LLM)          (HITL)
```

```bash
export ANTHROPIC_API_KEY=...     # runs entirely on Claude
go run ./adk46 "Mount Marcy — a day hike this weekend"
```

It's the whole toolkit wearing an Adirondack hat: **fan-out/fan-in**,
**search-grounded Claude** (each scout's `geminitool.GoogleSearch{}` → Anthropic
`web_search`), an LLM **synthesizer**, and a **HITL** approval — and the output is
genuinely useful (it'll catch a summit snow forecast or a closed trail).

Three companions round out the ADK × ADK corner, each foregrounding a different
capability:

- **`adk46er`** — a **durable** 46er tracker: `bag "Marcy"`, `list` your progress
  toward 46 (persisted in SQLite), and `next` asks a Claude mentor which peak to
  do next given what you've bagged.
- **`rangerguide`** — the **cross-provider A2A mesh**, themed: a Claude "trail
  guide" consults a Gemini "park ranger" (regulations) over A2A, then advises.
- **`whichpeak`** — a **search-grounded** "Trailhead Oracle": give it your fitness
  and time; it checks the forecast and recommends a peak for today.

## Evaluating agents — an LLM-as-judge harness (`eval/`)

ADK 2.0's Dev UI has an **Evals** tab, but in Go v2.0.0 the backend's eval REST
endpoints are stubs (`controllers.Unimplemented` → HTTP 501) and there's no public
eval package. So `eval/` is a DIY harness in the spirit of the upstream
`examples/web/agents/llmauditor.go` (an LLM critic + reviser).

For each case it produces a recommendation (running the agent-under-test, or a
planted `ForceOutput`), then a **panel of judge agents** (different Claude models)
scores it against the case rubric and returns JSON `{pass, score, rationale}`; the
panel majority-votes. The demo evaluates an Adirondack peak recommender against
hiker-profile rubrics.

Two things keep it honest rather than a rubber stamp — both came out of an
**adversarial review of the harness itself** (see the review-workflow pattern
above), which found that a naïve 5/5 pass rate proves nothing:

- a **negative control** — a planted, dangerous recommendation the judges *must*
  fail, so an always-pass judge is caught instead of rewarded; and
- a **judge panel** (`opus-4-8` + `sonnet-5` + `haiku-4-5`) with majority vote, so
  one lenient/noisy judge can't decide a case.

```bash
export ANTHROPIC_API_KEY=...      # all agents run on Claude
go run ./eval                      # exits non-zero if the panel contests a labeled verdict
```

It has teeth: a sample run rejected the negative control 3/3, and flagged a real
borderline — a peak recommended for a 2-hour window that actually needs 3–4 hours.
`parseVerdict` / `vote` are unit-tested.

## Self-critique loop (`loopcritic/`)

The counterpart to `eval/`: there a judge grades an agent from the **outside**;
here a critic lives **inside** the agent and it self-corrects until it passes.

Built on ADK's **loop agent** (`agent/workflowagents/loopagent`), which runs its
sub-agents in sequence and repeats up to `MaxIterations` — or until a sub-agent
*escalates*:

```
┌──────────────── loop (≤ 4 iterations) ─────────────────┐
│  ✍️  planner  → drafts / revises the trip plan          │
│  🔍 safety_critic → PASS? call exit_loop (escalate→stop)│
│                     FAIL? numbered critique → next round │
└─────────────────────────────────────────────────────────┘
```

The critic holds the `tool/exitlooptool` tool; calling `exit_loop` sets
`Actions().Escalate = true`, which the loop detects to stop early. That tool call
runs through the Claude `model.LLM` adapter's tool-calling path — so this also
exercises `claudemodel`'s function calling.

```bash
export ANTHROPIC_API_KEY=...
go run ./loopcritic "Algonquin Peak, day hike, this weekend"
```

A sample run: draft v1 is a rough sketch → the critic rejects it with 5 numbered
fixes (*"I will not approve a sketch"*) → draft v2 addresses every one (exact
route, real distance/gain, weather thresholds, gear, a noon turnaround + Wright
Peak bailout) → the critic approves and `exit_loop`s. Also serves in the Dev UI
(`go run ./loopcritic web -port 8793 webui -api_server_address http://localhost:8793/api api`).

## Anatomy (the v2 pieces)

| Piece | Package | What it does |
|-------|---------|--------------|
| Model | `model/gemini` | `gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{APIKey})` |
| Tool | `tool/functiontool` | wraps a `func(agent.Context, In) (Out, error)`; schema inferred from Go types |
| Agent | `agent/llmagent` | `llmagent.New(Config{Name, Model, Instruction, Tools})` |
| Session | `session` | `session.InMemoryService()` — conversation state |
| Runner | `runner` | `runner.New(...)`, then `r.Run(...)` returns an `iter.Seq2[*session.Event, error]` |

## What's new in 2.0

- **Graph-based workflow engine** (`google.golang.org/adk/v2/workflow`) — nodes +
  edges, a scheduler, state persistence, and resumption across process restarts.
- **Human-in-the-loop** as a first-class primitive (pause / resume).
- **LLM agent modes**: Chat, Task, SingleTurn.
- **Unified `agent.Context`** — replaces 1.x's `ToolContext` / `CallbackContext`
  (breaking change). Don't mix `google.golang.org/adk` (v1) and `.../adk/v2` imports.

## Next ideas

The original four (search grounding, fan-out/fan-in, durable resume, HTTP
serving) plus schema-validated pauses and A2A are all done above. Further:

- Chain A2A: have the `serve/` HTTP graph call the `a2a/` agent as a
  `remoteagent` node — a multi-agent system split across processes.
- Swap the console for the **web Dev UI** (`go run ./serve web -port 8791 webui`)
  to click through the HITL pause in a browser.
- Add a durable, schema-validated pause served over HTTP (combine `schemahitl` +
  `serve` + the SQLite `SessionService`).
- Structured payloads: use `RequestInput.Payload` to ship the whole draft object
  (not just a string) to the UI, and a richer `ResponseSchema` for the reply.
