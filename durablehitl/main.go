// Command durablehitl is a search-grounded, DURABLE human-in-the-loop workflow:
// the pause survives a full process restart because session state is persisted
// to SQLite. It combines two ADK for Go 2.0 features from the hitlgraph demo:
//
//   - a Google-Search-grounded LLM node drafts a post from LIVE info, and
//   - the human-approval pause is persisted, so a *fresh* process resumes it.
//
// Graph (same three nodes as ../hitlgraph, but draft now searches):
//
//	Start ─▶ draft ─▶ review ─▶ publish
//	        (Gemini    (HITL     (function)
//	         + Google   pause,
//	         Search)    persisted)
//
// Two-phase demo across TWO separate processes sharing adk_sessions.db:
//
//	export GOOGLE_API_KEY=...
//	go run ./durablehitl submit "the ADK for Go 2.0 launch"
//	  → draft searches + writes a post, workflow PAUSES, process EXITS.
//	go run ./durablehitl resume "ADK for Go 2.0 is here 🚀 ..."
//	  → a brand-new process reloads the paused session and publishes.
//
// How the pause persists: the review node emits a RequestInput, which the engine
// records as a long-running FunctionCall named workflow.WorkflowInputFunctionCallName
// ("adk_request_input") in the session's event log (stored in SQLite). resume
// reloads the session, finds the unanswered call, and replies to it with a
// genai.FunctionResponse — exactly what the interactive console launcher does
// internally, but split across processes.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/glebarez/sqlite" // pure-Go SQLite (no cgo); the driver ADK itself uses
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/workflow"
)

const (
	appName   = "durable_hitl"
	userID    = "user-1"
	sessionID = "post-1" // fixed so `resume` can find what `submit` paused
	dbPath    = "adk_sessions.db"
	modelName = "gemini-2.5-flash-lite" // Gemini 2 model: required for Google Search grounding
)

func apiKey() string {
	if k := os.Getenv("GOOGLE_API_KEY"); k != "" {
		return k
	}
	return os.Getenv("GEMINI_API_KEY")
}

// openService opens (creating if needed) the SQLite-backed, durable session
// service and ensures its schema exists.
func openService() (session.Service, error) {
	svc, err := database.NewSessionService(sqlite.Open(dbPath), &gorm.Config{
		// Silence GORM's default logger: optional app/user-state lookups miss on
		// a fresh DB and would otherwise spam "record not found" every startup.
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		return nil, fmt.Errorf("open session db: %w", err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		return nil, fmt.Errorf("migrate session db: %w", err)
	}
	return svc, nil
}

// buildRootAgent assembles the Start→draft→review→publish workflow. Both submit
// and resume build it identically: only session state is persisted, not the
// graph — which is exactly how a durable workflow survives a redeploy.
func buildRootAgent(ctx context.Context, key string) (agent.Agent, error) {
	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: key})
	if err != nil {
		return nil, fmt.Errorf("create model: %w", err)
	}

	draftAgent, err := llmagent.New(llmagent.Config{
		Name:        "draft",
		Model:       model,
		Description: "researches the topic with Google Search and drafts one social post",
		Instruction: "You are a punchy copywriter. Use the Google Search tool to find one " +
			"recent, specific, factual detail about the topic, then write ONE short, engaging " +
			"social-media post (max ~220 chars) grounded in what you found. " +
			"Output only the post text — no preamble, no quotes.",
		Tools: []tool.Tool{geminitool.GoogleSearch{}},
	})
	if err != nil {
		return nil, fmt.Errorf("create draft agent: %w", err)
	}
	draftNode, err := workflow.NewAgentNode(draftAgent, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		return nil, fmt.Errorf("create draft node: %w", err)
	}

	// review: emit the draft as a RequestInput and pause the workflow.
	review := workflow.NewEmittingFunctionNode[any, any]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
			draft := strings.TrimSpace(fmt.Sprint(in))
			prompt := fmt.Sprintf(
				"Proposed post:\n\n    %s\n\n"+
					"Edit it and submit your final version, or paste it back as-is to approve:",
				draft)
			if err := emit(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message:     prompt,
			})); err != nil {
				return nil, err
			}
			return nil, workflow.ErrNodeInterrupted
		},
		workflow.NodeConfig{},
	)

	// publish: receives the human's approved/edited text on resume.
	publish := workflow.NewFunctionNode("publish",
		func(_ agent.Context, approved string) (string, error) {
			approved = strings.TrimSpace(approved)
			if approved == "" {
				return "Nothing to publish — no text approved.", nil
			}
			return "✅ Published:\n\n    " + approved, nil
		},
		workflow.NodeConfig{},
	)

	return workflowagent.New(workflowagent.Config{
		Name:        "draft_review_publish_durable",
		Description: "search-grounded draft, durable human approval, then publish",
		Edges:       workflow.Chain(workflow.Start, draftNode, review, publish),
		SubAgents:   []agent.Agent{draftAgent},
	})
}

// pending is one unanswered human-input request found in a persisted session.
type pending struct {
	id, name, message string
}

// findPending scans a session's persisted events for a workflow RequestInput
// function call ("adk_request_input") that has no matching FunctionResponse yet.
// This is the durable equivalent of the console launcher's collectPendingInterrupts,
// reconstructed from storage rather than a live event stream.
func findPending(events session.Events) (pending, bool) {
	answered := map[string]bool{}
	for ev := range events.All() {
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if fr := p.FunctionResponse; fr != nil && fr.ID != "" {
				answered[fr.ID] = true
			}
		}
	}
	for ev := range events.All() {
		if ev == nil || ev.Partial || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			fc := p.FunctionCall
			if fc == nil || fc.Name != workflow.WorkflowInputFunctionCallName || answered[fc.ID] {
				continue
			}
			msg, _ := fc.Args["message"].(string)
			return pending{id: fc.ID, name: fc.Name, message: msg}, true
		}
	}
	return pending{}, false
}

// submit runs the workflow until it pauses for human approval, then exits,
// leaving the paused state persisted in SQLite.
func submit(ctx context.Context, key, topic string) error {
	svc, err := openService()
	if err != nil {
		return err
	}
	// Start each demo from a clean slate so the fixed session ID is reusable.
	_ = svc.Delete(ctx, &session.DeleteRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if _, err := svc.Create(ctx, &session.CreateRequest{
		AppName: appName, UserID: userID, SessionID: sessionID,
	}); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	root, err := buildRootAgent(ctx, key)
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: root, SessionService: svc})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	fmt.Printf("▶ submit: %q\n  (draft node is searching + writing…)\n\n", topic)
	msg := genai.NewContentFromText(topic, genai.RoleUser)
	for _, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
	}

	// Reload the session FROM STORAGE to prove the pause persisted.
	got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return fmt.Errorf("get session: %w", err)
	}
	p, ok := findPending(got.Session.Events())
	if !ok {
		return fmt.Errorf("workflow did not pause (no pending input request found)")
	}

	fmt.Println("⏸  WORKFLOW PAUSED — awaiting human approval")
	fmt.Println("───────────────────────────────────────────")
	fmt.Println(p.message)
	fmt.Println("───────────────────────────────────────────")
	fmt.Printf("\nState saved to %s (session %q). This process can now exit.\n", dbPath, sessionID)
	fmt.Printf("Resume from a NEW process:\n  go run ./durablehitl resume \"<your approved/edited post>\"\n")
	return nil
}

// resume loads the paused session from SQLite (in a fresh process) and replies
// to the pending human-input request, letting the workflow run to completion.
func resume(ctx context.Context, key, approved string) error {
	svc, err := openService()
	if err != nil {
		return err
	}
	got, err := svc.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		return fmt.Errorf("get session (run `submit` first?): %w", err)
	}
	if got.Session == nil {
		return fmt.Errorf("no session %q found — run `submit` first", sessionID)
	}
	p, ok := findPending(got.Session.Events())
	if !ok {
		return fmt.Errorf("nothing to resume — no pending human-input request in session %q", sessionID)
	}

	fmt.Printf("▶ resume: replying to paused request %q with:\n    %s\n\n", p.id, approved)

	root, err := buildRootAgent(ctx, key)
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: root, SessionService: svc})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	// Reply to the paused node with a FunctionResponse keyed by its call ID —
	// the workflow routes it to the waiting node and delivers it to `publish`.
	msg := &genai.Content{
		Role: string(genai.RoleUser),
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       p.id,
				Name:     p.name,
				Response: map[string]any{"payload": approved},
			},
		}},
	}

	var finalOutput any
	var text strings.Builder
	for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if ev.Content == nil {
			// Terminal/function nodes carry their result on Event.Output.
			if ev.Output != nil {
				finalOutput = ev.Output
			}
			continue
		}
		for _, pt := range ev.Content.Parts {
			text.WriteString(pt.Text)
		}
	}

	if s := strings.TrimSpace(text.String()); s != "" {
		fmt.Println(s)
	}
	if finalOutput != nil {
		fmt.Println(renderOutput(finalOutput))
	}
	return nil
}

// renderOutput formats a node's Output value: strings as-is, else compact JSON.
func renderOutput(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if b, err := json.Marshal(v); err == nil {
		return string(b)
	}
	return fmt.Sprint(v)
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, `  durablehitl submit "<topic>"          run until the human-approval pause, then exit`)
	fmt.Fprintln(os.Stderr, `  durablehitl resume "<approved post>"  reload the paused session and publish`)
	os.Exit(2)
}

func main() {
	ctx := context.Background()
	key := apiKey()
	if key == "" {
		log.Fatal("set GOOGLE_API_KEY (or GEMINI_API_KEY) — get one at https://aistudio.google.com/apikey")
	}

	args := os.Args[1:]
	if len(args) < 1 {
		usage()
	}
	switch args[0] {
	case "submit":
		topic := "the launch of ADK for Go 2.0"
		if len(args) > 1 {
			topic = strings.Join(args[1:], " ")
		}
		if err := submit(ctx, key, topic); err != nil {
			log.Fatal(err)
		}
	case "resume":
		if len(args) < 2 {
			usage()
		}
		if err := resume(ctx, key, strings.Join(args[1:], " ")); err != nil {
			log.Fatal(err)
		}
	default:
		usage()
	}
}
