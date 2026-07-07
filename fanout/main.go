// Command fanout is an ADK for Go 2.0 workflow that fans out to several
// search-grounded drafters in PARALLEL, joins their results, has an LLM editor
// pick the best, then gates on a human before publishing.
//
// Graph:
//
//	         ┌─▶ drafter_punchy    ─┐
//	Start ───┼─▶ drafter_technical ─┼─▶ gather ─▶ format ─▶ editor ─▶ review
//	         └─▶ drafter_concise   ─┘  (Join)    (func)    (LLM)    (HITL)
//
//   - fan-out: three llmagent drafter nodes wired straight from Start run
//     concurrently, each searching the web and writing in a different voice;
//   - fan-in: a JoinNode barrier waits for all three and hands the next node a
//     map[nodeName]output;
//   - a FunctionNode turns that map into one comparison prompt;
//   - an "editor" AgentNode picks the single best post;
//   - a re-entry HITL review node lets you approve / edit / discard it.
//
// Unlike the other console demos, this program uses its OWN runner + formatter
// (not the generic console launcher) so the three parallel drafts and the pick
// print as a clean, labeled report instead of one concatenated blob.
//
// Run:
//
//	export GOOGLE_API_KEY=...
//	go run ./fanout                       # prompts for a topic
//	go run ./fanout "your topic here"     # or pass one
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/workflow"
)

const modelName = "gemini-2.5-flash-lite" // Gemini 2 model: required for Google Search grounding

const (
	appName   = "fanout"
	userID    = "you"
	sessionID = "s1"
)

func apiKey() string {
	if k := os.Getenv("GOOGLE_API_KEY"); k != "" {
		return k
	}
	return os.Getenv("GEMINI_API_KEY")
}

// drafter is one fan-out branch: a distinct voice for the same topic.
type drafter struct {
	name, label, emoji, instruction string
}

var drafters = []drafter{
	{"drafter_punchy", "punchy", "✨", "You are a bold, hype-driven social copywriter. Use the Google Search " +
		"tool to find one recent, specific fact about the topic, then write ONE punchy, " +
		"exciting social post (max ~200 chars). Output only the post."},
	{"drafter_technical", "technical", "🔧", "You are a precise, technical writer for a developer audience. Use the " +
		"Google Search tool to find one concrete, recent detail about the topic, then write ONE " +
		"informative, credible social post (max ~220 chars). Output only the post."},
	{"drafter_concise", "concise", "✂️", "You are a minimalist. Use the Google Search tool to ground yourself in " +
		"the topic, then write ONE ultra-concise, clever social post (max ~120 chars). Output only the post."},
}

// formatCandidates turns the JoinNode's map[nodeName]draft into one comparison
// prompt for the editor. Ranging the fixed slice keeps the order deterministic.
func formatCandidates(_ agent.Context, gathered map[string]any) (string, error) {
	var sb strings.Builder
	sb.WriteString("Here are candidate posts on the same topic, each in a different voice.\n\n")
	for i, d := range drafters {
		text := "(no draft)"
		if v, ok := gathered[d.name]; ok && v != nil {
			if s := strings.TrimSpace(fmt.Sprint(v)); s != "" {
				text = s
			}
		}
		fmt.Fprintf(&sb, "Candidate %d [%s]:\n%s\n\n", i+1, d.label, text)
	}
	return sb.String(), nil
}

// buildGraph assembles the fan-out workflow and wraps it as an agent.
func buildGraph(m model.LLM) (agent.Agent, error) {
	var subAgents []agent.Agent
	var draftNodes []workflow.Node
	for _, d := range drafters {
		a, err := llmagent.New(llmagent.Config{
			Name:        d.name,
			Model:       m,
			Description: "writes one candidate post in a specific voice",
			Instruction: d.instruction,
			Tools:       []tool.Tool{geminitool.GoogleSearch{}},
		})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", d.name, err)
		}
		n, err := workflow.NewAgentNode(a, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
		if err != nil {
			return nil, fmt.Errorf("%s node: %w", d.name, err)
		}
		subAgents = append(subAgents, a)
		draftNodes = append(draftNodes, n)
	}

	gather := workflow.NewJoinNode("gather")
	format := workflow.NewFunctionNode("format", formatCandidates, workflow.NodeConfig{})

	editorAgent, err := llmagent.New(llmagent.Config{
		Name:        "editor",
		Model:       m,
		Description: "selects the single best social post from several candidates",
		Instruction: "You are a senior editor. You are given several candidate social posts on " +
			"the same topic. Choose the SINGLE best one (most engaging, accurate, and shareable). " +
			"You may lightly tighten it. Output only the final chosen post text — nothing else.",
	})
	if err != nil {
		return nil, fmt.Errorf("editor: %w", err)
	}
	subAgents = append(subAgents, editorAgent)
	editorNode, err := workflow.NewAgentNode(editorAgent, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		return nil, fmt.Errorf("editor node: %w", err)
	}

	// review: re-entry HITL node — "yes"/Enter approves the pick, "no" discards,
	// anything else is your edited version.
	rerun := true
	review := workflow.NewEmittingFunctionNode[any, string]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (string, error) {
			pick := strings.TrimSpace(fmt.Sprint(in))
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message:     "approve/edit/discard the editor's pick",
			})
			if err != nil {
				return "", err
			}
			answer := strings.TrimSpace(fmt.Sprint(reply))
			switch strings.ToLower(answer) {
			case "", "y", "yes", "ok", "okay", "approve", "publish":
				return "✅ Published (approved as-is):\n\n    " + pick, nil
			case "n", "no", "cancel", "skip", "reject", "discard":
				return "🗑️  Discarded — nothing published.", nil
			default:
				return "✅ Published (your edit):\n\n    " + answer, nil
			}
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)

	eb := workflow.NewEdgeBuilder()
	eb.AddFanOut(workflow.Start, draftNodes...)
	eb.AddFanIn(gather, draftNodes...)
	eb.Add(gather, format)
	eb.Add(format, editorNode)
	eb.Add(editorNode, review)

	return workflowagent.New(workflowagent.Config{
		Name:        "fanout_draft_editor_review",
		Description: "three drafters in parallel, an editor picks the best, a human approves, then publish",
		Edges:       eb.Build(),
		SubAgents:   subAgents,
	})
}

// wrap word-wraps s to width columns, indenting every line.
func wrap(s string, width int, indent string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return indent + "(no draft)"
	}
	words := strings.Fields(s)
	line := indent + words[0]
	var b strings.Builder
	for _, w := range words[1:] {
		if len(line)+1+len(w) > width {
			b.WriteString(line + "\n")
			line = indent + w
		} else {
			line += " " + w
		}
	}
	b.WriteString(line)
	return b.String()
}

// pendingInterrupt identifies the review node's paused human-input request.
type pendingInterrupt struct{ id, name string }

// findPending scans events for the workflow input FunctionCall with no reply yet.
func findPending(events []*session.Event) (pendingInterrupt, bool) {
	for _, ev := range events {
		if ev == nil || ev.Partial || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if fc := p.FunctionCall; fc != nil && fc.Name == workflow.WorkflowInputFunctionCallName {
				return pendingInterrupt{id: fc.ID, name: fc.Name}, true
			}
		}
	}
	return pendingInterrupt{}, false
}

// runTurn runs one turn and returns every event plus text collected per author.
func runTurn(ctx context.Context, r *runner.Runner, msg *genai.Content) ([]*session.Event, map[string]string) {
	byAuthor := map[string]*strings.Builder{}
	var events []*session.Event
	for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{StreamingMode: agent.StreamingModeNone}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		events = append(events, ev)
		if ev.Content == nil || ev.Partial {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.Text == "" {
				continue
			}
			if byAuthor[ev.Author] == nil {
				byAuthor[ev.Author] = &strings.Builder{}
			}
			byAuthor[ev.Author].WriteString(p.Text)
		}
	}
	text := make(map[string]string, len(byAuthor))
	for a, b := range byAuthor {
		text[a] = b.String()
	}
	return events, text
}

func main() {
	ctx := context.Background()

	key := apiKey()
	if key == "" {
		log.Fatal("set GOOGLE_API_KEY (or GEMINI_API_KEY) — get one at https://aistudio.google.com/apikey")
	}

	m, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: key})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}
	root, err := buildGraph(m)
	if err != nil {
		log.Fatalf("failed to build graph: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	reader := bufio.NewReader(os.Stdin)

	// Topic: from args (ignoring a stray "console"), else prompt.
	var argTopic []string
	for _, a := range os.Args[1:] {
		if a != "console" {
			argTopic = append(argTopic, a)
		}
	}
	topic := strings.Join(argTopic, " ")
	if topic == "" {
		fmt.Print("Topic: ")
		line, _ := reader.ReadString('\n')
		topic = strings.TrimSpace(line)
	}
	if topic == "" {
		log.Fatal("no topic given")
	}

	// Turn 1: fan out, join, edit — pauses at review.
	fmt.Printf("\n⚙  writing 3 drafts in parallel + picking the best…\n")
	events, text := runTurn(ctx, r, genai.NewContentFromText(topic, genai.RoleUser))

	fmt.Printf("\n── Candidates (written in parallel) ─────────────────────────\n")
	for _, d := range drafters {
		fmt.Printf("\n %s  %s\n%s\n", d.emoji, d.label, wrap(text[d.name], 76, "    "))
	}
	fmt.Printf("\n🏆 Editor's pick\n%s\n", wrap(text["editor"], 76, "    "))

	pend, ok := findPending(events)
	if !ok {
		fmt.Println("\n(no approval step reached)")
		return
	}

	// Human decision.
	fmt.Printf("\n────────────────────────────────────────────────────────────\n")
	fmt.Print("Approve? [Enter = publish the pick · type an edit · \"no\" = discard]: ")
	line, _ := reader.ReadString('\n')
	answer := strings.TrimRight(line, "\r\n")

	// Turn 2: resume the paused review node with the human's answer.
	resume := &genai.Content{
		Role: string(genai.RoleUser),
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       pend.id,
				Name:     pend.name,
				Response: map[string]any{"payload": answer},
			},
		}},
	}
	events2, _ := runTurn(ctx, r, resume)

	var out any
	for _, ev := range events2 {
		if ev.Output != nil {
			out = ev.Output
		}
	}
	fmt.Println()
	if s, ok := out.(string); ok {
		fmt.Println(s)
	} else if out != nil {
		fmt.Printf("%v\n", out)
	}
}

