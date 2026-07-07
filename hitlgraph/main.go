// Command hitlgraph is a graph-based ADK for Go 2.0 workflow with a
// human-in-the-loop (HITL) pause.
//
// The graph is a two-node chain:
//
//	Start ─▶ draft ─▶ review
//
//	draft    an LLM agent node (Gemini): turns the topic you type into a
//	         one-sentence social post.
//	review   a HITL node: it shows you the draft and PAUSES the whole workflow
//	         until you answer. Type "yes" (or just Enter) to publish it as-is,
//	         "no" to discard it, or type an edited version to publish that.
//
// The pause is the interesting part: review emits a RequestInput and, on the
// first pass, returns workflow.ErrNodeInterrupted, which suspends the run. It is
// a re-entry node (NodeConfig.RerunOnResume=&true), so after you answer it
// re-runs WITH the draft still in hand plus your reply, and decides what to do.
//
// Run it (interactive):
//
//	export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
//	go run ./hitlgraph console
//
//	User  -> a haiku about Go channels          # type a topic, press Enter
//	Agent -> (draft appears, workflow PAUSES)
//	User  -> yes                                 # approve as-is (or type an edit, or "no")
//	Agent -> ✅ Published: ...
//	User  ->                                     # empty line / Ctrl-D to quit
//
// Note: each line you type at "User ->" when nothing is pending starts a fresh
// run on a new topic — that is just the console read-eval-print loop.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

const modelName = "gemini-2.5-flash-lite"

// apiKey pulls a Gemini API key from either common env var.
func apiKey() string {
	if k := os.Getenv("GOOGLE_API_KEY"); k != "" {
		return k
	}
	return os.Getenv("GEMINI_API_KEY")
}

func main() {
	ctx := context.Background()

	key := apiKey()
	if key == "" {
		log.Fatal("set GOOGLE_API_KEY (or GEMINI_API_KEY) — get one at https://aistudio.google.com/apikey")
	}

	// 1. Create the Gemini model, backed by the Gemini Developer API (API key).
	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{
		APIKey: key,
	})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	// --- Node 1: the LLM "draft" agent. As the first node after Start it
	// receives the user's typed topic as its single-turn input, and emits a
	// one-sentence post.
	draftAgent, err := llmagent.New(llmagent.Config{
		Name:        "draft",
		Model:       model,
		Description: "drafts a single social-media post from a topic",
		Instruction: "You are a punchy copywriter. Given a topic, write ONE short, " +
			"engaging social-media post (max ~200 chars). Output only the post text — " +
			"no preamble, no quotes, no hashtags unless they fit naturally.",
	})
	if err != nil {
		log.Fatalf("failed to create draft agent: %v", err)
	}
	draftNode, err := workflow.NewAgentNode(draftAgent, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		log.Fatalf("failed to create draft node: %v", err)
	}

	// --- Node 2: the human-in-the-loop pause + decision. This is a re-entry
	// node (NodeConfig.RerunOnResume=&true): on the first pass it shows the
	// draft and PAUSES (ResumeOrRequestInput returns ErrNodeInterrupted); after
	// you answer, the node re-runs WITH the draft still in hand plus your reply,
	// and interprets it — "yes"/Enter approves the draft, "no" discards it, and
	// anything else is treated as your edited version.
	rerun := true
	review := workflow.NewEmittingFunctionNode[any, string]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (string, error) {
			draft := strings.TrimSpace(fmt.Sprint(in))
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message: fmt.Sprintf("Proposed post:\n\n    %s\n\n"+
					"👉 Type 'yes' to publish as-is (Enter works too), 'no' to discard, "+
					"or type an edited version to publish that instead:", draft),
			})
			if err != nil {
				return "", err // ErrNodeInterrupted on the first pass — this IS the pause
			}

			answer := strings.TrimSpace(fmt.Sprint(reply))
			switch strings.ToLower(answer) {
			case "", "y", "yes", "ok", "okay", "approve", "publish":
				return "✅ Published (approved as-is):\n\n    " + draft, nil
			case "n", "no", "cancel", "skip", "reject", "discard":
				return "🗑️  Discarded — nothing published.", nil
			default:
				return "✅ Published (your edit):\n\n    " + answer, nil
			}
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)

	// Wire the graph: Start ─▶ draft ─▶ review.
	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "draft_review_publish",
		Description: "an LLM drafts a post, then a human approves, edits, or discards it",
		Edges:       workflow.Chain(workflow.Start, draftNode, review),
		// Register the wrapped LLM agent so the runner can resolve event authors.
		SubAgents: []agent.Agent{draftAgent},
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("draft→review ready — type a topic, then approve (yes), edit, or discard (no)")

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
