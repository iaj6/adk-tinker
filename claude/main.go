// Command claude is the hitlgraph demo — an LLM drafts a post, a human approves
// or edits it — but running on Anthropic's Claude instead of Google's Gemini.
//
// The only change from ../hitlgraph is the model: llmagent.Config.Model is a
// claudemodel.NewModel(...) instead of gemini.NewModel(...). Everything else —
// the workflow graph, the AgentNode, the re-entry HITL review node, the console
// launcher — is identical ADK. That's the point: ADK is model-agnostic.
//
//	# needs Anthropic creds: ANTHROPIC_API_KEY, or `ant auth login`
//	go run ./claude console
//
//	User  -> a haiku about Go channels
//	Agent -> (Claude drafts a post, the workflow PAUSES)
//	User  -> yes            # approve as-is, type an edit, or "no" to discard
//	Agent -> ✅ Published (approved as-is): ...
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"

	"adk-tinker/claudemodel"
)

// modelName is any Claude model ID; empty → claude-opus-4-8.
const modelName = "" // e.g. "claude-sonnet-5" for a cheaper/faster run

func main() {
	ctx := context.Background()

	// The one Claude-specific line: a Claude-backed model.LLM.
	model := claudemodel.NewModel(modelName)

	draftAgent, err := llmagent.New(llmagent.Config{
		Name:        "draft",
		Model:       model,
		Description: "drafts a single social-media post from a topic",
		Instruction: "You are a punchy copywriter. Given a topic, write ONE short, " +
			"engaging social-media post (max ~200 chars). Output only the post text — " +
			"no preamble, no quotes.",
	})
	if err != nil {
		log.Fatalf("failed to create draft agent: %v", err)
	}
	draftNode, err := workflow.NewAgentNode(draftAgent, workflow.NodeConfig{})
	if err != nil {
		log.Fatalf("failed to create draft node: %v", err)
	}

	// Re-entry HITL review node: yes/Enter approves, no discards, else = your edit.
	rerun := true
	review := workflow.NewEmittingFunctionNode[any, string]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (string, error) {
			draft := strings.TrimSpace(fmt.Sprint(in))
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message: fmt.Sprintf("Proposed post (via Claude):\n\n    %s\n\n"+
					"👉 Type 'yes' to publish as-is (Enter works too), 'no' to discard, "+
					"or type an edited version to publish that instead:", draft),
			})
			if err != nil {
				return "", err
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

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "claude_draft_review",
		Description: "Claude drafts a post, then a human approves, edits, or discards it",
		Edges:       workflow.Chain(workflow.Start, draftNode, review),
		SubAgents:   []agent.Agent{draftAgent},
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("Claude-backed draft→review ready (model: %s)", firstNonEmpty(modelName, "claude-opus-4-8"))

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
