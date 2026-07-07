// Command schemahitl is an ADK for Go 2.0 HITL workflow whose pause carries a
// JSON *ResponseSchema*: the human's reply must be a structured object
// {"decision":"approve|reject|edit","text":"..."} — not free text. The engine
// validates the reply against the schema before resuming (a mismatch yields
// workflow.ErrInvalidResumeResponse and keeps the node waiting for a retry).
//
// Graph:  Start ─▶ draft (LLM) ─▶ decide (HITL, schema-validated) ─▶ (done)
//
// The decide node uses the RE-ENTRY pattern (workflow.ResumeOrRequestInput +
// NodeConfig.RerunOnResume=&true): on the first pass it emits the request and
// pauses; after the human answers, the node re-runs WITH its draft input still
// in hand AND the validated reply, so it can act on the decision (approve →
// publish the draft, edit → publish the human's text, reject → drop it).
//
// A schema-validated reply arrives as a Go map[string]any (a JSON object decoded
// into `any`), so fields are read as m["decision"].(string) / m["text"].(string).
//
// Run (console): the operator must type a full JSON OBJECT on one line.
//
//	export GOOGLE_API_KEY=...
//	go run ./schemahitl console
//
//	User  -> a post about Go generics
//	Agent -> Proposed post: ...        (pauses; prints the expected schema)
//	User  -> {"decision":"edit","text":"Generics in Go: type-safe, zero boilerplate."}
//	Agent -> ✅ Published (edited): ...
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

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

	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: key})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	draftAgent, err := llmagent.New(llmagent.Config{
		Name:        "draft",
		Model:       model,
		Description: "drafts a single social-media post from a topic",
		Instruction: "You are a punchy copywriter. Given a topic, write ONE short social-media " +
			"post (max ~200 chars). Output only the post text.",
	})
	if err != nil {
		log.Fatalf("failed to create draft agent: %v", err)
	}
	draftNode, err := workflow.NewAgentNode(draftAgent, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		log.Fatalf("failed to create draft node: %v", err)
	}

	// The schema the human's reply must satisfy: an object with a required
	// enum `decision` and an optional `text`. jsonschema.Schema uses Go fields
	// directly for validation (Type is a single string; Enum is []any).
	decisionSchema := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"decision": {
				Type:        "string",
				Enum:        []any{"approve", "reject", "edit"},
				Description: "what to do with the proposed post",
			},
			"text": {
				Type:        "string",
				Description: "the edited post; required when decision is \"edit\"",
			},
		},
		Required: []string{"decision"},
	}

	// decide: re-entry HITL node. First pass emits the schema'd request and
	// pauses; on resume it re-runs with the draft input + the validated reply.
	rerun := true
	decide := workflow.NewEmittingFunctionNode[any, string]("decide",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (string, error) {
			draft := strings.TrimSpace(fmt.Sprint(in))
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID:    "decide-" + ctx.InvocationID(),
				Message:        fmt.Sprintf("Proposed post:\n\n    %s\n\nReply with a JSON object: {\"decision\":\"approve|reject|edit\",\"text\":\"...\"}", draft),
				ResponseSchema: decisionSchema,
			})
			if err != nil {
				return "", err // ErrNodeInterrupted on the first pass
			}

			// A schema-validated reply decodes to map[string]any.
			m, _ := reply.(map[string]any)
			decision, _ := m["decision"].(string)
			text, _ := m["text"].(string)
			switch decision {
			case "approve":
				return "✅ Published (approved as drafted):\n\n    " + draft, nil
			case "edit":
				if strings.TrimSpace(text) == "" {
					return "⚠️  edit requested but no text supplied — nothing published.", nil
				}
				return "✅ Published (human-edited):\n\n    " + text, nil
			case "reject":
				return "🗑️  Rejected — nothing published.", nil
			default:
				return "⚠️  unrecognized decision: " + decision, nil
			}
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "schema_gated_review",
		Description: "LLM drafts a post; a human approves/rejects/edits it via a schema-validated decision",
		Edges:       workflow.Chain(workflow.Start, draftNode, decide),
		SubAgents:   []agent.Agent{draftAgent},
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	log.Printf("schema-gated review ready — type a topic, then reply with a JSON decision object")

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
