// Command a2amesh is a CROSS-PROVIDER agent mesh: a Claude orchestrator delegates
// a factual question to a Gemini specialist over the A2A protocol.
//
//	Claude orchestrator (claude-opus-4-8)
//	        │  calls the "gemini_specialist" tool
//	        ▼
//	agenttool  ──A2A/JSON-RPC──▶  Gemini specialist (gemini-3.5-flash)
//	        ▲                            served over A2A on localhost
//	        └──────────── answer ────────┘
//	then Claude synthesizes the final reply.
//
// It ties together everything built earlier: the Claude model.LLM adapter, its
// tool-calling (the agent-as-tool uses the Parameters JSON-Schema path that was
// the blocker fix), and the A2A server/client.
//
//	export GOOGLE_API_KEY=...     # Gemini specialist
//	# + Anthropic creds (ANTHROPIC_API_KEY or `ant auth login`) for the orchestrator
//	go run ./a2amesh "your factual question"
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/remoteagent/v2"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"

	"adk-tinker/claudemodel"
)

const geminiModel = "gemini-3.5-flash" // current-gen model, its own fresh daily quota bucket

func main() {
	ctx := context.Background()

	geminiKey := os.Getenv("GOOGLE_API_KEY")
	if geminiKey == "" {
		geminiKey = os.Getenv("GEMINI_API_KEY")
	}
	if geminiKey == "" {
		log.Fatal("set GOOGLE_API_KEY for the Gemini specialist")
	}

	question := "What is the tallest known mountain in the solar system, on which body is it, and roughly how tall is it?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}

	// --- Gemini specialist, served over A2A ---
	gm, err := gemini.NewModel(ctx, geminiModel, &genai.ClientConfig{APIKey: geminiKey})
	if err != nil {
		log.Fatalf("gemini model: %v", err)
	}
	specialist, err := llmagent.New(llmagent.Config{
		Name:        "gemini_specialist",
		Model:       gm,
		Description: "A Gemini-powered fact specialist. Give it a factual question; it answers in one precise sentence.",
		Instruction: "You are a precise fact specialist. Answer the question in ONE concise, factual sentence.",
	})
	if err != nil {
		log.Fatalf("specialist agent: %v", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("bind: %v", err)
	}
	baseURL := (&url.URL{Scheme: "http", Host: listener.Addr().String()}).String()
	go serveA2A(specialist, listener)
	waitForCard(baseURL)
	log.Printf("gemini_specialist (%s) serving over A2A at %s", geminiModel, baseURL)

	// --- Claude orchestrator: remote Gemini agent → agenttool → tool ---
	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "gemini_specialist",
		AgentCardProvider: remoteagent.NewAgentCardProvider(baseURL),
	})
	if err != nil {
		log.Fatalf("remote agent: %v", err)
	}
	specialistTool := agenttool.New(remote, &agenttool.Config{})

	orchestrator, err := llmagent.New(llmagent.Config{
		Name:        "claude_orchestrator",
		Model:       claudemodel.NewModel(""), // claude-opus-4-8
		Description: "delegates factual questions to a specialist and reports the answer",
		Instruction: "You coordinate a team. You have a tool named gemini_specialist — an expert " +
			"agent running on Google's Gemini. For any factual question, you MUST call gemini_specialist " +
			"with the question, then present its answer to the user in a friendly sentence, explicitly " +
			"noting that you consulted a Gemini agent over A2A.",
		Tools: []tool.Tool{specialistTool},
	})
	if err != nil {
		log.Fatalf("orchestrator agent: %v", err)
	}

	// --- Run one turn and trace the cross-provider hop ---
	const appName, userID, sessionID = "a2amesh", "user-1", "session-1"
	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		log.Fatalf("session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: orchestrator, SessionService: svc})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	fmt.Printf("\n=== cross-provider A2A mesh ===\nQ (to Claude): %s\n\n--- trace ---\n", question)
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: question}}}
	var final strings.Builder
	for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Printf("  🔵 %s (Claude) → A2A call to %s\n", ev.Author, p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Printf("  🟡 %s ← answer from Gemini specialist\n", p.FunctionResponse.Name)
			case p.Text != "":
				t := strings.TrimSpace(p.Text)
				if ev.Author == "claude_orchestrator" {
					final.WriteString(t)
				} else {
					fmt.Printf("  🟢 %s: %s\n", ev.Author, t)
				}
			}
		}
	}
	fmt.Printf("\n--- Claude's final answer ---\n%s\n", strings.TrimSpace(final.String()))
}

// serveA2A exposes an agent over A2A (agent card + JSON-RPC) on the given listener.
func serveA2A(a agent.Agent, listener net.Listener) {
	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}
	const agentPath = "/invoke"
	card := &a2a.AgentCard{
		Name:        a.Name(),
		Description: a.Description(),
		SupportedInterfaces: []*a2a.AgentInterface{{
			URL:             baseURL.JoinPath(agentPath).String(),
			ProtocolBinding: a2a.TransportProtocolJSONRPC,
			ProtocolVersion: a2a.Version,
		}},
		Version:            "1.0.0",
		DefaultInputModes:  []string{"text/plain"},
		DefaultOutputModes: []string{"text/plain"},
		Skills:             adka2a.BuildAgentSkills(a),
		Capabilities:       a2a.AgentCapabilities{Streaming: true},
	}
	executor := adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{AppName: a.Name(), Agent: a, SessionService: session.InMemoryService()},
	})
	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))
	_ = http.Serve(listener, mux)
}

func waitForCard(baseURL string) {
	for i := 0; i < 50; i++ {
		if resp, err := http.Get(baseURL + a2asrv.WellKnownAgentCardPath); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
}
