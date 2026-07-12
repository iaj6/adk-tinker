// Command rangerguide is the cross-provider A2A mesh (see ../a2amesh) wearing an
// Adirondack hat: a Claude "trail guide" consults a Gemini "park ranger" — the
// regulations expert — over the A2A protocol, then gives the hiker friendly advice.
//
//	Claude trail_guide (claude-opus-4-8)
//	        │  calls the "park_ranger" tool
//	        ▼
//	agenttool ──A2A/JSON-RPC──▶ Gemini park_ranger (gemini-3.5-flash), served over A2A
//	        ▲                            answers the regs question
//	        └──────────── answer ────────┘   → the guide turns it into advice
//
//	export GOOGLE_API_KEY=...     # park ranger  (+ Anthropic creds for the guide)
//	go run ./rangerguide "can I have a campfire near Lake Colden and do I need a bear canister?"
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
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"

	"adk-tinker/claudemodel"
)

const geminiModel = "gemini-3.5-flash" // pick a model bucket with free daily quota remaining

func main() {
	ctx := context.Background()

	geminiKey := os.Getenv("GOOGLE_API_KEY")
	if geminiKey == "" {
		geminiKey = os.Getenv("GEMINI_API_KEY")
	}
	if geminiKey == "" {
		log.Fatal("set GOOGLE_API_KEY for the Gemini park ranger")
	}

	question := "I want to backcountry camp near Lake Colden this weekend with a group of six — what rules should I know about campfires, bear canisters, and group size?"
	if len(os.Args) > 1 {
		question = strings.Join(os.Args[1:], " ")
	}

	// --- Gemini park ranger, served over A2A ---
	gm, err := gemini.NewModel(ctx, geminiModel, &genai.ClientConfig{APIKey: geminiKey})
	if err != nil {
		log.Fatalf("gemini model: %v", err)
	}
	ranger, err := llmagent.New(llmagent.Config{
		Name:        "park_ranger",
		Model:       gm,
		Description: "A Gemini-powered Adirondack park ranger. Ask it a backcountry-regulations question; it answers precisely.",
		Instruction: "You are an Adirondack High Peaks Wilderness park ranger. Answer the backcountry-regulations " +
			"question factually and concisely (2-3 sentences): campfire rules, bear-canister requirements, group-size " +
			"limits, camping setbacks, etc. If a rule is region-specific, say so.",
	})
	if err != nil {
		log.Fatalf("ranger agent: %v", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatalf("bind: %v", err)
	}
	baseURL := (&url.URL{Scheme: "http", Host: listener.Addr().String()}).String()
	go serveA2A(ranger, listener)
	waitForCard(baseURL)
	log.Printf("park_ranger (%s) serving over A2A at %s", geminiModel, baseURL)

	// --- Claude trail guide: remote ranger → agenttool → tool ---
	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "park_ranger",
		AgentCardProvider: remoteagent.NewAgentCardProvider(baseURL),
	})
	if err != nil {
		log.Fatalf("remote agent: %v", err)
	}
	rangerTool := agenttool.New(remote, &agenttool.Config{})

	guide, err := llmagent.New(llmagent.Config{
		Name:        "trail_guide",
		Model:       claudemodel.NewModel(""), // claude-opus-4-8
		Description: "a friendly Adirondack trail guide who checks the regs with a ranger before advising",
		Instruction: "You are a warm, experienced Adirondack trail guide. You have a tool named park_ranger — a " +
			"regulations expert running on Google's Gemini. For any question about backcountry rules, you MUST call " +
			"park_ranger, then relay its answer to the hiker as friendly, practical advice, noting you checked with the " +
			"ranger (over A2A) to be sure.",
		Tools: []tool.Tool{rangerTool},
	})
	if err != nil {
		log.Fatalf("guide agent: %v", err)
	}

	// Dev UI / interactive console: chat with the guide in the browser; it
	// consults the Gemini ranger (still served in the goroutine above) over A2A.
	//   go run ./rangerguide web -port 8793 webui -api_server_address http://localhost:8793/api api
	if len(os.Args) > 1 && (os.Args[1] == "web" || os.Args[1] == "console") {
		l := full.NewLauncher()
		if err := l.Execute(ctx, &launcher.Config{AgentLoader: agent.NewSingleLoader(guide)}, os.Args[1:]); err != nil {
			log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
		}
		return
	}

	// --- Run one turn and trace the cross-provider hop ---
	const appName, userID, sessionID = "rangerguide", "hiker", "s1"
	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		log.Fatalf("session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: guide, SessionService: svc})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	fmt.Printf("\n=== 🥾 Adirondack ranger↔guide mesh ===\nHiker asks the guide: %s\n\n--- trace ---\n", question)
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
				fmt.Printf("  🧑‍✈️ %s (Claude) → A2A call to %s\n", ev.Author, p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Printf("  🎒 %s ← regs from the Gemini ranger\n", p.FunctionResponse.Name)
			case p.Text != "":
				t := strings.TrimSpace(p.Text)
				if ev.Author == "trail_guide" {
					final.WriteString(t)
				} else {
					fmt.Printf("  🟢 %s: %s\n", ev.Author, t)
				}
			}
		}
	}
	fmt.Printf("\n--- 🥾 the guide's advice ---\n%s\n", strings.TrimSpace(final.String()))
}

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
