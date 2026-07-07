// Command a2a exposes an ADK for Go 2.0 agent over the Agent-to-Agent (A2A)
// protocol and calls it from a separate client — agent-to-agent RPC over HTTP.
//
// SERVER side (google.golang.org/adk/v2/server/adka2a/v2, package adka2a):
// wrap a normal agent.Agent in adka2a.NewExecutor, then serve two handlers on a
// mux — the agent card at a2asrv.WellKnownAgentCardPath and the JSON-RPC
// transport at /invoke. The card's SupportedInterfaces[].URL points clients at
// the JSON-RPC endpoint.
//
// CLIENT side (google.golang.org/adk/v2/agent/remoteagent/v2, package
// remoteagent): remoteagent.NewA2A(...) fetches the card and returns a plain
// agent.Agent, so you run it through runner.New/Run exactly like a local agent.
//
// Subcommands:
//
//	a2a server [port]              serve the agent over A2A (default port 8792), blocking
//	a2a client <baseURL> "<q>"     call a remote A2A agent and print its answer
//	a2a demo ["<q>"]               all-in-one: serve in-process + call it, then exit
//
// The server needs GOOGLE_API_KEY (it runs the LLM); the client does not.
//
//	export GOOGLE_API_KEY=...
//	# two real processes:
//	a2a server 8792 &
//	a2a client http://127.0.0.1:8792 "the launch of ADK for Go 2.0"
//	# or all-in-one:
//	a2a demo "the launch of ADK for Go 2.0"
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
)

const (
	modelName   = "gemini-2.5-flash-lite"
	agentPath   = "/invoke"
	defaultPort = "8792"
)

func apiKey() string {
	if k := os.Getenv("GOOGLE_API_KEY"); k != "" {
		return k
	}
	return os.Getenv("GEMINI_API_KEY")
}

// buildAgent creates the LLM agent that the A2A server exposes.
func buildAgent(ctx context.Context) agent.Agent {
	key := apiKey()
	if key == "" {
		log.Fatal("set GOOGLE_API_KEY (or GEMINI_API_KEY) — get one at https://aistudio.google.com/apikey")
	}
	model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{APIKey: key})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}
	a, err := llmagent.New(llmagent.Config{
		Name:        "post_writer",
		Model:       model,
		Description: "Writes one short social-media post about a given topic.",
		Instruction: "You are a punchy copywriter. Given a topic, write ONE short, engaging " +
			"social-media post (max ~200 chars). Output only the post text.",
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}
	return a
}

// serveA2A wraps the agent as an A2A executor and serves the agent card +
// JSON-RPC transport on the given listener. Blocks until the server stops.
func serveA2A(a agent.Agent, listener net.Listener) error {
	baseURL := &url.URL{Scheme: "http", Host: listener.Addr().String()}

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
		RunnerConfig: runner.Config{
			AppName:        a.Name(),
			Agent:          a,
			SessionService: session.InMemoryService(),
		},
	})

	mux := http.NewServeMux()
	mux.Handle(a2asrv.WellKnownAgentCardPath, a2asrv.NewStaticAgentCardHandler(card))
	mux.Handle(agentPath, a2asrv.NewJSONRPCHandler(a2asrv.NewHandler(executor)))

	log.Printf("A2A server up:\n  agent card : %s%s\n  json-rpc   : %s",
		baseURL, a2asrv.WellKnownAgentCardPath, baseURL.JoinPath(agentPath))
	return http.Serve(listener, mux)
}

// runClient builds a remote A2A agent pointing at baseURL and runs one turn
// through a local runner, printing the remote agent's answer.
func runClient(ctx context.Context, baseURL, question string) error {
	remote, err := remoteagent.NewA2A(remoteagent.A2AConfig{
		Name:              "remote_post_writer",
		AgentCardProvider: remoteagent.NewAgentCardProvider(baseURL),
	})
	if err != nil {
		return fmt.Errorf("create remote agent: %w", err)
	}

	// The remote agent is an ordinary agent.Agent — run it like a local one.
	r, err := runner.New(runner.Config{
		AppName:           "a2a_client",
		Agent:             remote,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}

	fmt.Printf("→ asking remote agent at %s: %q\n", baseURL, question)
	var out strings.Builder
	for ev, err := range r.Run(ctx, "user-1", "session-1",
		genai.NewContentFromText(question, genai.RoleUser),
		agent.RunConfig{StreamingMode: agent.StreamingModeNone}) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if ev.ErrorMessage != "" {
			return fmt.Errorf("remote error: %s", ev.ErrorMessage)
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				out.WriteString(p.Text)
			}
		}
	}
	fmt.Printf("← FINAL (from remote agent over A2A): %s\n", strings.TrimSpace(out.String()))
	return nil
}

// waitForCard polls the agent-card endpoint until the in-process server is ready.
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

func main() {
	ctx := context.Background()
	args := os.Args[1:]
	mode := "demo"
	if len(args) > 0 {
		mode = args[0]
	}

	switch mode {
	case "server":
		port := defaultPort
		if len(args) > 1 {
			port = args[1]
		}
		listener, err := net.Listen("tcp", "127.0.0.1:"+port)
		if err != nil {
			log.Fatalf("failed to bind :%s: %v", port, err)
		}
		if err := serveA2A(buildAgent(ctx), listener); err != nil {
			log.Fatalf("server stopped: %v", err)
		}

	case "client":
		if len(args) < 3 {
			log.Fatal(`usage: a2a client <baseURL> "<question>"`)
		}
		if err := runClient(ctx, args[1], strings.Join(args[2:], " ")); err != nil {
			log.Fatal(err)
		}

	case "demo":
		question := "the launch of ADK for Go 2.0"
		if len(args) > 1 {
			question = strings.Join(args[1:], " ")
		}
		listener, err := net.Listen("tcp", "127.0.0.1:0") // OS-assigned port
		if err != nil {
			log.Fatalf("failed to bind: %v", err)
		}
		baseURL := (&url.URL{Scheme: "http", Host: listener.Addr().String()}).String()
		go func() {
			if err := serveA2A(buildAgent(ctx), listener); err != nil {
				log.Printf("server stopped: %v", err)
			}
		}()
		waitForCard(baseURL)
		if err := runClient(ctx, baseURL, question); err != nil {
			log.Fatal(err)
		}

	default:
		log.Fatalf("unknown mode %q — use: server | client | demo", mode)
	}
}
