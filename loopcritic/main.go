// Command loopcritic is a self-critique loop: an Adirondack trip planner whose
// plan does NOT leave base camp until an internal safety critic signs off.
//
// It uses ADK's loop agent (agent/workflowagents/loopagent), which runs its
// sub-agents in sequence and repeats up to MaxIterations — or until a sub-agent
// "escalates". Two sub-agents alternate each iteration:
//
//	┌──────────────────────────── loop (≤ N iterations) ─────────────────────────┐
//	│  ✍️  planner  → drafts / revises the trip plan (sees any prior critique)     │
//	│  🔍 safety_critic → PASS? call exit_loop (escalate → stop).  FAIL? critique. │
//	└─────────────────────────────────────────────────────────────────────────────┘
//
// The critic holds the exit_loop tool (tool/exitlooptool); calling it sets
// Actions().Escalate = true, which the loop agent detects to stop early. That
// tool call runs through the Claude model.LLM adapter's tool-calling path.
//
// This is the counterpart to eval/: there, a judge grades an agent from the
// OUTSIDE; here, the critic lives INSIDE the agent so it self-corrects until it
// passes. Runs entirely on Claude.
//
//	export ANTHROPIC_API_KEY=...      # or `ant auth login`
//	go run ./loopcritic "Mount Marcy, day hike, this weekend"
//	go run ./loopcritic web -port 8793 webui -api_server_address http://localhost:8793/api api
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
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/exitlooptool"

	"adk-tinker/claudemodel"
)

const (
	appName    = "loopcritic"
	userID     = "hiker"
	sessionID  = "s1"
	plannerName = "planner"
	criticName  = "safety_critic"
	maxRounds   = 4
)

func apiKeyPresent() bool {
	return os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" ||
		func() bool { _, err := os.Stat(os.Getenv("HOME") + "/.config/anthropic/credentials/default.json"); return err == nil }()
}

func buildLoop() (agent.Agent, error) {
	m := claudemodel.NewModel("")

	exitTool, err := exitlooptool.New()
	if err != nil {
		return nil, fmt.Errorf("exit_loop tool: %w", err)
	}

	planner, err := llmagent.New(llmagent.Config{
		Name:        plannerName,
		Model:       m,
		Description: "drafts and revises an Adirondack trip plan",
		Instruction: "You are an Adirondack High Peaks trip planner working in a draft-and-refine loop with a safety " +
			"critic. Read the conversation so far.\n" +
			"- If the critic has NOT yet given feedback (this is your first pass), sketch a QUICK, rough outline: the " +
			"peak, the general route, and a couple of high-level notes. Keep it short — do not try to be complete yet.\n" +
			"- If the critic HAS given feedback, produce a COMPLETE, improved trip plan that fixes every numbered point " +
			"they raised, structured with: Peak & Route (specific trailhead, round-trip distance, elevation gain), " +
			"Conditions (weather/season contingency), Gear (essentials), and a Safety plan (explicit turnaround time " +
			"and bailout option).\n" +
			"Output ONLY the trip plan — no preamble, no commentary about revising.",
	})
	if err != nil {
		return nil, fmt.Errorf("planner: %w", err)
	}

	critic, err := llmagent.New(llmagent.Config{
		Name:        criticName,
		Model:       m,
		Description: "a stern Adirondack safety critic who must approve a plan before it is released",
		Instruction: "You are a stern Adirondack safety critic reviewing the trip plan just proposed. A plan PASSES " +
			"only if it includes ALL of: (1) a specific trailhead and route; (2) real round-trip distance AND " +
			"elevation-gain numbers; (3) a weather/conditions contingency; (4) essential gear; and (5) an explicit " +
			"turnaround time AND a bailout option. If ANY are missing or vague, do NOT approve: reply with a short, " +
			"numbered list of exactly what to fix, and do not call any tool. ONLY if the plan satisfies all five, " +
			"reply 'APPROVED' and then call the exit_loop tool to release the plan. Be strict — a hiker's safety " +
			"depends on it.",
		Tools: []tool.Tool{exitTool},
	})
	if err != nil {
		return nil, fmt.Errorf("critic: %w", err)
	}

	return loopagent.New(loopagent.Config{
		MaxIterations: maxRounds,
		AgentConfig: agent.Config{
			Name:        "safety_gated_planner",
			Description: "an Adirondack trip planner that self-critiques until a safety critic approves",
			SubAgents:   []agent.Agent{planner, critic},
		},
	})
}

func wrap(s, indent string, width int) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return indent + "(empty)"
	}
	var b strings.Builder
	for _, para := range strings.Split(s, "\n") {
		words := strings.Fields(para)
		if len(words) == 0 {
			b.WriteString("\n")
			continue
		}
		line := indent + words[0]
		for _, w := range words[1:] {
			if len(line)+1+len(w) > width {
				b.WriteString(line + "\n")
				line = indent + w
			} else {
				line += " " + w
			}
		}
		b.WriteString(line + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func main() {
	ctx := context.Background()
	if !apiKeyPresent() {
		log.Fatal("need Anthropic creds — set ANTHROPIC_API_KEY or run `ant auth login`")
	}

	root, err := buildLoop()
	if err != nil {
		log.Fatalf("failed to build loop: %v", err)
	}

	// Dev UI / interactive console.
	if len(os.Args) > 1 && (os.Args[1] == "web" || os.Args[1] == "console") {
		l := full.NewLauncher()
		if err := l.Execute(ctx, &launcher.Config{AgentLoader: agent.NewSingleLoader(root)}, os.Args[1:]); err != nil {
			log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
		}
		return
	}

	topic := "Mount Marcy — a day hike this weekend"
	if len(os.Args) > 1 {
		topic = strings.Join(os.Args[1:], " ")
	}

	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		log.Fatalf("session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: root, SessionService: svc})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	fmt.Printf("⛰️  self-critique trip planner — %q\n   (the plan won't leave base camp until the safety critic signs off…)\n", topic)

	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "Plan a trip: " + topic}}}
	drafts := 0
	approved := false
	latestPlan := ""
	for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == "exit_loop" {
				approved = true
				fmt.Printf("\n  ✅ safety critic: APPROVED — plan clears base camp.\n")
				continue
			}
			text := strings.TrimSpace(p.Text)
			if text == "" {
				continue
			}
			switch ev.Author {
			case plannerName:
				drafts++
				latestPlan = text
				fmt.Printf("\n  ✍️  draft v%d (planner):\n%s\n", drafts, wrap(text, "      ", 74))
			case criticName:
				fmt.Printf("\n  🔍 safety critic:\n%s\n", wrap(text, "      ", 74))
			}
		}
	}

	fmt.Printf("\n────────────────────────────────────────────────────────────\n")
	if approved {
		fmt.Printf("🏔️  Approved trip plan (after %d draft%s):\n\n%s\n", drafts, plural(drafts), wrap(latestPlan, "  ", 76))
	} else {
		fmt.Printf("⚠️  Hit %d rounds without the critic approving — latest draft (use with caution):\n\n%s\n", maxRounds, wrap(latestPlan, "  ", 76))
	}
}

func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}
