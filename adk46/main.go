// Command adk46 is a bit of word-play: use the ADK (Google's Agent Development
// Kit for Go) to help bag the ADK (the Adirondacks) — specifically the 46 High
// Peaks. A team of Adirondack "scout" agents fans out to research a peak in
// PARALLEL (each grounded on live web search), a "head guide" agent synthesizes
// a trip brief, and you — the hiker — approve, edit, or send it back.
//
// It runs entirely on Claude via the claudemodel adapter: each scout declares
// geminitool.GoogleSearch{}, which the adapter maps to Anthropic's web_search,
// so the whole team is grounded on current trail info.
//
//	  🗺️ route_scout ─┐
//	Start ─ ⛅ sky_watcher ─┼─ gather ─ format ─ 🏔️ head_guide ─ review (you)
//	  🎒 pack_master ─┘   (Join)  (func)     (LLM)          (HITL)
//
//	export ANTHROPIC_API_KEY=...    # or `ant auth login`
//	go run ./adk46 "Mount Marcy — a day hike this fall"
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
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/workflow"

	"adk-tinker/claudemodel"
)

const (
	appName   = "adk46"
	userID    = "hiker"
	sessionID = "trip-1"
)

// scout is one fan-out branch: an Adirondack specialist who web-searches the peak.
type scout struct {
	name, label, emoji, instruction string
}

var scouts = []scout{
	{"route_scout", "route scout", "🗺️", "You are a seasoned Adirondack route scout. Use the web search tool to find the " +
		"main hiking route for the given High Peak: trailhead, round-trip distance, elevation gain, and difficulty. " +
		"Report in 2-3 tight sentences with real numbers."},
	{"sky_watcher", "sky watcher", "⛅", "You are an Adirondack weather-watcher. Use the web search tool to find the current " +
		"season's conditions and the near-term forecast for the High Peaks region (Lake Placid / Marcy area). " +
		"In 2-3 sentences, say what a hiker should expect (temps, precip, trail conditions like mud/ice/snow)."},
	{"pack_master", "pack master", "🎒", "You are an Adirondack gear guru and safety steward. Given the peak and season, use the " +
		"web search tool if helpful, then list the essential gear and ONE concrete safety tip for this hike in 2-3 sentences."},
}

func apiKeyPresent() bool {
	return os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" ||
		fileExists(os.Getenv("HOME") + "/.config/anthropic/credentials/default.json")
}

func fileExists(p string) bool { _, err := os.Stat(p); return err == nil }

// formatReports turns the JoinNode's map[scoutName]report into one briefing for
// the head guide. Ranging the fixed slice keeps the order deterministic.
func formatReports(_ agent.Context, gathered map[string]any) (string, error) {
	var sb strings.Builder
	sb.WriteString("Scout reports from the field:\n\n")
	for _, s := range scouts {
		text := "(no report)"
		if v, ok := gathered[s.name]; ok && v != nil {
			if t := strings.TrimSpace(fmt.Sprint(v)); t != "" {
				text = t
			}
		}
		fmt.Fprintf(&sb, "%s %s:\n%s\n\n", s.emoji, s.label, text)
	}
	return sb.String(), nil
}

func buildGraph(m model.LLM) (agent.Agent, error) {
	var subAgents []agent.Agent
	var scoutNodes []workflow.Node
	for _, s := range scouts {
		a, err := llmagent.New(llmagent.Config{
			Name:        s.name,
			Model:       m,
			Description: "an Adirondack scout who researches one aspect of the hike",
			Instruction: s.instruction,
			Tools:       []tool.Tool{geminitool.GoogleSearch{}}, // → Anthropic web_search on Claude
		})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", s.name, err)
		}
		n, err := workflow.NewAgentNode(a, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
		if err != nil {
			return nil, fmt.Errorf("%s node: %w", s.name, err)
		}
		subAgents = append(subAgents, a)
		scoutNodes = append(scoutNodes, n)
	}

	gather := workflow.NewJoinNode("gather")
	format := workflow.NewFunctionNode("format", formatReports, workflow.NodeConfig{})

	headGuide, err := llmagent.New(llmagent.Config{
		Name:        "head_guide",
		Model:       m,
		Description: "the head guide who turns scout reports into a friendly trip brief",
		Instruction: "You are the head guide — a warm, slightly grizzled Adirondack 46er. Turn the scouts' field " +
			"reports into ONE motivating, practical trip brief for the hiker: a short intro, then bullet-ish lines " +
			"for Route, Conditions, and Gear/Safety, then a one-line send-off. Keep it tight and encouraging. " +
			"Ground it only in the reports you were given.",
	})
	if err != nil {
		return nil, fmt.Errorf("head_guide: %w", err)
	}
	subAgents = append(subAgents, headGuide)
	guideNode, err := workflow.NewAgentNode(headGuide, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		return nil, fmt.Errorf("head_guide node: %w", err)
	}

	// review: re-entry HITL — "yes"/Enter files the plan, "no" scraps it, else = your edit.
	rerun := true
	review := workflow.NewEmittingFunctionNode[any, string]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (string, error) {
			plan := strings.TrimSpace(fmt.Sprint(in))
			reply, err := workflow.ResumeOrRequestInput(ctx, emit, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message: fmt.Sprintf("Proposed trip plan:\n\n%s\n\n"+
					"🥾 Type 'yes' to file it (Enter works too), 'no' to scrap it, or type your own edits:", plan),
			})
			if err != nil {
				return "", err
			}
			switch strings.ToLower(strings.TrimSpace(fmt.Sprint(reply))) {
			case "", "y", "yes", "ok", "okay", "approve", "file":
				return "✅ Trip filed — happy trails, 46er-in-training! ⛰️\n\n" + plan, nil
			case "n", "no", "cancel", "scrap", "reject":
				return "🗑️  Plan scrapped — the mountain will wait.", nil
			default:
				answer := strings.TrimSpace(fmt.Sprint(reply))
				return "✅ Trip filed (your edits) — happy trails! ⛰️\n\n" + answer, nil
			}
		},
		workflow.NodeConfig{RerunOnResume: &rerun},
	)

	eb := workflow.NewEdgeBuilder()
	eb.AddFanOut(workflow.Start, scoutNodes...)
	eb.AddFanIn(gather, scoutNodes...)
	eb.Add(gather, format)
	eb.Add(format, guideNode)
	eb.Add(guideNode, review)

	return workflowagent.New(workflowagent.Config{
		Name:        "adk46_trip_planner",
		Description: "Adirondack High Peaks trip planner: scouts fan out, head guide synthesizes, hiker approves",
		Edges:       eb.Build(),
		SubAgents:   subAgents,
	})
}

func wrap(s string, width int, indent string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return indent + "(no report)"
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

type pendingInterrupt struct{ id, name string }

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
	if !apiKeyPresent() {
		log.Fatal("need Anthropic creds — set ANTHROPIC_API_KEY or run `ant auth login`")
	}

	m := claudemodel.NewModel("") // claude-opus-4-8

	root, err := buildGraph(m)
	if err != nil {
		log.Fatalf("failed to build trip planner: %v", err)
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
	var argPeak []string
	for _, a := range os.Args[1:] {
		argPeak = append(argPeak, a)
	}
	peak := strings.Join(argPeak, " ")
	if peak == "" {
		fmt.Print("🏔️  Which High Peak (and when)? ")
		line, _ := reader.ReadString('\n')
		peak = strings.TrimSpace(line)
	}
	if peak == "" {
		peak = "Mount Marcy — a day hike this fall"
	}

	fmt.Printf("\n⛰️  ADK × ADK — planning: %s\n", peak)
	fmt.Printf("   (three Adirondack scouts are searching the web in parallel…)\n")
	events, text := runTurn(ctx, r, genai.NewContentFromText("Peak/trip: "+peak, genai.RoleUser))

	fmt.Printf("\n── Scout reports (gathered in parallel) ─────────────────────\n")
	for _, s := range scouts {
		fmt.Printf("\n %s  %s\n%s\n", s.emoji, s.label, wrap(text[s.name], 74, "    "))
	}
	fmt.Printf("\n🏔️  Head guide's trip brief\n%s\n", wrap(text["head_guide"], 74, "    "))

	pend, ok := findPending(events)
	if !ok {
		fmt.Println("\n(no approval step reached)")
		return
	}

	fmt.Printf("\n────────────────────────────────────────────────────────────\n")
	fmt.Print("🥾 File this plan? [Enter = file · type edits · \"no\" = scrap]: ")
	line, _ := reader.ReadString('\n')
	answer := strings.TrimRight(line, "\r\n")

	resume := &genai.Content{
		Role: string(genai.RoleUser),
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{ID: pend.id, Name: pend.name, Response: map[string]any{"payload": answer}},
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
