// Command whichpeak is a search-grounded "Trailhead Oracle": tell it your fitness
// and how much time you have, and a Claude agent checks the current High Peaks
// forecast (via web search) and recommends ONE Adirondack High Peak to hike today,
// with reasoning. It runs on Claude — the geminitool.GoogleSearch{} tool is mapped
// to Anthropic's web_search by claudemodel.
//
//	export ANTHROPIC_API_KEY=...   # or `ant auth login`
//	go run ./whichpeak "beginner, half a day, hate exposure"
//	go run ./whichpeak "strong hiker, full day, want a challenge"
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
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"

	"adk-tinker/claudemodel"
)

func main() {
	ctx := context.Background()

	profile := "moderate fitness, a full day, comfortable with some scrambling"
	if len(os.Args) > 1 {
		profile = strings.Join(os.Args[1:], " ")
	}

	oracle, err := llmagent.New(llmagent.Config{
		Name:        "trailhead_oracle",
		Model:       claudemodel.NewModel(""),
		Description: "recommends an Adirondack High Peak based on the hiker and the forecast",
		Instruction: "You are the Trailhead Oracle for the Adirondack High Peaks. Given the hiker's fitness and " +
			"available time, use the web search tool to check the current High Peaks region forecast and any trail " +
			"advisories, then recommend ONE specific High Peak (or, for beginners/short days, an accessible pick like " +
			"Cascade or Phelps) that fits. Answer in one tight paragraph: the peak, why it fits their profile, and one " +
			"line on today's conditions. Be encouraging and practical.",
		Tools: []tool.Tool{geminitool.GoogleSearch{}}, // → Anthropic web_search on Claude
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	const appName, userID, sessionID = "whichpeak", "hiker", "s1"
	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		log.Fatalf("session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: oracle, SessionService: svc})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	fmt.Printf("🔮 Trailhead Oracle\nHiker: %s\n(checking the forecast…)\n\n", profile)
	msg := genai.NewContentFromText("Hiker profile: "+profile, genai.RoleUser)
	var out strings.Builder
	for ev, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				out.WriteString(p.Text)
			}
		}
	}
	fmt.Println(strings.TrimSpace(out.String()))
}
