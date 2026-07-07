// Command claudesearch shows a search-grounded Claude agent: an llmagent that
// declares geminitool.GoogleSearch{} — Gemini's built-in web search — running on
// Claude. claudemodel detects the GoogleSearch tool and maps it to Anthropic's
// own web_search server tool, so the SAME agent definition is grounded on either
// provider. Claude searches the web and answers with a fresh, cited fact.
//
//	# needs Anthropic creds: ANTHROPIC_API_KEY, or `ant auth login`
//	go run ./claudesearch "the latest stable Go release"
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

	topic := "the most recent Claude model Anthropic has released"
	if len(os.Args) > 1 {
		topic = strings.Join(os.Args[1:], " ")
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "grounded_writer",
		Model:       claudemodel.NewModel(""), // claude-opus-4-8
		Description: "answers with a fresh, web-grounded fact",
		Instruction: "You have a web search tool. Search the web for current information " +
			"about the user's topic, then answer in ONE sentence stating one specific, recent " +
			"fact, and cite the source domain in parentheses.",
		Tools: []tool.Tool{geminitool.GoogleSearch{}}, // ← mapped to Anthropic web_search on Claude
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	const (
		appName   = "grounded_writer"
		userID    = "user-1"
		sessionID = "session-1"
	)
	svc := session.InMemoryService()
	if _, err = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		log.Fatalf("failed to create session: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: a, SessionService: svc})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	prompt := "Topic: " + topic
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: prompt}}}

	fmt.Printf("=== search-grounded Claude ===\nTopic: %s\n(searching the web…)\n\n", topic)
	var out strings.Builder
	for event, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		if event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				out.WriteString(part.Text)
			}
		}
	}
	fmt.Println(strings.TrimSpace(out.String()))
}
