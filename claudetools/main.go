// Command claudetools is the hello-agent's function-tool demo — a Gemini agent
// that answers "what is 2 + 3?" by calling an `add` Go function — but running on
// Claude via claudemodel. It proves the adapter's tool-calling path: Claude emits
// a tool_use, ADK executes addFunc, and the tool_result feeds the final answer.
//
//	# needs Anthropic creds: ANTHROPIC_API_KEY, or `ant auth login`
//	go run ./claudetools
//
// Expected: the agent calls add(2, 3) and reports 5.
package main

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	"adk-tinker/claudemodel"
)

type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

type addResult struct {
	Sum int `json:"sum"`
}

func addFunc(_ agent.Context, in addArgs) (addResult, error) {
	log.Printf("[tool] add(%d, %d) called by Claude", in.A, in.B)
	return addResult{Sum: in.A + in.B}, nil
}

func main() {
	ctx := context.Background()

	// Claude instead of Gemini — the only provider-specific line.
	model := claudemodel.NewModel("")

	addTool, err := functiontool.New(functiontool.Config{
		Name:        "add",
		Description: "Adds two integers and returns their sum.",
	}, addFunc)
	if err != nil {
		log.Fatalf("failed to create tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "claude_calculator",
		Model:       model,
		Description: "answers arithmetic questions using the add tool",
		Instruction: "You are a helpful assistant. When the user asks you to add numbers, " +
			"call the add tool and then report the result in a short sentence.",
		Tools: []tool.Tool{addTool},
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	const (
		appName   = "claude_calculator"
		userID    = "user-1"
		sessionID = "session-1"
	)
	sessionSvc := session.InMemoryService()
	if _, err = sessionSvc.Create(ctx, &session.CreateRequest{
		AppName: appName, UserID: userID, SessionID: sessionID,
	}); err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          a,
		SessionService: sessionSvc,
	})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	const prompt = "What is 2 + 3? Use the add tool to compute it."
	msg := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: prompt}}}

	fmt.Println("=== add tool on Claude ===")
	fmt.Printf("User:  %s\n", prompt)
	fmt.Print("Agent: ")
	for event, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run failed: %v", err)
		}
		if event.Content == nil {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				fmt.Print(part.Text)
			}
		}
	}
	fmt.Println()
}
