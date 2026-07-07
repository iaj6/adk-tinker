// Package main is a minimal, runnable "hello agent" built on ADK for Go 2.0.
//
// It constructs a single Gemini-backed conversational LLM agent, gives it one
// trivial function tool ("add") to demonstrate tool calling, and runs exactly
// one hard-coded turn, printing the agent's response to stdout.
//
// Set a Gemini API key (from https://aistudio.google.com/apikey) before running:
//
//	export GOOGLE_API_KEY=your_key_here   # or GEMINI_API_KEY
//	go run .
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// addArgs are the inputs to the "add" tool. The JSON tags become the tool's
// parameter names in the schema inferred by functiontool.
type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

// addResult is the output of the "add" tool.
type addResult struct {
	Sum int `json:"sum"`
}

// addFunc is the tool handler. Its signature matches functiontool.Func[TArgs,
// TResults]: func(agent.Context, TArgs) (TResults, error).
func addFunc(_ agent.Context, in addArgs) (addResult, error) {
	return addResult{Sum: in.A + in.B}, nil
}

// apiKey pulls a Gemini API key from either common env var.
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

	// 1. Create the Gemini model, backed by the Gemini Developer API (API key).
	model, err := gemini.NewModel(ctx, "gemini-2.5-flash-lite", &genai.ClientConfig{
		APIKey: key,
	})
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	// 2. Create one trivial function tool to demonstrate tool calling.
	addTool, err := functiontool.New(functiontool.Config{
		Name:        "add",
		Description: "Adds two integers and returns their sum.",
	}, addFunc)
	if err != nil {
		log.Fatalf("failed to create tool: %v", err)
	}

	// 3. Create the single conversational LLM agent.
	a, err := llmagent.New(llmagent.Config{
		Name:        "hello_agent",
		Model:       model,
		Description: "A minimal hello-world agent that can add two integers.",
		Instruction: "You are a friendly assistant. When the user asks you to add " +
			"numbers, call the add tool and then report the result in a short sentence.",
		Tools: []tool.Tool{
			addTool,
		},
	})
	if err != nil {
		log.Fatalf("failed to create agent: %v", err)
	}

	// 4. Wire up an in-memory session store and create a session for this run.
	const (
		appName   = "hello_agent"
		userID    = "user-1"
		sessionID = "session-1"
	)
	sessionSvc := session.InMemoryService()
	if _, err = sessionSvc.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		log.Fatalf("failed to create session: %v", err)
	}

	// 5. Build the runner that executes the agent.
	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          a,
		SessionService: sessionSvc,
	})
	if err != nil {
		log.Fatalf("failed to create runner: %v", err)
	}

	// 6. Run a single hard-coded turn.
	const prompt = "What is 2 + 3? Use the add tool to compute it."
	msg := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: prompt}},
	}

	fmt.Println("=== ADK for Go 2.0 hello agent ===")
	fmt.Printf("User:  %s\n", prompt)
	fmt.Print("Agent: ")

	// Runner.Run returns a Go 1.23+ range-over-func iterator
	// (iter.Seq2[*session.Event, error]). Each Event embeds model.LLMResponse,
	// whose Content holds the model's genai parts.
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
