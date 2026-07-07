// Command serve exposes a HITL workflow over HTTP instead of the interactive
// console. The SAME full.NewLauncher() that powers `console` also serves a REST
// API — there is no separate "rest" keyword; the mode is `web api`:
//
//	export GOOGLE_API_KEY=...
//	go run ./serve web -port 8080 api        # REST under http://localhost:8080/api
//	go run ./serve console                    # identical wiring, interactive
//
// Flag placement is positional: -port is a `web` flag (before the `api`
// keyword); api flags (e.g. -path_prefix) come after it.
//
// Graph:  Start ─▶ draft (Gemini + Google Search) ─▶ review (HITL) ─▶ publish
//
// HITL over the wire (all under the /api prefix):
//
//	POST /api/apps/{app}/users/{user}/sessions          → {"id": "<sessionId>"}
//	POST /api/run  {appName,userId,sessionId,newMessage:{role,parts:[{text}]}}
//	     → JSON array of events; the pause event has requestedInput.interruptId
//	POST /api/run  with newMessage.parts[0].functionResponse
//	     {id:<interruptId>, name:"adk_request_input", response:{payload:"<text>"}}
//	     → resumes; an event's "output" field carries the publish result
//
// Sessions persist in SQLite (adk_sessions.db) because we wire a durable
// SessionService into launcher.Config — otherwise the web launcher defaults to
// an in-memory store that vanishes on restart.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/workflow"
)

const (
	modelName = "gemini-2.5-flash-lite"
	dbPath    = "adk_sessions.db"
)

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
		Description: "researches the topic with Google Search and drafts one social post",
		Instruction: "You are a punchy copywriter. Use the Google Search tool to ground yourself " +
			"in the topic, then write ONE short social post (max ~220 chars). Output only the post.",
		Tools: []tool.Tool{geminitool.GoogleSearch{}},
	})
	if err != nil {
		log.Fatalf("failed to create draft agent: %v", err)
	}
	draftNode, err := workflow.NewAgentNode(draftAgent, workflow.NodeConfig{RetryConfig: workflow.DefaultRetryConfig()})
	if err != nil {
		log.Fatalf("failed to create draft node: %v", err)
	}

	review := workflow.NewEmittingFunctionNode[any, any]("review",
		func(ctx agent.Context, in any, emit func(*session.Event) error) (any, error) {
			if err := emit(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "review-" + ctx.InvocationID(),
				Message:     "Approve or edit this post: " + strings.TrimSpace(fmt.Sprint(in)),
			})); err != nil {
				return nil, err
			}
			return nil, workflow.ErrNodeInterrupted
		},
		workflow.NodeConfig{},
	)

	publish := workflow.NewFunctionNode("publish",
		func(_ agent.Context, approved string) (string, error) {
			approved = strings.TrimSpace(approved)
			if approved == "" {
				return "Nothing to publish.", nil
			}
			return "✅ Published: " + approved, nil
		},
		workflow.NodeConfig{},
	)

	rootAgent, err := workflowagent.New(workflowagent.Config{
		Name:        "served_review",
		Description: "search-grounded draft, human approval over HTTP, then publish",
		Edges:       workflow.Chain(workflow.Start, draftNode, review, publish),
		SubAgents:   []agent.Agent{draftAgent},
	})
	if err != nil {
		log.Fatalf("failed to create workflow agent: %v", err)
	}

	// Durable session store so HTTP sessions survive a restart.
	svc, err := database.NewSessionService(sqlite.Open(dbPath), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		log.Fatalf("failed to open session db: %v", err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		log.Fatalf("failed to migrate session db: %v", err)
	}

	log.Printf("app %q ready — serve with:  serve web -port 8080 api   (REST under /api)", rootAgent.Name())

	l := full.NewLauncher()
	if err := l.Execute(ctx, &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(rootAgent),
		SessionService: svc,
	}, os.Args[1:]); err != nil {
		log.Fatalf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
