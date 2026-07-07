// Package claudemodel implements ADK for Go 2.0's model.LLM interface backed by
// Anthropic's Claude models (via the official github.com/anthropics/anthropic-sdk-go).
//
// ADK is model-agnostic: an agent takes any model.LLM, and the built-in
// google.golang.org/adk/v2/model/gemini is just one implementation. This package
// is a second one — it translates ADK's genai-shaped request/response into
// Anthropic's Messages API and back, so the *same* ADK agents and graphs run on
// Claude instead of Gemini.
//
//	model := claudemodel.NewModel("")                 // defaults to claude-opus-4-8
//	agent, _ := llmagent.New(llmagent.Config{Model: model, ...})
//
// Supports text generation, system instruction, AND function/tool calling — so
// an llmagent with functiontool tools (e.g. an "add" tool) works: Claude emits a
// tool_use, ADK executes the Go function, and the tool_result is fed back for the
// final answer. Gemini's built-in server tools (e.g. geminitool.GoogleSearch)
// have no Claude equivalent through this path and are ignored.
//
// Auth: uses the Anthropic SDK's default credential resolution — set
// ANTHROPIC_API_KEY, or log in with `ant auth login` (the SDK reads the profile).
package claudemodel

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// maxTokens caps the response. Generous headroom while staying well under SDK
// HTTP timeouts (non-streaming).
const maxTokens = 4096

type claudeModel struct {
	client anthropic.Client
	model  anthropic.Model
}

// NewModel returns a model.LLM backed by Claude. modelName is any Claude model
// ID (e.g. "claude-sonnet-5"); empty defaults to claude-opus-4-8. Extra SDK
// options (e.g. option.WithAPIKey) are forwarded to the client.
func NewModel(modelName string, opts ...option.RequestOption) model.LLM {
	mdl := anthropic.ModelClaudeOpus4_8
	if modelName != "" {
		mdl = anthropic.Model(modelName)
	}
	return &claudeModel{
		client: anthropic.NewClient(opts...),
		model:  mdl,
	}
}

func (m *claudeModel) Name() string { return string(m.model) }

// GenerateContent implements model.LLM. It always yields a single, complete
// (non-partial) response — the runner and workflow engine handle that fine — so
// we ignore the stream flag and use the non-streaming Messages API.
func (m *claudeModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

func (m *claudeModel) modelFor(req *model.LLMRequest) anthropic.Model {
	if req.Model != "" { // may be overridden by a BeforeModelCallback
		return anthropic.Model(req.Model)
	}
	return m.model
}

func (m *claudeModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	params := anthropic.MessageNewParams{
		Model:     m.modelFor(req),
		MaxTokens: maxTokens,
		Messages:  toAnthropicMessages(req.Contents),
	}
	if req.Config != nil {
		// genai's Config.SystemInstruction is a *genai.Content — flatten its text
		// into Anthropic's top-level system prompt.
		if req.Config.SystemInstruction != nil {
			if sys := partsText(req.Config.SystemInstruction.Parts); strings.TrimSpace(sys) != "" {
				params.System = []anthropic.TextBlockParam{{Text: sys}}
			}
		}
		if tools := toAnthropicTools(req.Config); len(tools) > 0 {
			params.Tools = tools
		}
	}

	resp, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	// A server-side tool (web_search) may pause the turn when its internal loop
	// hits its iteration cap; re-send so the server resumes, bounded to be safe.
	for i := 0; i < 5 && resp.StopReason == anthropic.StopReasonPauseTurn; i++ {
		params.Messages = append(params.Messages, resp.ToParam())
		if resp, err = m.client.Messages.New(ctx, params); err != nil {
			return nil, fmt.Errorf("failed to resume model: %w", err)
		}
	}

	// Shape the reply back into the genai.Content the ADK runtime expects, with
	// the "model" role. Text blocks become text parts; tool_use blocks become
	// genai FunctionCall parts (keyed by the Anthropic tool-use ID) so the ADK
	// runner executes the Go tool and loops.
	var parts []*genai.Part
	for _, block := range resp.Content {
		switch b := block.AsAny().(type) {
		case anthropic.TextBlock:
			if b.Text != "" {
				parts = append(parts, &genai.Part{Text: b.Text})
			}
		case anthropic.ToolUseBlock:
			var args map[string]any
			if len(b.Input) > 0 {
				_ = json.Unmarshal(b.Input, &args)
			}
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   b.ID,
				Name: b.Name,
				Args: args,
			}})
		}
	}
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}
	return &model.LLMResponse{
		Content:      &genai.Content{Role: "model", Parts: parts},
		ModelVersion: string(resp.Model),
	}, nil
}

// toAnthropicMessages converts the genai conversation into Anthropic messages.
// genai role "model" maps to Anthropic "assistant"; everything else to "user".
// Each part maps to the corresponding block type:
//   - Text            → text block
//   - FunctionCall    → tool_use block (on a "model"/assistant turn)
//   - FunctionResponse→ tool_result block (on a "user" turn)
//
// IDs thread through unchanged so Anthropic pairs each tool_result with its
// tool_use across the multi-turn tool loop.
func toAnthropicMessages(contents []*genai.Content) []anthropic.MessageParam {
	var msgs []anthropic.MessageParam
	for _, c := range contents {
		if c == nil {
			continue
		}
		var blocks []anthropic.ContentBlockParamUnion
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			switch {
			case p.FunctionCall != nil:
				fc := p.FunctionCall
				args := fc.Args
				if args == nil {
					args = map[string]any{}
				}
				blocks = append(blocks, anthropic.NewToolUseBlock(fc.ID, args, fc.Name))
			case p.FunctionResponse != nil:
				fr := p.FunctionResponse
				blocks = append(blocks, anthropic.NewToolResultBlock(fr.ID, funcResponseString(fr), false))
			case p.Text != "":
				blocks = append(blocks, anthropic.NewTextBlock(p.Text))
			}
		}
		if len(blocks) == 0 {
			continue
		}
		role := anthropic.MessageParamRoleUser
		if c.Role == "model" {
			role = anthropic.MessageParamRoleAssistant
		}
		// Coalesce consecutive same-role turns. Anthropic requires strict
		// user/assistant alternation, but ADK — targeting Gemini, which tolerates
		// repeats — can emit two same-role turns in a row (e.g. a peer agent's
		// reply rewritten to a "user" turn, then the real user turn, in a
		// multi-agent graph). Merging their blocks preserves order and content.
		if n := len(msgs); n > 0 && msgs[n-1].Role == role {
			msgs[n-1].Content = append(msgs[n-1].Content, blocks...)
			continue
		}
		if role == anthropic.MessageParamRoleAssistant {
			msgs = append(msgs, anthropic.NewAssistantMessage(blocks...))
		} else {
			msgs = append(msgs, anthropic.NewUserMessage(blocks...))
		}
	}
	// Anthropic rejects an assistant tool_use that isn't answered by a
	// tool_result in the immediately following user turn (e.g. a long-running
	// tool that defers its response, or a conversation ending on a tool_use).
	msgs = ensureToolResults(msgs)
	// The Messages API requires a non-empty, user-first conversation.
	if len(msgs) == 0 {
		msgs = append(msgs, anthropic.NewUserMessage(anthropic.NewTextBlock("Continue.")))
	}
	return msgs
}

// ensureToolResults guarantees every tool_use in an assistant turn is answered
// by a tool_result in the immediately following user turn, injecting an error
// placeholder for any that aren't (and a trailing user turn if the conversation
// ends on an unanswered tool_use). It is a no-op for fully-answered histories
// (the common synchronous case).
func ensureToolResults(msgs []anthropic.MessageParam) []anthropic.MessageParam {
	var out []anthropic.MessageParam
	for i := 0; i < len(msgs); i++ {
		out = append(out, msgs[i])
		if msgs[i].Role != anthropic.MessageParamRoleAssistant {
			continue
		}
		useIDs := blockIDs(msgs[i], true)
		if len(useIDs) == 0 {
			continue
		}
		answered := map[string]bool{}
		nextIsUser := i+1 < len(msgs) && msgs[i+1].Role == anthropic.MessageParamRoleUser
		if nextIsUser {
			for _, id := range blockIDs(msgs[i+1], false) {
				answered[id] = true
			}
		}
		var pending []anthropic.ContentBlockParamUnion
		for _, id := range useIDs {
			if !answered[id] {
				pending = append(pending, anthropic.NewToolResultBlock(id, "Tool result not available.", true))
			}
		}
		if len(pending) == 0 {
			continue
		}
		if nextIsUser {
			// tool_result blocks must lead the user turn.
			msgs[i+1].Content = append(pending, msgs[i+1].Content...)
		} else {
			out = append(out, anthropic.NewUserMessage(pending...))
		}
	}
	return out
}

// blockIDs returns the tool_use IDs (toolUse=true) or the tool_result target IDs
// (toolUse=false) from a message's content blocks, in order.
func blockIDs(m anthropic.MessageParam, toolUse bool) []string {
	var ids []string
	for _, b := range m.Content {
		switch {
		case toolUse && b.OfToolUse != nil:
			ids = append(ids, b.OfToolUse.ID)
		case !toolUse && b.OfToolResult != nil:
			ids = append(ids, b.OfToolResult.ToolUseID)
		}
	}
	return ids
}

// toAnthropicTools converts the request's genai function declarations into
// Anthropic tool definitions. functiontool populates FunctionDeclaration.
// ParametersJsonSchema with a standard JSON Schema, so it maps directly.
// Non-function tools (e.g. Gemini's GoogleSearch) have no declarations and are
// skipped.
func toAnthropicTools(cfg *genai.GenerateContentConfig) []anthropic.ToolUnionParam {
	var tools []anthropic.ToolUnionParam
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		// Map Gemini's built-in Google Search to Anthropic's web_search server
		// tool, so an agent that declares geminitool.GoogleSearch{} is grounded on
		// either provider. web_search_20260209 needs Opus 4.6+/Sonnet 4.6+ (our
		// default claude-opus-4-8 qualifies).
		if t.GoogleSearch != nil || t.GoogleSearchRetrieval != nil {
			tools = append(tools, anthropic.ToolUnionParam{
				OfWebSearchTool20260209: &anthropic.WebSearchTool20260209Param{},
			})
			continue
		}
		for _, d := range t.FunctionDeclarations {
			if d == nil || d.Name == "" {
				continue
			}
			tp := anthropic.ToolParam{
				Name:        d.Name,
				InputSchema: toolInputSchema(d),
			}
			if d.Description != "" {
				tp.Description = anthropic.String(d.Description)
			}
			tools = append(tools, anthropic.ToolUnionParam{OfTool: &tp})
		}
	}
	return tools
}

// toolInputSchema turns a genai FunctionDeclaration's JSON Schema into an
// Anthropic ToolInputSchemaParam, preserving properties, required, and any other
// schema keywords (additionalProperties, $defs, enums, …) via ExtraFields.
func toolInputSchema(d *genai.FunctionDeclaration) anthropic.ToolInputSchemaParam {
	schema := anthropic.ToolInputSchemaParam{}
	// functiontool uses ParametersJsonSchema (standard JSON Schema); other
	// standard tools (agenttool, loadmemorytool, loadartifactstool) use the
	// sibling Parameters (*genai.Schema). Prefer the former; fall back to the latter.
	var src any = d.ParametersJsonSchema
	if src == nil && d.Parameters != nil {
		src = d.Parameters
	}
	if src == nil {
		return schema
	}
	raw, err := json.Marshal(src)
	if err != nil {
		return schema
	}
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return schema
	}
	// A marshaled *genai.Schema carries UPPERCASE type enums ("OBJECT"/"STRING"),
	// which are invalid JSON Schema; normalize every "type" value to lowercase.
	lowercaseTypes(m)
	if props, ok := m["properties"]; ok {
		schema.Properties = props
	}
	if req, ok := m["required"].([]any); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				schema.Required = append(schema.Required, s)
			}
		}
	}
	extra := map[string]any{}
	for k, v := range m {
		switch k {
		case "type", "properties", "required":
			// modeled by dedicated fields
		default:
			extra[k] = v
		}
	}
	if len(extra) > 0 {
		schema.ExtraFields = extra
	}
	return schema
}

// lowercaseTypes recursively lowercases every JSON Schema "type" value in a
// decoded schema map (genai.Schema marshals them uppercase). It leaves a
// property literally named "type" alone by only rewriting string / []string
// values, and recurses into everything else.
func lowercaseTypes(v any) {
	switch t := v.(type) {
	case map[string]any:
		for k, val := range t {
			if k == "type" {
				switch s := val.(type) {
				case string:
					t[k] = strings.ToLower(s)
					continue
				case []any:
					for i, e := range s {
						if es, ok := e.(string); ok {
							s[i] = strings.ToLower(es)
						}
					}
					continue
				}
			}
			lowercaseTypes(val)
		}
	case []any:
		for _, e := range t {
			lowercaseTypes(e)
		}
	}
}

// funcResponseString renders a genai FunctionResponse's payload as the string
// content of an Anthropic tool_result block.
func funcResponseString(fr *genai.FunctionResponse) string {
	if fr.Response == nil {
		return ""
	}
	if b, err := json.Marshal(fr.Response); err == nil {
		return string(b)
	}
	return fmt.Sprint(fr.Response)
}

func partsText(parts []*genai.Part) string {
	var b strings.Builder
	for _, p := range parts {
		if p != nil && p.Text != "" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
