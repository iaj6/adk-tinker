package claudemodel

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"google.golang.org/genai"
)

// Guards the blocker fix: tools whose schema lives in FunctionDeclaration.Parameters
// (*genai.Schema, e.g. agenttool/loadmemorytool/loadartifactstool) must still get a
// valid, LOWERCASE JSON Schema — genai marshals type enums uppercase ("OBJECT"/"STRING").
func TestToolInputSchema_ParametersFallbackLowercasesTypes(t *testing.T) {
	decl := &genai.FunctionDeclaration{
		Name: "run_agent",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"request": {Type: genai.TypeString, Description: "the request"},
			},
			Required: []string{"request"},
		},
	}
	raw, err := json.Marshal(toolInputSchema(decl))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := string(raw)
	if strings.Contains(got, "STRING") || strings.Contains(got, "OBJECT") {
		t.Errorf("uppercase genai types leaked into JSON schema: %s", got)
	}
	if !strings.Contains(got, `"request"`) {
		t.Errorf("property lost: %s", got)
	}
	if !strings.Contains(got, `"required":["request"]`) {
		t.Errorf("required lost: %s", got)
	}
	if !strings.Contains(got, `"type":"string"`) {
		t.Errorf("nested type not lowercased: %s", got)
	}
}

// Guards the same-role coalescing fix: two consecutive genai "user" turns must
// become ONE Anthropic user message (else Anthropic 400s on "roles must alternate").
func TestToAnthropicMessages_CoalescesConsecutiveUserTurns(t *testing.T) {
	msgs := toAnthropicMessages([]*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "first"}}},
		{Role: "user", Parts: []*genai.Part{{Text: "second"}}},
	})
	if len(msgs) != 1 {
		t.Fatalf("want 1 coalesced user message, got %d", len(msgs))
	}
	if msgs[0].Role != anthropic.MessageParamRoleUser {
		t.Errorf("want user role, got %s", msgs[0].Role)
	}
	if len(msgs[0].Content) != 2 {
		t.Errorf("want 2 merged blocks, got %d", len(msgs[0].Content))
	}
}

// Guards the tool_use/tool_result pairing fix: an assistant tool_use with no
// matching tool_result (a deferred/long-running tool, or a history ending on a
// tool_use) must get a synthesized user tool_result so Anthropic doesn't 400.
func TestEnsureToolResults_FillsDanglingToolUse(t *testing.T) {
	msgs := toAnthropicMessages([]*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "start the job"}}},
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_A", Name: "long_job"}}}},
	})
	if len(msgs) != 3 { // user, assistant(tool_use), synthesized user(tool_result)
		t.Fatalf("want 3 messages, got %d", len(msgs))
	}
	last := msgs[len(msgs)-1]
	if last.Role != anthropic.MessageParamRoleUser {
		t.Fatalf("want trailing user message, got %s", last.Role)
	}
	if ids := blockIDs(last, false); len(ids) != 1 || ids[0] != "call_A" {
		t.Errorf("want one tool_result for call_A, got %v", ids)
	}
}

// The synchronous happy path must remain a no-op for the pairing pass: a fully
// answered tool call stays a clean 3-message user/assistant/user sequence.
func TestEnsureToolResults_NoOpWhenAnswered(t *testing.T) {
	msgs := toAnthropicMessages([]*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "add 2 and 3"}}},
		{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_A", Name: "add"}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "call_A", Name: "add", Response: map[string]any{"sum": 5}}}}},
	})
	if len(msgs) != 3 {
		t.Fatalf("want 3 messages unchanged, got %d", len(msgs))
	}
}
