// Command eval is a small LLM-as-judge evaluation harness for ADK for Go agents.
// ADK 2.0's Dev UI has an "Evals" tab, but in v2.0.0 the Go backend's eval REST
// endpoints are stubs (HTTP 501) and there's no public eval package — so this is
// a DIY harness in the spirit of the upstream examples/web/agents/llmauditor.go.
//
// For each case, runEval produces a recommendation (either by running the
// agent-under-test on the case Input, or by using a planted ForceOutput), then a
// PANEL of judge agents (different Claude models) each score it against the case
// Rubric and return JSON {pass, score, rationale}. The panel majority-votes; the
// case is correct iff the panel's verdict matches the case's expected verdict.
//
// Two things make the eval trustworthy rather than a rubber stamp:
//   - a NEGATIVE CONTROL: a planted, dangerous recommendation the judges MUST
//     fail — so an always-pass judge is caught, not rewarded; and
//   - a JUDGE PANEL with majority vote, so one noisy/lenient judge can't decide.
//
//	export ANTHROPIC_API_KEY=...   # or `ant auth login`  (all agents run on Claude)
//	go run ./eval
//
// Exits non-zero if the panel disagrees with any case's expected verdict.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"

	"adk-tinker/claudemodel"
)

// judgeModels is the heterogeneous panel — different models vote independently so
// one lenient/noisy judge can't decide a case.
var judgeModels = []string{"claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"}

// EvalCase is one test. ExpectPass is the CORRECT verdict; ForceOutput, when set,
// plants a recommendation (used for negative controls) instead of running the agent.
type EvalCase struct {
	Name        string
	Input       string
	Rubric      string
	ExpectPass  bool
	ForceOutput string
}

// Verdict is one judge's structured decision.
type Verdict struct {
	Pass      bool
	Score     int // 1..5
	Rationale string
}

// Panel is the aggregate of the judges' verdicts for a case.
type Panel struct {
	Pass      bool   // majority
	Score     int    // median
	Agree     int    // judges agreeing with the majority pass
	N         int    // judges that returned a valid verdict
	Rationale string // a representative rationale
}

// Result pairs a case with the planted/produced output and the panel verdict.
type Result struct {
	Case     EvalCase
	Output   string
	Panel    Panel
	ParseErr string
}

const judgeInstruction = `You are a strict, safety-minded evaluator of Adirondack High Peaks hiking advice.
You are given a hiker PROFILE, an advisor's RECOMMENDATION, and a RUBRIC describing what a good
recommendation must satisfy. Decide whether the RECOMMENDATION satisfies the RUBRIC for that PROFILE.
Be strict: recommending a dangerous or ill-fitting peak fails, even if the prose is nice.
Respond with ONLY a JSON object and nothing else:
{"pass": true or false, "score": <integer 1-5>, "rationale": "<one short sentence>"}`

func defaultCases() []EvalCase {
	beginnerRubric := "PASS only if it recommends an accessible, mostly-forested, short peak such as Cascade, Porter, or Phelps, AND does NOT recommend a long, exposed, or technical peak (e.g. Marcy, Algonquin, Allen, Gothics, Saddleback, the Great Range)."
	return []EvalCase{
		{
			Name:       "beginner/short/no-exposure",
			Input:      "Complete beginner, first-ever High Peak, only half a day, dislikes heights and exposure.",
			Rubric:     beginnerRubric,
			ExpectPass: true,
		},
		{
			Name:       "expert/wants-hardest",
			Input:      "Very fit, experienced scrambler, wants the single hardest challenge, has a full day plus.",
			Rubric:     "PASS only if it recommends a genuinely demanding/remote or technical peak (e.g. Allen, Marshall, the Santanonis, Saddleback/Gothics scrambles, or a Great Range route) with challenge-appropriate reasoning. FAIL if it recommends an easy beginner peak like Cascade.",
			ExpectPass: true,
		},
		{
			Name:  "wants-highest-in-NY (reasoning)",
			Input: "Moderately fit hiker who wants to bag the highest peak in New York State but is nervous about committing to a big day.",
			// A reasoning dimension so a pure factual lookup ("Marcy") isn't enough.
			Rubric:     "PASS only if it recommends Mount Marcy (NY's highest, 5,344 ft) AND correctly frames it as a long, full-day effort (~14-15 mi round trip), setting realistic time/effort expectations. FAIL if it names Marcy but implies it's short/easy/casual, or recommends any other peak as the primary pick.",
			ExpectPass: true,
		},
		{
			Name:  "afraid-of-heights",
			Input: "Wants to summit a real High Peak but is genuinely afraid of heights, cliffs, and exposed knife-edges.",
			// Reframed: PASS hinges on picking a no-scramble peak AND acknowledging any open
			// summit stretch — not on pretending exposure is absent (Cascade's summit is open rock).
			Rubric:     "PASS only if it recommends a non-technical, no-scramble, no-knife-edge peak (e.g. Cascade, Phelps, Tabletop, Street/Nye) AND explicitly acknowledges any open/above-treeline summit stretch and how to manage it. FAIL if it recommends an exposed/scrambly peak like Gothics, Saddleback, Basin, or the Range, or if it ignores the hiker's fear of heights.",
			ExpectPass: true,
		},
		{
			Name:       "very-short-window",
			Input:      "Only has about two hours near Lake Placid and wants the quickest possible summit.",
			Rubric:     "PASS if it either recommends the shortest reasonable option (e.g. Cascade) OR honestly notes that a true 46 High Peak likely needs more than two hours and suggests a shorter alternative. FAIL if it recommends a long peak (Marcy, Allen, Santanonis) as doable in two hours without caveat.",
			ExpectPass: true,
		},
		{
			// NEGATIVE CONTROL: a planted, dangerous recommendation for a scared beginner.
			// The panel MUST reject it; an always-pass/lenient judge is caught here.
			Name:        "negative-control/dangerous-pick",
			Input:       "Complete beginner, first-ever High Peak, only half a day, dislikes heights and exposure.",
			Rubric:      beginnerRubric,
			ForceOutput: "Go summit Gothics via the cabled cliffs — the exposed knife-edge and steel-cable scramble are the highlight, and you'll be totally fine on your first-ever hike even with a fear of heights.",
			ExpectPass:  false,
		},
	}
}

func runOneTurn(ctx context.Context, r *runner.Runner, userID, sessionID, prompt string) (string, error) {
	var out strings.Builder
	for ev, err := range r.Run(ctx, userID, sessionID, genai.NewContentFromText(prompt, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return "", err
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				out.WriteString(p.Text)
			}
		}
	}
	return strings.TrimSpace(out.String()), nil
}

// parseVerdict extracts the judge's JSON verdict, tolerating prose or braces
// around it, and requiring the pass/score fields to actually be present (a
// missing field routes to an error, not a silent false/zero).
func parseVerdict(s string) (Verdict, error) {
	type raw struct {
		Pass      *bool  `json:"pass"`
		Score     *int   `json:"score"`
		Rationale string `json:"rationale"`
	}
	try := func(sub string) (Verdict, bool) {
		var r raw
		if err := json.NewDecoder(strings.NewReader(sub)).Decode(&r); err != nil {
			return Verdict{}, false
		}
		if r.Pass == nil || r.Score == nil {
			return Verdict{}, false
		}
		v := Verdict{Pass: *r.Pass, Score: *r.Score, Rationale: r.Rationale}
		if v.Score < 1 {
			v.Score = 1
		} else if v.Score > 5 {
			v.Score = 5
		}
		return v, true
	}
	if v, ok := try(strings.TrimSpace(s)); ok {
		return v, nil
	}
	for idx := strings.IndexByte(s, '{'); idx >= 0; {
		if v, ok := try(s[idx:]); ok {
			return v, nil
		}
		next := strings.IndexByte(s[idx+1:], '{')
		if next < 0 {
			break
		}
		idx = idx + 1 + next
	}
	return Verdict{}, fmt.Errorf("no complete JSON verdict in judge output")
}

// vote aggregates the panel's verdicts: majority pass (ties → fail, conservative)
// and median score.
func vote(vs []Verdict) Panel {
	if len(vs) == 0 {
		return Panel{}
	}
	passes := 0
	scores := make([]int, 0, len(vs))
	for _, v := range vs {
		if v.Pass {
			passes++
		}
		scores = append(scores, v.Score)
	}
	majority := passes*2 > len(vs)
	agree := 0
	rationale := ""
	for _, v := range vs {
		if v.Pass == majority {
			agree++
			if rationale == "" {
				rationale = v.Rationale
			}
		}
	}
	sort.Ints(scores)
	return Panel{Pass: majority, Score: scores[len(scores)/2], Agree: agree, N: len(vs), Rationale: rationale}
}

func runEval(ctx context.Context, underTest agent.Agent, judges []*runner.Runner, cases []EvalCase) []Result {
	recRunner, err := runner.New(runner.Config{AppName: "eval-agent", Agent: underTest, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		fatal("create agent runner: %v", err)
	}

	results := make([]Result, 0, len(cases))
	for i, c := range cases {
		fmt.Printf("  case %d/%d: %s …\n", i+1, len(cases), c.Name)
		out := c.ForceOutput
		if out == "" {
			out, err = runOneTurn(ctx, recRunner, "hiker", fmt.Sprintf("case-%d", i), c.Input)
			if err != nil {
				results = append(results, Result{Case: c, ParseErr: "agent error: " + err.Error()})
				continue
			}
		}
		judgePrompt := fmt.Sprintf("PROFILE:\n%s\n\nRECOMMENDATION:\n%s\n\nRUBRIC:\n%s", c.Input, out, c.Rubric)
		var verdicts []Verdict
		for j, jr := range judges {
			jout, err := runOneTurn(ctx, jr, "judge", fmt.Sprintf("j%d-case-%d", j, i), judgePrompt)
			if err != nil {
				continue // a dead judge just doesn't vote
			}
			if v, perr := parseVerdict(jout); perr == nil {
				verdicts = append(verdicts, v)
			}
		}
		res := Result{Case: c, Output: out}
		if len(verdicts) == 0 {
			res.ParseErr = "no judge returned a valid verdict"
		} else {
			res.Panel = vote(verdicts)
		}
		results = append(results, res)
	}
	return results
}

func printReport(results []Result) int {
	fmt.Printf("\n═══ eval report (panel of %d judges: %s) ═══\n\n", len(judgeModels), strings.Join(judgeModels, ", "))
	correct, total := 0, 0
	agentScoreSum, agentScoreN := 0, 0
	for _, r := range results {
		total++
		exp := verdictWord(r.Case.ExpectPass)
		if r.ParseErr != "" {
			fmt.Printf("⚠️  ERR   expected %-4s  %s\n        %s\n", exp, r.Case.Name, r.ParseErr)
			continue
		}
		got := verdictWord(r.Panel.Pass)
		ok := r.Panel.Pass == r.Case.ExpectPass
		mark := "❌ MISS"
		if ok {
			mark = "✅ OK  "
			correct++
		}
		// The "agent score" is only meaningful for real (expected-PASS) cases; a
		// low score on the planted negative control is correct, not agent quality.
		if r.Case.ExpectPass {
			agentScoreSum += r.Panel.Score
			agentScoreN++
		}
		fmt.Printf("%s  expected %-4s got %-4s (%d/%d agree, score %d/5)  %s\n        %s\n",
			mark, exp, got, r.Panel.Agree, r.Panel.N, r.Panel.Score, r.Case.Name, r.Panel.Rationale)
	}
	avg := 0.0
	if agentScoreN > 0 {
		avg = float64(agentScoreSum) / float64(agentScoreN)
	}
	fmt.Printf("\n── summary ──\njudge accuracy: %d/%d cases matched the expected verdict\nagent quality:  avg %.1f/5 over %d real cases\n",
		correct, total, avg, agentScoreN)
	if correct == total {
		fmt.Println("✅ the panel discriminated correctly (incl. the negative control).")
		return 0
	}
	fmt.Println("❌ the panel contested a labeled verdict — inspect the agent's answer, the label, or the rubric for that case.")
	return 1
}

func verdictWord(pass bool) string {
	if pass {
		return "PASS"
	}
	return "FAIL"
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(2)
}

func apiKeyPresent() bool {
	return os.Getenv("ANTHROPIC_API_KEY") != "" || os.Getenv("ANTHROPIC_AUTH_TOKEN") != "" ||
		func() bool { _, err := os.Stat(os.Getenv("HOME") + "/.config/anthropic/credentials/default.json"); return err == nil }()
}

func main() {
	ctx := context.Background()
	if !apiKeyPresent() {
		fatal("need Anthropic creds — set ANTHROPIC_API_KEY or run `ant auth login`")
	}

	// Agent under test: an Adirondack peak recommender (no web search, for a fast,
	// reasoning-focused eval). Swap this for any agent.Agent you want to evaluate.
	underTest, err := llmagent.New(llmagent.Config{
		Name:        "peak_recommender",
		Model:       claudemodel.NewModel(""),
		Description: "recommends an Adirondack High Peak for a hiker",
		Instruction: "You are an Adirondack High Peaks advisor. Given a hiker's profile, recommend ONE specific " +
			"High Peak that fits them, in 2-3 sentences with a brief why. Weigh difficulty, exposure, and available time.",
	})
	if err != nil {
		fatal("build agent-under-test: %v", err)
	}

	// Judge panel: one runner per model.
	var judges []*runner.Runner
	for _, m := range judgeModels {
		j, err := llmagent.New(llmagent.Config{
			Name:        "judge",
			Model:       claudemodel.NewModel(m),
			Description: "grades hiking recommendations against a rubric",
			Instruction: judgeInstruction,
		})
		if err != nil {
			fatal("build judge %s: %v", m, err)
		}
		jr, err := runner.New(runner.Config{AppName: "eval-judge-" + m, Agent: j, SessionService: session.InMemoryService(), AutoCreateSession: true})
		if err != nil {
			fatal("judge runner %s: %v", m, err)
		}
		judges = append(judges, jr)
	}

	fmt.Printf("🧪 Evaluating peak_recommender with a %d-judge panel…\n", len(judges))
	results := runEval(ctx, underTest, judges, defaultCases())
	os.Exit(printReport(results))
}
