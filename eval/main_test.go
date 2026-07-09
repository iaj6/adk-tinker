package main

import "testing"

// Guards parseVerdict: it must find the JSON verdict amid prose/braces AND treat
// a missing pass/score as an error (not a silent false/[1/5]).
func TestParseVerdict(t *testing.T) {
	cases := []struct {
		name      string
		in        string
		wantPass  bool
		wantScore int
		wantErr   bool
	}{
		{"clean", `{"pass":true,"score":4,"rationale":"good"}`, true, 4, false},
		{"prose-around", `Reasoning here. {"pass":false,"score":2,"rationale":"nope"} thanks`, false, 2, false},
		{"prose-with-braces", `I weighed {this} and {that}, verdict: {"pass":true,"score":5,"rationale":"ok"}`, true, 5, false},
		{"brace-in-rationale", `{"pass":true,"score":5,"rationale":"bring {crampons}"}`, true, 5, false},
		{"score-clamped-high", `{"pass":true,"score":9,"rationale":"x"}`, true, 5, false},
		{"score-clamped-low", `{"pass":false,"score":0,"rationale":"x"}`, false, 1, false},
		{"missing-pass", `{"score":3,"rationale":"x"}`, false, 0, true},
		{"missing-score", `{"pass":true,"rationale":"x"}`, false, 0, true},
		{"no-json", `PASS — looks fine to me`, false, 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v, err := parseVerdict(c.in)
			if (err != nil) != c.wantErr {
				t.Fatalf("err=%v, wantErr=%v", err, c.wantErr)
			}
			if err != nil {
				return
			}
			if v.Pass != c.wantPass || v.Score != c.wantScore {
				t.Errorf("got {pass:%v score:%d} want {pass:%v score:%d}", v.Pass, v.Score, c.wantPass, c.wantScore)
			}
		})
	}
}

// Guards vote: majority pass, median score, tie → fail (conservative).
func TestVote(t *testing.T) {
	maj := vote([]Verdict{{Pass: true, Score: 4}, {Pass: true, Score: 5}, {Pass: false, Score: 1}})
	if !maj.Pass {
		t.Error("2/3 pass should be a majority PASS")
	}
	if maj.Score != 4 { // sorted [1,4,5] → median 4
		t.Errorf("median score = %d, want 4", maj.Score)
	}
	if maj.Agree != 2 {
		t.Errorf("agree = %d, want 2", maj.Agree)
	}
	if tie := vote([]Verdict{{Pass: true, Score: 5}, {Pass: false, Score: 1}}); tie.Pass {
		t.Error("1/2 pass is a tie and must NOT be a majority PASS (conservative)")
	}
}
