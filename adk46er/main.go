// Command adk46er is a durable Adirondack 46er tracker: log which of the 46 High
// Peaks you've bagged (persisted in SQLite, the same store the workflow HITL
// demos use), see your progress, and ask a Claude "mentor" agent which peak to
// tackle next.
//
//	adk46er bag "Mount Marcy"     record a summit  (fuzzy-matched to the 46)
//	adk46er list                  show progress N/46 + a bar
//	adk46er next                  a Claude 46er mentor suggests your next peak
//	adk46er reset                 clear the log
//
// The tracker data lives in adk46er.db and survives across runs; `next` needs
// Anthropic creds (ANTHROPIC_API_KEY or `ant auth login`).
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"

	"adk-tinker/claudemodel"
)

const dbPath = "adk46er.db"

// The 46 traditional Adirondack High Peaks.
var highPeaks = []string{
	"Mount Marcy", "Algonquin Peak", "Mount Haystack", "Mount Skylight", "Whiteface Mountain",
	"Dix Mountain", "Gray Peak", "Iroquois Peak", "Basin Mountain", "Gothics",
	"Mount Colden", "Giant Mountain", "Nippletop", "Santanoni Peak", "Mount Redfield",
	"Wright Peak", "Saddleback Mountain", "Panther Peak", "TableTop Mountain", "Rocky Peak Ridge",
	"Macomb Mountain", "Armstrong Mountain", "Hough Peak", "Seward Mountain", "Mount Marshall",
	"Allen Mountain", "Big Slide Mountain", "Esther Mountain", "Upper Wolfjaw Mountain", "Lower Wolfjaw Mountain",
	"Street Mountain", "Phelps Mountain", "Mount Donaldson", "Seymour Mountain", "Sawteeth",
	"Cascade Mountain", "South Dix", "Porter Mountain", "Mount Colvin", "Mount Emmons",
	"Dial Mountain", "Grace Peak", "Blake Peak", "Cliff Mountain", "Nye Mountain",
	"Couchsachraga Peak",
}

// Peak is a bagged summit row.
type Peak struct {
	Name     string `gorm:"primaryKey"`
	BaggedAt time.Time
}

func openDB() *gorm.DB {
	db, err := gorm.Open(sqlite.Open(dbPath), &gorm.Config{Logger: logger.Default.LogMode(logger.Silent)})
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	if err := db.AutoMigrate(&Peak{}); err != nil {
		log.Fatalf("migrate: %v", err)
	}
	return db
}

// matchPeak fuzzily resolves user input to a canonical High Peak name.
func matchPeak(input string) (string, bool) {
	q := strings.ToLower(strings.TrimSpace(input))
	if q == "" {
		return "", false
	}
	for _, p := range highPeaks { // exact-ish: canonical contains the query
		if strings.Contains(strings.ToLower(p), q) {
			return p, true
		}
	}
	// looser: the query contains a distinctive word of the canonical name
	for _, p := range highPeaks {
		for _, w := range strings.Fields(strings.ToLower(p)) {
			if len(w) >= 4 && strings.Contains(q, w) {
				return p, true
			}
		}
	}
	return "", false
}

func bagged(db *gorm.DB) []Peak {
	var peaks []Peak
	db.Order("bagged_at").Find(&peaks)
	return peaks
}

func remaining(db *gorm.DB) []string {
	got := map[string]bool{}
	for _, p := range bagged(db) {
		got[p.Name] = true
	}
	var rem []string
	for _, p := range highPeaks {
		if !got[p] {
			rem = append(rem, p)
		}
	}
	return rem
}

func cmdBag(db *gorm.DB, input string) {
	name, ok := matchPeak(input)
	if !ok {
		fmt.Printf("🤔 %q isn't one of the 46 High Peaks I recognize. Try e.g. \"Marcy\", \"Algonquin\", \"Gothics\".\n", input)
		return
	}
	db.Save(&Peak{Name: name, BaggedAt: time.Now()})
	n := len(bagged(db))
	fmt.Printf("🥾 Bagged %s!  Progress: %d/46\n", name, n)
	if n == 46 {
		fmt.Println("🎉🏔️  You're an ADIRONDACK 46ER!! Sign the register at the ADK Loj. 🍺")
	}
}

func cmdList(db *gorm.DB) {
	peaks := bagged(db)
	n := len(peaks)
	filled := n * 24 / 46
	fmt.Printf("🏔️  Adirondack 46er progress: %d / 46\n\n[%s%s]\n\n",
		n, strings.Repeat("█", filled), strings.Repeat("░", 24-filled))
	if n > 0 {
		fmt.Println("Bagged:")
		for _, p := range peaks {
			fmt.Printf("  ✓ %-22s %s\n", p.Name, p.BaggedAt.Format("2006-01-02"))
		}
	}
	if n < 46 {
		fmt.Printf("\n%d to go. Run `adk46er next` for a suggestion.\n", 46-n)
	}
}

func cmdNext(ctx context.Context, db *gorm.DB) {
	rem := remaining(db)
	if len(rem) == 0 {
		fmt.Println("🎉 You've bagged all 46 — there is no next peak, only celebration.")
		return
	}
	baggedNames := make([]string, 0)
	for _, p := range bagged(db) {
		baggedNames = append(baggedNames, p.Name)
	}
	done := "none yet"
	if len(baggedNames) > 0 {
		done = strings.Join(baggedNames, ", ")
	}

	mentor, err := llmagent.New(llmagent.Config{
		Name:        "mentor",
		Model:       claudemodel.NewModel(""),
		Description: "a wise Adirondack 46er mentor",
		Instruction: "You are a warm, experienced Adirondack 46er mentor. Given the peaks a hiker has already " +
			"bagged and the ones remaining, recommend the SINGLE best next peak from the remaining list — one that " +
			"builds their skills sensibly. Answer in one short, motivating paragraph naming the peak and why.",
	})
	if err != nil {
		log.Fatalf("mentor agent: %v", err)
	}
	const appName, userID, sessionID = "adk46er", "hiker", "s1"
	svc := session.InMemoryService()
	_, _ = svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	r, err := runner.New(runner.Config{AppName: appName, Agent: mentor, SessionService: svc})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	prompt := fmt.Sprintf("Already bagged (%d): %s.\n\nRemaining (%d): %s.\n\nWhich remaining peak should I do next, and why?",
		len(baggedNames), done, len(rem), strings.Join(rem, ", "))
	fmt.Print("🧭 Consulting your 46er mentor…\n\n")
	var out strings.Builder
	for ev, err := range r.Run(ctx, userID, sessionID, genai.NewContentFromText(prompt, genai.RoleUser), agent.RunConfig{}) {
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

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, `  adk46er bag "<peak>"   record a bagged High Peak`)
	fmt.Fprintln(os.Stderr, `  adk46er list           show progress`)
	fmt.Fprintln(os.Stderr, `  adk46er next           mentor suggests your next peak`)
	fmt.Fprintln(os.Stderr, `  adk46er reset          clear the log`)
	os.Exit(2)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	db := openDB()
	switch os.Args[1] {
	case "bag":
		if len(os.Args) < 3 {
			usage()
		}
		cmdBag(db, strings.Join(os.Args[2:], " "))
	case "list":
		cmdList(db)
	case "next":
		cmdNext(context.Background(), db)
	case "reset":
		db.Where("1 = 1").Delete(&Peak{})
		fmt.Println("🧹 Log cleared. Back to 0/46.")
	default:
		usage()
	}
}
