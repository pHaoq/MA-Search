# MA Search: thesis topic finder

A multi-agent loop that hunts for master thesis **topics** (themes, not job positions).
Claude, Gemini and GPT each run deep web searches, rotating over six source strategies.
Every topic they find is checked against a local store, and only new ones get validated
by all three models and saved as **valid** or **invalid** (with the reason). The loop keeps
going until a stop rule is met.

## How it works

Each round:

1. **Plan.** Claude writes a research brief for each model, for that model's strategy this
   round. The brief lists every topic already stored, so the agents look somewhere new.
   Rejected topics are listed with their reason: the exact topic is off the table, but a
   neighbour that fixes the reason is still allowed.
2. **Deep search, in parallel.**
   - Gemini: Deep Research agent (`deep-research-preview-04-2026`, Interactions API)
   - Claude: Opus 5.5 with the web search and web fetch tools
   - GPT: GPT-6.1 Sol with the web search tool, high reasoning effort, background mode
3. **Extract.** Claude Haiku 5.5 turns each report into structured topic records.
4. **Dedup.** Exact title match, then meaning match via embeddings. Borderline cases go to
   Claude Haiku ("same topic?"). Duplicates are skipped and their `times_seen` counter goes up.
5. **Validate.** OpenAlex finds the closest existing papers, then each model scores the
   topic (novelty, feasibility, data access, demand, fit). Valid = at least 2 of 3 pass and
   the mean score is at least 3.5.
6. **Store** in SQLite (`data/topics.db`), then check the stop rules.

### The six strategies

| # | Strategy | Where it looks |
|---|----------|----------------|
| 0 | Conference calls for papers | ECIS, ICIS, WI, HICSS, AMCIS, PACIS tracks and workshops |
| 1 | Research agendas and editorials | Special issues and editorials of IS journals, "research agenda" papers (+ OpenAlex seeds) |
| 2 | Replicate or transfer a study | Highly cited papers (OpenAlex, by citations), moved to a new context |
| 3 | Find contradictions | Reviews and papers with conflicting results (+ OpenAlex seeds) |
| 4 | Funded project databases | CORDIS, FFG, DFG GEPRIS |
| 5 | Industry job postings | Austrian and German job boards, skill-gap reports |

Edit them in `src/thesis_finder/strategies.py`.

### When it stops

Whichever comes first:

- **Budget cap**: the run has spent `budget_usd` (default $20). Always checked.
- **Max rounds**: `max_rounds` (default 60).
- **Target**: `target_valid` valid topics are stored (default 40).
- **Saturation**: `saturation_rounds` rounds in a row (default 3) where fewer than
  `saturation_new_ratio` (default 10%) of found topics are new.

Target and saturation only count once every model has tried every strategy at least once
(3 models x 6 strategies = 18 pairs, covered in 6 rounds). The store is kept between runs,
so a later run continues where the last one stopped. Press Ctrl+C to stop early; everything
found so far is saved.

## Setup

You need API keys. These are separate from Claude Pro, Gemini Pro and ChatGPT Pro subscriptions:

- Anthropic: https://console.anthropic.com
- Google Gemini: https://aistudio.google.com
- OpenAI: https://platform.openai.com

Set a monthly spend limit in each console.

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                 # add your three keys
cp config.example.yaml config.yaml   # fill in your profile: interests, methods, university
```

Windows (PowerShell), from the repo folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
copy .env.example .env
copy config.example.yaml config.yaml
```

Activate the venv in every new terminal before running the tool. If `thesis-finder` is still
not found, `python -m thesis_finder` does the same thing (for example
`python -m thesis_finder run --demo`).

## Usage

```bash
thesis-finder run --demo          # try the loop offline with fake agents (no keys, no cost)
thesis-finder run --rounds 1      # one real round, to check keys and cost
thesis-finder run                 # run until a stop rule is met
thesis-finder run --budget 5      # override the budget cap for this run

thesis-finder list --status valid
thesis-finder list --status invalid
thesis-finder report              # writes reports/shortlist.md
thesis-finder runs                # past runs, spend and why each stopped
```

To use fewer providers (for example Claude only while testing), set `providers: [claude]`
in `config.yaml`. Claude is always used as the coordinator.

## Cost

Prices are in `src/thesis_finder/budget.py` (USD, October 2026) and can be overridden in
`config.yaml`. The Gemini Deep Research agent has no published flat rate, so its price is an
estimate: compare with your Google AI Studio usage after the first run. Start with
`--rounds 1` to see what one round really costs.

## Tests

```bash
pytest
```

The tests use fake providers and run offline.

## Project layout

```
src/thesis_finder/
  cli.py          command line and wiring
  loop.py         the loop, rotation and stop rules
  strategies.py   the six source strategies
  prompts.py      all prompt text
  providers/      claude.py, gemini.py, openai_provider.py, fake.py
  dedup.py        exact + embedding duplicate check
  validator.py    3-model scoring
  store.py        SQLite topic store
  embeddings.py   OpenAI, Gemini or local embeddings
  sources/openalex.py  scholarly search for seeds and the novelty check
  budget.py       spend tracking
  report.py       Markdown shortlist
```
