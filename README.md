# MA Search: thesis topic finder

A multi-agent loop that hunts for master thesis **topics** (themes, not job positions).
Claude, Gemini and GPT each run deep web searches, rotating over seven source strategies.
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
3. **Extract.** Claude Haiku 5.5 turns each report into structured topic records, including the
   closest existing paper and what the thesis adds beyond it.
4. **Dedup.** Exact title match, then meaning match via embeddings. Borderline cases go to
   Claude Haiku ("same topic?"). Duplicates are skipped and their `times_seen` counter goes up.
5. **Prior-art check (free).** OpenAlex and Semantic Scholar return the closest papers; if one is
   too similar (embedding similarity at least `prior_art_cutoff`), the topic is stored as invalid
   with "already done: <paper>", and no judge is paid.
6. **Validate.** A topic that does not say what it adds beyond existing work fails right away.
   Otherwise the judges score it (novelty, feasibility, data access, demand, fit), with the
   closest papers in front of them. Valid = at least 2 of 3 pass and the mean is at least 3.5.
   GPT and Gemini judge first; Claude is only asked when they disagree.
7. **Store** in SQLite (`data/topics.db`), then check the stop rules.

### Statuses and the review step

| Status | Meaning |
|--------|---------|
| `invalid` | Failed screening or the literature check; the reason is kept and the agents won't propose it again |
| `valid` | Passed the first screening by the three judges |
| `checked` | Ranked into the top picks by the strongest model and passed a literature review |
| `parked` | Valid, but not picked in `park_after` reviews in a row |

Every `every_rounds` rounds (default 7) and at the end of a run, the review step runs:
GPT-6.1 Sol (high reasoning) compares all valid topics side by side and picks the top `top_k`
(default 5). Each pick gets a literature review (GPT with web search, plus OpenAlex): is the
question already answered, is the gap real, is there enough literature, is the data reachable?
Set `review.provider: claude` to use Claude Fable 5.1 and Sonnet 5.5 instead (several times the cost).
Pass: `checked`, with the review summary and key references saved. Fail: back to `invalid`.
A review costs roughly $1-2 with GPT (about $3-6 with Claude). Run it by hand with `thesis-finder review`.

The first screening is strict on purpose: judges are told a typical proposal scores 3 and asked
whether the topic is a top-10% master thesis; a topic needs a mean score of at least 3.5 and
novelty of at least 3 from every judge. If almost everything fails, run
`thesis-finder list --status invalid` to see the reasons, and lower `min_mean_score` if needed.

### Originality

Topics that just rephrase an older paper are filtered in four places:

- **Fresh sources.** Briefs only allow calls, agendas, funded projects and job ads from the last
  `fresh_months` (18), and starting papers from `paper_from_year` (2024) on.
- **What's new.** Every topic must name its closest existing paper and what it adds (a new
  context, method, data or population). No stated addition: invalid, without a judge call.
  Judges score novelty on that addition, and a rephrased paper scores 1 or 2.
- **Prior-art check.** See step 5. Run `thesis-finder run -v` to see each topic's similarity to
  its closest paper; raise `prior_art_cutoff` if good topics are rejected as "already done",
  lower it if copies slip through.
- **Combine strategy.** Strategy 6 builds topics from combinations that have not been published:
  a gap plus a contradiction, a known finding in a new context, or two of your interests.

What helps most is a specific profile: `seed_papers` (3-5 papers you liked), `data_access`
(data or field access only you have), `target_chair` (the chair you aim for and its focus), and
specific interests and methods. The agents use these to find topics only you can do.

### The seven strategies

| # | Strategy | Where it looks |
|---|----------|----------------|
| 0 | Conference calls for papers | ECIS, ICIS, WI, HICSS, AMCIS, PACIS tracks and workshops |
| 1 | Research agendas and editorials | Special issues and editorials of IS journals, "research agenda" papers (+ OpenAlex seeds) |
| 2 | Replicate or transfer a study | Highly cited papers (OpenAlex, by citations), moved to a new context |
| 3 | Find contradictions | Reviews and papers with conflicting results (+ OpenAlex seeds) |
| 4 | Funded project databases | CORDIS, FFG, DFG GEPRIS |
| 5 | Industry job postings | Austrian and German job boards, skill-gap reports |
| 6 | Combine gaps and contexts | Recent "future research" sections, Austrian/EU reports, your interests (+ OpenAlex seeds) |

Edit them in `src/thesis_finder/strategies.py`.

### When it stops

Whichever comes first:

- **Budget cap**: the run has spent `budget_usd` (default $5). Always checked, and a new
  round is not started if the last rounds suggest it would overshoot the cap.
- **Max rounds**: `max_rounds` (default 60).
- **Target**: `target_valid` valid topics are stored (default 40).
- **Saturation**: `saturation_rounds` rounds in a row (default 3) where fewer than
  `saturation_new_ratio` (default 10%) of found topics are new.

Target and saturation only count once every model has tried every strategy at least once
(3 models x 7 strategies = 21 pairs, covered in 7 rounds). The store is kept between runs,
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

thesis-finder list --status checked
thesis-finder list --status valid
thesis-finder list --status invalid
thesis-finder review              # rank the valid topics and literature-review the top 5 now
thesis-finder report              # writes reports/shortlist.md, checked topics first
thesis-finder runs                # past runs, spend and why each stopped
thesis-finder costs               # spend by provider, job and model, from data/costs.csv
```

To use fewer providers (for example Claude only while testing), set `providers: [claude]`
in `config.yaml`. Claude Haiku 5.5 always extracts topics and checks duplicates (it costs cents per run).

## Cost

What drives Claude's cost:

- **Deep search** (`claude_search` in the config). Every search result and fetched page is
  added to Claude's context, and the whole context is billed again at each step of the
  search. More searches and fetches, bigger pages and higher effort multiply the cost. The
  defaults are Sonnet 5.5, 8 searches, 4 fetches, pages capped at 15,000 tokens, medium effort.
- **Judging**: Claude (Sonnet 5.5) only judges a topic when GPT and Gemini disagree.
- **Review**: runs on GPT by default; `review.provider: claude` moves it back to Fable 5.1 + Sonnet 5.5.
- **Extraction and duplicate checks** use Haiku 5.5 and are cheap.
- **Planning** is a free template by default; `planner: claude` adds one Claude call per search.

Built-in savings that keep the same models:

- **Flex processing** (`cost.flex_search`, on by default): OpenAI and Gemini deep searches run at
  half the token price. They may queue for a few minutes, which barely shows on searches that
  already take several. When flex has no capacity, the call runs at standard price. Search fees
  are not discounted. `cost.flex_judge: true` halves judge cost too, but each topic can then wait
  minutes, so rounds get much slower.
- **Free filters before paid judges**: duplicates, topics without a stated addition, and topics
  the prior-art check finds already published never reach a judge.
- **Early exit**: GPT and Gemini judge first. When they agree (both pass or both fail), Claude
  is not asked; it only breaks ties.
- **Cached input**: OpenAI's discounted cached tokens are now billed as such in the cost log.

Not used, and why: the 50% Batch APIs return results in minutes to hours and do not support
Claude's refusal fallback, which does not fit a loop that judges each topic before moving on;
Claude prompt caching already applies inside deep searches (you see it as cache tokens in the
cost log), and the other prompts are too short and too different to cache.

Every API call is written to `data/costs.csv`; run `thesis-finder costs` to see where the
money went. To spend less on Claude, set `providers: [gemini, openai]` so Claude only
coordinates (it still extracts topics, cheaply), or lower `claude_search` further.

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
  strategies.py   the seven source strategies
  prompts.py      all prompt text
  providers/      claude.py, gemini.py, openai_provider.py, fake.py
  dedup.py        exact + embedding duplicate check
  validator.py    3-model scoring with early exit
  prior_art.py    OpenAlex + Semantic Scholar check for already-published topics
  review.py       ranking + literature review: valid -> checked
  store.py        SQLite topic store
  embeddings.py   OpenAI, Gemini or local embeddings
  sources/        openalex.py and semantic_scholar.py: free scholarly search
  budget.py       spend tracking
  report.py       Markdown shortlist
```
