"""Command line: thesis-finder run | list | report | runs"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import prompts
from .budget import Budget, summarize_log
from .config import Config, load_config
from .dedup import Deduper
from .embeddings import LocalEmbedder, make_embedder
from .loop import TopicLoop
from .report import markdown_report
from .review import ReviewStep
from .store import TopicStore
from .validator import Validator

KEY_VARS = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


def build_loop(cfg: Config, demo: bool = False) -> TopicLoop:
    budget = Budget(cfg.loop.budget_usd, cfg.prices, log_path=None if demo else cfg.cost_log_path)
    store = TopicStore(cfg.store_path)

    if demo:
        from .providers.fake import FakeCoordinator, FakeProvider

        searchers = [FakeProvider(name) for name in cfg.providers]
        coordinator = FakeCoordinator()
        embedder = LocalEmbedder()
        validator = Validator(searchers, cfg.profile, cfg.validation.min_pass_votes,
                              cfg.validation.min_mean_score, cfg.validation.min_novelty,
                              find_related=lambda t, q: [])
        deduper = Deduper(store, embedder, cfg.dedup.duplicate_threshold, cfg.dedup.borderline_threshold)
        review = ReviewStep(cfg, store, budget, coordinator, find_related=lambda t, q: [])
        return TopicLoop(cfg, store, budget, searchers, coordinator, deduper, validator,
                         seed_search=lambda *a, **k: [], review_step=review)

    missing = [KEY_VARS[p] for p in {"claude", *cfg.providers} if not os.environ.get(KEY_VARS[p])]
    if cfg.embedding.provider in KEY_VARS and not os.environ.get(KEY_VARS[cfg.embedding.provider]):
        missing.append(KEY_VARS[cfg.embedding.provider])
    if missing:
        sys.exit(f"Missing API keys: {', '.join(sorted(set(missing)))}. Add them to .env (see .env.example).")

    from .providers.claude import ClaudeProvider

    claude = ClaudeProvider(cfg.models, budget, cfg.claude_search)
    searchers = []
    for name in cfg.providers:
        if name == "claude":
            searchers.append(claude)
        elif name == "gemini":
            from .providers.gemini import GeminiProvider

            searchers.append(GeminiProvider(cfg.models, budget))
        elif name == "openai":
            from .providers.openai_provider import OpenAIProvider

            searchers.append(OpenAIProvider(cfg.models, budget))
        else:
            sys.exit(f"Unknown provider in config: {name}")

    embedder = make_embedder(cfg.embedding.provider, cfg.embedding.model, budget)
    deduper = Deduper(
        store, embedder, cfg.dedup.duplicate_threshold, cfg.dedup.borderline_threshold,
        same_topic=lambda a, b: claude.same_topic(prompts.same_topic_prompt(a, b)),
    )
    validator = Validator(searchers, cfg.profile, cfg.validation.min_pass_votes, cfg.validation.min_mean_score,
                          cfg.validation.min_novelty)
    review = ReviewStep(cfg, store, budget, claude)
    return TopicLoop(cfg, store, budget, searchers, claude, deduper, validator, review_step=review)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="thesis-finder", description=__doc__)
    parser.add_argument("--config", help="Path to config.yaml (default: config.yaml, else config.example.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the topic-mining loop until a stop rule is met")
    run.add_argument("--rounds", type=int, help="Stop after this many rounds at most")
    run.add_argument("--budget", type=float, help="Override the budget cap in USD for this run")
    run.add_argument("--demo", action="store_true", help="Use offline fake providers (no keys, no cost)")

    lst = sub.add_parser("list", help="List stored topics")
    lst.add_argument("--status", choices=["checked", "valid", "invalid", "parked"])

    rev = sub.add_parser("review", help="Rank the valid topics now and literature-review the top ones")
    rev.add_argument("--budget", type=float, help="Override the budget cap in USD for this review")
    rev.add_argument("--demo", action="store_true", help="Use offline fake providers (no keys, no cost)")

    rep = sub.add_parser("report", help="Write a Markdown shortlist of the best valid topics")
    rep.add_argument("-o", "--output", default="reports/shortlist.md")
    rep.add_argument("-n", "--limit", type=int, default=20)

    sub.add_parser("runs", help="Show past runs and why they stopped")
    sub.add_parser("costs", help="Show spend from the cost log, by provider, job and model")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)

    if args.command == "run":
        if args.budget is not None:
            cfg.loop.budget_usd = args.budget
        if args.demo and cfg.store_path == "data/topics.db":
            cfg.store_path = "data/demo.db"
        loop = build_loop(cfg, demo=args.demo)
        loop.run(max_rounds=args.rounds)
        return

    if args.command == "review":
        if args.budget is not None:
            cfg.loop.budget_usd = args.budget
        if args.demo and cfg.store_path == "data/topics.db":
            cfg.store_path = "data/demo.db"
        build_loop(cfg, demo=args.demo).review_step.run()
        return

    if args.command == "costs":
        print(summarize_log(cfg.cost_log_path))
        return

    store = TopicStore(cfg.store_path)
    if args.command == "list":
        for t in store.all_topics(args.status):
            score = f"{t.mean_score:.1f}" if t.mean_score is not None else "  - "
            print(f"[{t.status:7}] {score}  x{t.times_seen}  {t.title}" + (f"  ({t.reason})" if t.status in ("invalid", "parked") else ""))
    elif args.command == "report":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown_report(store, args.limit))
        print(f"Wrote {args.output}")
    elif args.command == "runs":
        for r in store.runs():
            print(f"#{r['id']} {r['started']} -> {r['ended'] or 'running'}  ${r['spent_usd'] or 0:.2f}  {r['stop_reason'] or ''}")


if __name__ == "__main__":
    main()
