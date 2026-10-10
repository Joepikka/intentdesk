"""Run with uv run intentdesk prepare/train/predict from the project root."""

import argparse
import json
from pathlib import Path
from urllib.error import URLError


def main() -> None:
    parser = argparse.ArgumentParser(description="IntentDesk: a reproducible ML learning project")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Download pinned data, audit it, and create partitions")
    commands.add_parser("train", help="Train B0/B1 and report tune metrics; do not evaluate final test")
    commands.add_parser("embed", help="Cache frozen MPNet embeddings for fit/tune on this PC")
    commands.add_parser("train-neural", help="Train small/large heads and a linear control on cached features")
    commands.add_parser("prepare-retrieval", help="Download and audit the MS MARCO research corpus")
    commands.add_parser("index-retrieval", help="Build a resumable disk-based BM25 search index")
    commands.add_parser("evaluate-retrieval", help="Evaluate a fixed official development query sample")
    commands.add_parser("evaluate-final", help="Evaluate the frozen classifier once; never select models on final results")
    commands.add_parser("prepare-scope", help="Audit additional OOS examples and freeze the new advisory protocol")
    commands.add_parser("train-scope", help="Fit the separately versioned scope advisory on local CPU")
    commands.add_parser("evaluate-scope", help="Check the frozen scope advisory once; never enables actions")
    search = commands.add_parser("search", help="Search the separate local research passage corpus")
    search.add_argument("text")
    commands.add_parser("build-policy", help="Calibrate frozen candidates and independently check one routing policy")
    server = commands.add_parser("serve", help="Run the free local review app on 127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    pilot_report = commands.add_parser("pilot-report", help="Export recorded manual/assisted pilot outcomes from the running local app")
    pilot_report.add_argument("--port", type=int, default=8765)
    routing = commands.add_parser("route", help="Recommend an intent route or human review using the frozen policy")
    routing.add_argument("text", help="A request, enclosed in quotes")
    neural_inference = commands.add_parser("predict-neural", help="Suggest intents using the larger local neural head")
    neural_inference.add_argument("text", help="A request, enclosed in quotes")
    inference = commands.add_parser("predict", help="Suggest intents using the local baseline")
    inference.add_argument("text", help="A request, enclosed in quotes")
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "pyproject.toml").is_file() or not (root / "src" / "intentdesk").is_dir():
        parser.error("Run this command from the IntentDesk project directory.")
    try:
        if args.command == "serve":
            if not 1024 <= args.port <= 65535:
                parser.error("Choose a port between 1024 and 65535.")
            from intentdesk.app import create_app
            import uvicorn
            uvicorn.run(create_app(root), host="127.0.0.1", port=args.port, workers=1, access_log=False)
            return
        elif args.command == "prepare":
            from intentdesk.data import prepare
            audit = prepare(root)
            result = {"counts": audit["counts"], "exclusions": audit["exclusions"],
                      "test_evaluated": False, "audit": str(root / "reports" / "data_audit.json")}
        elif args.command == "pilot-report":
            from intentdesk.pilot_report import export_report
            result = export_report(root, args.port)
        elif args.command == "train":
            from intentdesk.train import train
            result = train(root)
        elif args.command == "embed":
            from intentdesk.neural import embed
            result = embed(root)
        elif args.command == "train-neural":
            from intentdesk.neural import train_neural
            result = train_neural(root)
        elif args.command == "predict-neural":
            from intentdesk.neural import predict_neural
            result = predict_neural(root, args.text)
        elif args.command == "prepare-retrieval":
            from intentdesk.retrieval_data import prepare_retrieval
            result = prepare_retrieval(root)
        elif args.command == "build-policy":
            from intentdesk.policy import build_policy
            result = build_policy(root)
        elif args.command == "index-retrieval":
            from intentdesk.search import build_index
            result = build_index(root)
        elif args.command == "evaluate-retrieval":
            from intentdesk.search import evaluate_search
            result = evaluate_search(root)
        elif args.command == "evaluate-final":
            from intentdesk.final_eval import evaluate_final
            result = evaluate_final(root)
        elif args.command == "prepare-scope":
            from intentdesk.scope_experiment import prepare_scope
            result = prepare_scope(root)
        elif args.command == "train-scope":
            from intentdesk.scope_experiment import train_scope
            result = train_scope(root)
        elif args.command == "evaluate-scope":
            from intentdesk.scope_experiment import evaluate_scope
            result = evaluate_scope(root)
        elif args.command == "search":
            from intentdesk.search import Searcher
            result = Searcher(root).search(args.text)
        elif args.command == "route":
            from intentdesk.scope import ScopeRouter
            result = ScopeRouter(root).route(args.text)
        else:
            from intentdesk.predict import predict
            result = predict(root, args.text)
    except (ValueError, FileNotFoundError, URLError) as error:
        parser.exit(2, f"IntentDesk: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
