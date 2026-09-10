.PHONY: sync lint typecheck test ingest semantic benchmark gate serve ui docker-sandbox

sync:
	uv sync --all-groups

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run mypy

test:
	uv run pytest -m "not integration and not llm" --cov=loom --cov-report=term-missing

ingest:
	uv run python scripts/ingest.py --dataset all

semantic:
	uv run python scripts/build_semantic.py --dataset all

benchmark:
	uv run python scripts/run_benchmark.py --sample 40 --injection all --out outputs/eval/latest

gate:
	uv run python scripts/eval_gate.py --candidate outputs/eval/latest/results.json --baseline benchmarks/baseline_results.json

serve:
	uv run uvicorn loom.api.app:app --reload --port 8000

ui:
	uv run streamlit run src/loom/ui/app.py

docker-sandbox:
	docker build -t loom-sandbox:latest -f sandbox/Dockerfile .
