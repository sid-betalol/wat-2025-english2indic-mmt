.PHONY: help open install test verify lint format clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

open:  ## Open workspace in Cursor with auto-activated environment
	code wat-2025-english2indic-mmt.code-workspace

install:  ## Install dependencies
	uv sync --extra dev

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov

verify:  ## Verify the setup is working correctly
	python verify_setup.py

lint:  ## Run linting
	uv run ruff check .

lint-fix:  ## Fix linting issues
	uv run ruff check . --fix

format:  ## Format code
	uv run ruff format .

format-check:  ## Check formatting without making changes
	uv run ruff format . --check

check-all:  ## Run all checks (linting, formatting, tests)
	uv run ruff check . --fix && uv run ruff format . && uv run pytest

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
