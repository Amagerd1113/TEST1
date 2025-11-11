.PHONY: help install install-dev test test-unit test-integration lint format clean docker-build docker-run docs

# Default target
help:
	@echo "VLA-GR Navigation Framework - Available Commands"
	@echo "=================================================="
	@echo "Installation:"
	@echo "  make install          - Install the package"
	@echo "  make install-dev      - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             - Run linters (flake8, mypy)"
	@echo "  make format           - Format code (black, isort)"
	@echo "  make format-check     - Check code formatting"
	@echo "  make pre-commit       - Run pre-commit hooks"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run Docker container"
	@echo "  make docker-dev       - Run development Docker container"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             - Build documentation"
	@echo "  make docs-serve       - Serve documentation locally"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean            - Remove build artifacts"
	@echo "  make clean-all        - Remove all generated files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-watch:
	pytest-watch tests/ -v

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check src/ tests/

pre-commit:
	pre-commit run --all-files

# Docker
docker-build:
	docker build -t vla-gr:latest .

docker-build-dev:
	docker build --target dev -t vla-gr:dev .

docker-run:
	docker-compose up vla-gr

docker-dev:
	docker-compose up vla-gr-dev

docker-stop:
	docker-compose down

docker-clean:
	docker-compose down -v
	docker rmi vla-gr:latest vla-gr:dev || true

# Documentation
docs:
	cd docs && make html
	@echo "Documentation generated in docs/_build/html/index.html"

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete

clean-all: clean
	rm -rf logs/
	rm -rf outputs/
	rm -rf wandb/
	rm -rf .ruff_cache/

# Development helpers
verify:
	python scripts/verify_installation.py

train:
	python src/training/train.py --config config.yaml

eval:
	python scripts/run_evaluation.py --checkpoint checkpoints/best.pt

demo:
	python demo.py

tensorboard:
	tensorboard --logdir logs/

# CI simulation
ci: format-check lint test
	@echo "All CI checks passed!"
