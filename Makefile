# OpenAccelerator Build Automation
# Author: Nik Jois <nikjois@llamasearch.ai>
# Complete automated build, test, and deployment pipeline

.PHONY: help install install-dev test test-fast test-security test-benchmark test-integration \
        clean clean-build clean-test clean-pyc clean-docs \
        build build-dev build-docs build-docker \
        lint format type-check security-check \
        docker-build docker-run docker-stop docker-clean docker-test \
        deploy deploy-dev deploy-prod \
        docs docs-build docs-serve docs-clean \
        benchmark profile coverage \
        requirements freeze-deps check-deps \
        pre-commit setup-dev validate-all

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER_IMAGE := openaccelerator
DOCKER_TAG := latest
DOCKER_REGISTRY := localhost:5000
PORT := 8000
DOCS_PORT := 8080

# Paths
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist
HTMLCOV_DIR := htmlcov

# Default target
help: ## Show this help message
	@echo "OpenAccelerator Build System"
	@echo "Author: Nik Jois <nikjois@llamasearch.ai>"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package in development mode with all dependencies
	$(PIP) install -e .[dev,docs,benchmark,security]
	pre-commit install

setup-dev: install-dev ## Complete development environment setup
	@echo "Setting up development environment..."
	@echo "Installing pre-commit hooks..."
	pre-commit install --hook-type pre-commit
	pre-commit install --hook-type pre-push
	@echo "Development environment setup complete!"

# Testing targets
test: ## Run all tests
	$(PYTEST) -v --tb=short --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast: ## Run fast tests only (excluding slow integration tests)
	$(PYTEST) -v -m "not slow" --tb=short

test-security: ## Run security tests
	$(PYTEST) tests/security/ -v --tb=short

test-benchmark: ## Run benchmark tests
	$(PYTEST) tests/benchmark/ -v --tb=short --benchmark-only

test-integration: ## Run integration tests
	$(PYTEST) tests/test_integration.py -v --tb=short

test-medical: ## Run medical compliance tests
	$(PYTEST) tests/ -v -k "medical" --tb=short

test-agents: ## Run AI agents tests
	$(PYTEST) tests/ -v -k "agent" --tb=short

test-api: ## Run API tests
	$(PYTEST) tests/ -v -k "api" --tb=short

test-all: test test-security test-benchmark test-integration ## Run all test suites

# Code quality targets
lint: ## Run all linting checks
	ruff check $(SRC_DIR) $(TEST_DIR)
	pylint $(SRC_DIR)
	mypy $(SRC_DIR)

format: ## Format code
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)
	ruff format $(SRC_DIR) $(TEST_DIR)

type-check: ## Run type checking
	mypy $(SRC_DIR) --strict

security-check: ## Run security analysis
	bandit -r $(SRC_DIR) -f json -o security-report.json
	safety check --json --output safety-report.json

# Coverage targets
coverage: ## Generate coverage report
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "Coverage report generated in $(HTMLCOV_DIR)/index.html"

coverage-upload: coverage ## Upload coverage to codecov
	codecov -f coverage.xml

# Build targets
build: clean ## Build package
	$(PYTHON) -m build

build-dev: ## Build package for development
	$(PYTHON) -m build --wheel

build-docs: ## Build documentation
	cd $(DOCS_DIR) && make html

build-docker: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

build-docker-dev: ## Build Docker image for development
	docker build -t $(DOCKER_IMAGE):dev -f Dockerfile.dev .

build-all: build build-docs build-docker ## Build everything

# Docker targets
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):latest

docker-run: ## Run Docker container
	docker run -d --name openaccelerator -p $(PORT):$(PORT) $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-dev: ## Run Docker container in development mode
	docker run -it --rm -p $(PORT):$(PORT) -v $(PWD):/app $(DOCKER_IMAGE):dev

docker-stop: ## Stop Docker container
	docker stop openaccelerator || true
	docker rm openaccelerator || true

docker-clean: ## Clean Docker images and containers
	docker stop openaccelerator || true
	docker rm openaccelerator || true
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true
	docker rmi $(DOCKER_IMAGE):latest || true
	docker system prune -f

docker-test: ## Run tests in Docker container
	docker run --rm -v $(PWD):/app $(DOCKER_IMAGE):$(DOCKER_TAG) pytest -v

docker-compose-up: ## Start services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	docker-compose down

docker-compose-test: ## Run tests with docker-compose
	docker-compose run --rm app pytest -v

# Documentation targets
docs: ## Build and serve documentation
	cd $(DOCS_DIR) && make html

docs-build: ## Build documentation
	cd $(DOCS_DIR) && make html

docs-serve: docs-build ## Serve documentation locally
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server $(DOCS_PORT)

docs-clean: ## Clean documentation build
	cd $(DOCS_DIR) && make clean

docs-auto: ## Build docs with auto-reload
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html --host 0.0.0.0 --port $(DOCS_PORT)

# Performance targets
benchmark: ## Run performance benchmarks
	$(PYTEST) tests/benchmark/ --benchmark-only --benchmark-json=benchmark-results.json

profile: ## Run profiling
	$(PYTHON) -m cProfile -o profile.prof examples/comprehensive_simulation.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"

performance-test: ## Run performance tests
	$(PYTEST) tests/benchmark/ -v --benchmark-autosave

# Dependency management
requirements: ## Generate requirements files
	pip-compile pyproject.toml --output-file requirements.txt
	pip-compile pyproject.toml --extra dev --output-file requirements-dev.txt

freeze-deps: ## Freeze current dependencies
	$(PIP) freeze > requirements.lock

check-deps: ## Check for dependency updates
	pip-check
	safety check

update-deps: ## Update dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e .[dev,docs,benchmark,security]

# Clean targets
clean: clean-build clean-test clean-pyc clean-docs ## Clean all build artifacts

clean-build: ## Clean build artifacts
	rm -rf $(BUILD_DIR)/
	rm -rf $(DIST_DIR)/
	rm -rf *.egg-info/
	find . -name '*.egg' -delete

clean-test: ## Clean test artifacts
	rm -rf .pytest_cache/
	rm -rf $(HTMLCOV_DIR)/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.junit.xml
	rm -rf .tox/

clean-pyc: ## Clean Python cache files
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -exec rm -rf {} +

clean-docs: ## Clean documentation artifacts
	cd $(DOCS_DIR) && make clean

clean-docker: ## Clean Docker artifacts
	docker system prune -f
	docker volume prune -f

clean-all: clean clean-docker ## Clean everything including Docker

# Deployment targets
deploy-dev: ## Deploy to development environment
	@echo "Deploying to development environment..."
	docker-compose -f docker-compose.dev.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production environment..."
	docker-compose -f docker-compose.prod.yml up -d

deploy-docker: build-docker ## Deploy Docker image to registry
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# Pre-commit and validation
pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

validate-all: lint type-check security-check test ## Run all validation checks

pre-push: validate-all ## Run pre-push validation

# CI/CD targets
ci-install: ## Install dependencies for CI
	$(PIP) install -e .[dev,docs,benchmark,security]

ci-test: ## Run tests in CI environment
	$(PYTEST) -v --tb=short --cov=$(SRC_DIR) --cov-report=xml --cov-report=term-missing --junit-xml=test-results.xml

ci-lint: ## Run linting in CI environment
	ruff check $(SRC_DIR) $(TEST_DIR) --output-format=github
	mypy $(SRC_DIR) --junit-xml=mypy-results.xml

ci-security: ## Run security checks in CI environment
	bandit -r $(SRC_DIR) -f json -o security-report.json
	safety check --json --output safety-report.json

ci-build: ## Build in CI environment
	$(PYTHON) -m build
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

ci-deploy: ## Deploy in CI environment
	@echo "Deploying in CI environment..."
	# Add deployment commands here

# Development helpers
dev-server: ## Start development server
	uvicorn src.open_accelerator.api.main:app --reload --host 0.0.0.0 --port $(PORT)

dev-jupyter: ## Start Jupyter notebook server
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

dev-docs: ## Start documentation development server
	sphinx-autobuild $(DOCS_DIR) $(DOCS_DIR)/_build/html --host 0.0.0.0 --port $(DOCS_PORT)

# Monitoring and debugging
logs: ## View application logs
	tail -f openaccelerator.log

health-check: ## Check application health
	curl -f http://localhost:$(PORT)/api/v1/health || echo "Health check failed"

monitor: ## Monitor system resources
	htop

# Quick development workflow
quick-test: format lint test-fast ## Quick development test cycle

quick-build: clean build ## Quick build cycle

quick-deploy: build docker-build docker-run ## Quick deployment cycle

# Release targets
release-patch: ## Release patch version
	bumpver update --patch

release-minor: ## Release minor version
	bumpver update --minor

release-major: ## Release major version
	bumpver update --major

# Initialize project
init: install-dev ## Initialize project for development
	@echo "Initializing OpenAccelerator project..."
	@echo "Created by: Nik Jois <nikjois@llamasearch.ai>"
	@echo "Project initialized successfully!"

# Show system information
info: ## Show system information
	@echo "OpenAccelerator Build System Information"
	@echo "========================================"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not installed')"
	@echo "Author: Nik Jois <nikjois@llamasearch.ai>"
	@echo "Project: OpenAccelerator"
	@echo "Version: $(shell $(PYTHON) -c 'import open_accelerator; print(open_accelerator.__version__)' 2>/dev/null || echo 'Not installed')" 