.PHONY: help install install-dev install-all clean clean-all test test-fast test-coverage test-security test-performance lint format type-check build docs docs-build docs-serve docker docker-build docker-run benchmark validate security deploy

# OpenAccelerator - Comprehensive Makefile
# Author: Nik Jois <nikjois@llamasearch.ai>

# Default target
help: ## Show this help message
	@echo "OpenAccelerator - Enterprise-Grade Systolic Array Computing Framework"
	@echo "Author: Nik Jois <nikjois@llamasearch.ai>"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

install-test: ## Install testing dependencies
	pip install -e ".[test]"

install-docs: ## Install documentation dependencies
	pip install -e ".[docs]"

install-all: ## Install all dependencies
	pip install -e ".[dev,test,docs,benchmark]"

# Cleaning targets
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean ## Clean all artifacts including docs
	rm -rf docs/_build/
	rm -rf .benchmarks/

# Testing targets
test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow ones
	python -m pytest tests/ -v --tb=short -m "not slow"

test-coverage: ## Run tests with coverage report
	python -m pytest tests/ --cov=open_accelerator --cov-report=html --cov-report=term-missing --cov-report=xml

test-security: ## Run security tests
	python -m pytest tests/security/ -v
	bandit -r src/open_accelerator/
	safety check

test-performance: ## Run performance benchmarks
	python -m pytest tests/benchmark/ --benchmark-only --benchmark-sort=mean

test-medical: ## Run medical compliance tests
	python -m pytest tests/test_medical.py -v
	python scripts/hipaa_compliance_check.py --output-format json
	python scripts/fda_validation_check.py --output-format json

test-integration: ## Run integration tests
	python -m pytest tests/test_integration.py -v

test-api: ## Run API tests
	python -m pytest tests/test_api.py -v

test-all: test-coverage test-security test-performance ## Run comprehensive test suite

# Code quality targets
lint: ## Run linting
	ruff check src/ tests/
	flake8 src/ tests/

format: ## Format code
	black src/ tests/
	isort src/ tests/
	ruff format src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/
	ruff format --check src/ tests/

type-check: ## Run type checking
	mypy src/open_accelerator/

quality: lint type-check ## Run all code quality checks

# Build targets
build: clean ## Build package
	python -m build

build-wheel: ## Build wheel package
	python -m build --wheel

build-sdist: ## Build source distribution
	python -m build --sdist

# Documentation targets
docs: docs-build ## Build documentation

docs-build: ## Build Sphinx documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8080

docs-clean: ## Clean documentation build
	cd docs && make clean

docs-linkcheck: ## Check documentation links
	cd docs && make linkcheck

# Docker targets
docker: docker-build ## Build Docker image

docker-build: ## Build Docker image
	docker build -t openaccelerator:latest .

docker-build-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t openaccelerator:dev .

docker-run: ## Run Docker container
	docker run -p 8000:8000 openaccelerator:latest

docker-run-dev: ## Run development Docker container
	docker run -p 8000:8000 -v $(PWD):/app openaccelerator:dev

docker-compose-up: ## Start services with Docker Compose
	docker-compose up --build -d

docker-compose-down: ## Stop Docker Compose services
	docker-compose down

docker-compose-logs: ## View Docker Compose logs
	docker-compose logs -f

# Benchmarking and performance
benchmark: ## Run comprehensive benchmarks
	python tools/benchmark_generator.py
	python tools/simulation_runner.py --benchmark-suite benchmarks/generated_suite.json

benchmark-analysis: ## Run performance analysis
	python tools/performance_analyzer.py --results benchmark_results.json

benchmark-generate: ## Generate benchmark configurations
	python tools/benchmark_generator.py --output benchmarks/

# Validation and compliance
validate: ## Run system validation
	python FINAL_SYSTEM_VALIDATION.py

validate-medical: ## Run medical compliance validation
	python scripts/medical_compliance_check.py --output-format json
	python scripts/hipaa_compliance_check.py --output-format json
	python scripts/fda_validation_check.py --output-format json

validate-security: ## Run security validation
	python scripts/generate_security_report.py --output-dir security-report/

health-check: ## Run system health check
	python COMPREHENSIVE_SYSTEM_HEALTH_MONITOR.py

# Security and compliance
security: ## Run security analysis
	bandit -r src/open_accelerator/ -f json -o security_report.json
	safety check --json --output safety_report.json
	semgrep --config=auto src/ --json --output=semgrep_report.json

security-report: ## Generate comprehensive security report
	python scripts/generate_security_report.py

# API and server targets
serve: ## Start FastAPI development server
	uvicorn src.open_accelerator.api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod: ## Start FastAPI production server
	uvicorn src.open_accelerator.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# CLI targets
cli: ## Launch interactive CLI
	python -m open_accelerator.cli

# Release targets
release-check: test-all quality docs ## Pre-release validation
	python -m build
	twine check dist/*

release-test: ## Test release to TestPyPI
	twine upload --repository testpypi dist/*

release: ## Release to PyPI
	twine upload dist/*

# Development workflow targets
dev-setup: install-all ## Complete development setup
	pre-commit install
	@echo "Development environment setup complete!"

dev-test: format lint type-check test-fast ## Quick development test cycle

dev-full: format lint type-check test-all ## Full development validation

# CI/CD simulation
ci-test: ## Simulate CI/CD pipeline locally
	make clean
	make install-all
	make quality
	make test-all
	make build
	make docs-build

# Monitoring and metrics
monitor: ## Start monitoring services
	docker-compose -f docker-compose.monitoring.yml up -d

metrics: ## Generate performance metrics
	python tools/performance_analyzer.py --generate-metrics

# Database and data management
db-setup: ## Setup database for testing
	docker run -d --name openaccelerator-db -p 5432:5432 -e POSTGRES_DB=openaccelerator -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test postgres:13

db-reset: ## Reset test database
	docker exec openaccelerator-db psql -U test -d openaccelerator -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Configuration management
config-validate: ## Validate configuration files
	python -c "from open_accelerator.utils.config import validate_config; validate_config('config.yaml')"

config-generate: ## Generate configuration templates
	python tools/config_generator.py --output-dir config_templates/

# Environment management
env-check: ## Check environment requirements
	python -c "import sys; print(f'Python: {sys.version}'); import platform; print(f'Platform: {platform.platform()}')"
	python -c "import open_accelerator; print(f'OpenAccelerator: {open_accelerator.__version__}')"

env-info: ## Display environment information
	@echo "OpenAccelerator Environment Information"
	@echo "======================================"
	python -c "import sys; print(f'Python Version: {sys.version.split()[0]}')"
	python -c "import platform; print(f'Platform: {platform.platform()}')"
	python -c "import open_accelerator; print(f'OpenAccelerator Version: {open_accelerator.__version__}')"
	@echo "Dependencies:"
	pip list | grep -E "(numpy|fastapi|pytest|docker|sphinx)"

# Production deployment helpers
deploy-check: ## Check deployment readiness
	make ci-test
	make security
	make validate

deploy-docker: ## Deploy with Docker
	docker-compose -f docker-compose.prod.yml up --build -d

deploy-k8s: ## Deploy to Kubernetes
	kubectl apply -f k8s/

# Utility targets
version: ## Show version information
	@python -c "import open_accelerator; print(f'OpenAccelerator v{open_accelerator.__version__}')"

info: ## Show project information
	@echo "OpenAccelerator - Enterprise-Grade Systolic Array Computing Framework"
	@echo "Author: Nik Jois <nikjois@llamasearch.ai>"
	@echo "Version: $(shell python -c 'import open_accelerator; print(open_accelerator.__version__)')"
	@echo "Python: $(shell python --version)"
	@echo "Repository: https://github.com/nikjois/OpenAccelerator"
	@echo "Documentation: https://nikjois.github.io/OpenAccelerator/"

# Workspace management
workspace-setup: ## Setup complete workspace
	make dev-setup
	make db-setup
	make config-generate
	@echo "Workspace setup complete!"

workspace-reset: clean-all ## Reset workspace to clean state
	docker-compose down -v
	docker system prune -f
	@echo "Workspace reset complete!"

# Advanced targets
profile: ## Run performance profiling
	python -m cProfile -o profile.stats -m open_accelerator.cli benchmark
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

memory-profile: ## Run memory profiling
	python -m memory_profiler examples/performance_optimization.py

stress-test: ## Run stress tests
	python tests/stress/test_long_running_simulation.py --duration 1800

load-test: ## Run API load tests
	locust -f tests/load/api_load_test.py --host=http://localhost:8000

# Research and development
research-setup: ## Setup research environment
	jupyter notebook --notebook-dir=notebooks &
	make serve &
	@echo "Research environment started!"

experiment: ## Run experimental features
	python experiments/run_experiment.py

# Final validation
production-ready: ## Validate production readiness
	@echo "Running production readiness validation..."
	make clean
	make install-all
	make test-all
	make security
	make validate
	make build
	make docs-build
	@echo "Production readiness validation complete!"
