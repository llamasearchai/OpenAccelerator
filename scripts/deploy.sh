#!/bin/bash

# OpenAccelerator Production Deployment Script
# Author: Nik Jois <nikjois@llamasearch.ai>
# Version: 1.0.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="OpenAccelerator"
DOCKER_IMAGE_NAME="open-accelerator"
DOCKER_TAG="latest"
PORT=8000
ENV_FILE=".env"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 is required but not found"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_status "Docker found: $DOCKER_VERSION"
        DOCKER_AVAILABLE=true
    else
        print_warning "Docker not found - container deployment disabled"
        DOCKER_AVAILABLE=false
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_error "pip3 is required but not found"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ -f "pyproject.toml" ]; then
        print_status "Installing from pyproject.toml..."
        pip install -e .
    elif [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        print_status "Installing basic dependencies..."
        pip install fastapi uvicorn openai numpy matplotlib pytest pydantic
    fi
}

# Setup environment
setup_environment() {
    print_header "Setting Up Environment"
    
    if [ ! -f "$ENV_FILE" ]; then
        print_status "Creating environment file..."
        cat > $ENV_FILE << EOL
# OpenAccelerator Environment Configuration
# Author: Nik Jois <nikjois@llamasearch.ai>

# API Configuration
API_HOST=0.0.0.0
API_PORT=$PORT
API_RELOAD=false
API_LOG_LEVEL=info

# OpenAI Configuration (optional)
# OPENAI_API_KEY=your_openai_api_key_here

# Medical Compliance
ENABLE_MEDICAL_MODE=true
ENABLE_AUDIT_LOGGING=true

# Security
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Performance
MAX_SIMULATION_WORKERS=4
DEFAULT_TIMEOUT=300

# Database (if using persistent storage)
# DATABASE_URL=sqlite:///./accelerator.db

EOL
        print_status "Created $ENV_FILE - please configure as needed"
    else
        print_status "Environment file $ENV_FILE already exists"
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    if [ -f "test_complete_system.py" ]; then
        print_status "Running comprehensive system test..."
        python test_complete_system.py
    fi
    
    if [ -d "tests" ]; then
        print_status "Running pytest test suite..."
        python -m pytest tests/ -v
    fi
}

# Start development server
start_dev_server() {
    print_header "Starting Development Server"
    
    print_status "Starting FastAPI development server on port $PORT..."
    print_status "Access the API at: http://localhost:$PORT"
    print_status "Access API docs at: http://localhost:$PORT/api/v1/docs"
    print_status "Press Ctrl+C to stop the server"
    
    python -m uvicorn src.open_accelerator.api.main:app \
        --host 0.0.0.0 \
        --port $PORT \
        --reload \
        --env-file $ENV_FILE
}

# Build Docker image
build_docker() {
    print_header "Building Docker Image"
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        print_status "Building Docker image: $DOCKER_IMAGE_NAME:$DOCKER_TAG"
        docker build -t $DOCKER_IMAGE_NAME:$DOCKER_TAG .
        print_status "Docker image built successfully"
    else
        print_error "Docker not available - skipping container build"
    fi
}

# Start Docker container
start_docker() {
    print_header "Starting Docker Container"
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        print_status "Starting Docker container..."
        docker run -d \
            --name open-accelerator \
            -p $PORT:$PORT \
            --env-file $ENV_FILE \
            $DOCKER_IMAGE_NAME:$DOCKER_TAG
        
        print_status "Container started successfully"
        print_status "Access the API at: http://localhost:$PORT"
    else
        print_error "Docker not available - skipping container deployment"
    fi
}

# Stop Docker container
stop_docker() {
    print_header "Stopping Docker Container"
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker stop open-accelerator 2>/dev/null || true
        docker rm open-accelerator 2>/dev/null || true
        print_status "Container stopped and removed"
    else
        print_warning "Docker not available"
    fi
}

# Show help
show_help() {
    echo "OpenAccelerator Deployment Script"
    echo "Author: Nik Jois <nikjois@llamasearch.ai>"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup       - Complete setup (venv, deps, env, tests)"
    echo "  dev         - Start development server"
    echo "  test        - Run test suite"
    echo "  docker      - Build and start Docker container"
    echo "  stop        - Stop Docker container"
    echo "  clean       - Clean up build artifacts"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # First-time setup"
    echo "  $0 dev       # Start development server"
    echo "  $0 docker    # Deploy with Docker"
}

# Clean up
clean() {
    print_header "Cleaning Up"
    
    print_status "Removing build artifacts..."
    rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_status "Cleanup complete"
}

# Main script logic
main() {
    case "${1:-setup}" in
        setup)
            check_dependencies
            setup_venv
            install_dependencies
            setup_environment
            run_tests
            print_status "Setup complete! Run '$0 dev' to start the development server"
            ;;
        dev)
            source venv/bin/activate 2>/dev/null || true
            start_dev_server
            ;;
        test)
            source venv/bin/activate 2>/dev/null || true
            run_tests
            ;;
        docker)
            check_dependencies
            build_docker
            start_docker
            ;;
        stop)
            stop_docker
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 