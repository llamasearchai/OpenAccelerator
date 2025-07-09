#!/bin/bash
# start.sh

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OPENAI_API_KEY="your-api-key-here"

# Run the application
uvicorn src.open_accelerator.api.main:app --reload --host 0.0.0.0 --port 8000
