FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    htop \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash researcher && \
    mkdir -p /app && \
    chown -R researcher:researcher /app

USER researcher
WORKDIR /app

# Copy requirements and install Python dependencies
COPY --chown=researcher:researcher pyproject.toml ./
RUN pip install --user --no-cache-dir -e .[dev,medical]

# Copy source code
COPY --chown=researcher:researcher . .

# Install the package
RUN pip install --user -e .

# Expose ports for Jupyter and Dash
EXPOSE 8888 8050

# Default command
CMD ["bash"]
