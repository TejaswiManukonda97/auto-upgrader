# Use a lightweight Python image
FROM python:3.11-slim

# Install system dependencies (Git, build tools)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Keep container alive so we can send multiple exec commands
CMD ["tail", "-f", "/dev/null"]