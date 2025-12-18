#!/bin/bash

# 1. Setup Virtual Env
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv sync

# 3. Create mount directory for the sandbox
mkdir -p workspace_mount

# 4. Build the Docker Image manually once
docker build -t agent_sandbox_image sandbox/

# 5. Run the Agent
python agent/main.py