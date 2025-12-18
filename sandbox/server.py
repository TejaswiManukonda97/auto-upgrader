import os
import docker
from fastmcp import FastMCP
from pathlib import Path

# Initialize
mcp = FastMCP("DockerSandbox")

client = docker.from_env()
CONTAINER_NAME = "agent_sandbox"
IMAGE_NAME = "agent_sandbox_image"
MOUNT_DIR = Path.cwd() / "workspace_mount"

def _get_container():
    try:
        return client.containers.get(CONTAINER_NAME)
    except docker.errors.NotFound:
        client.images.build(path="sandbox", tag=IMAGE_NAME)
        return client.containers.run(
            IMAGE_NAME, 
            name=CONTAINER_NAME, 
            detach=True, 
            auto_remove=True,
            volumes={str(MOUNT_DIR): {'bind': '/workspace', 'mode': 'rw'}}
        )

def _sanitize_path(filename: str) -> Path:
    """
    Fixes paths so the agent can say '/workspace/app.py' or 'app.py' 
    and it always resolves to the correct local file.
    """
    clean_name = filename.strip()
    
    # Strip the container prefix if the agent uses it
    if clean_name.startswith("/workspace/"):
        clean_name = clean_name[len("/workspace/"):]
    elif clean_name.startswith("/"):
        clean_name = clean_name[1:]
        
    # Resolve absolute path on the HOST
    target_path = (MOUNT_DIR / clean_name).resolve()
    
    # Security: Ensure we are still inside the MOUNT_DIR
    if not str(target_path).startswith(str(MOUNT_DIR.resolve())):
        raise ValueError(f"Access denied: {filename} attempts to escape workspace.")
        
    return target_path

@mcp.tool
def list_files() -> str:
    """Lists all files in the workspace."""
    try:
        # Use shell to get a recursive list
        container = _get_container()
        res = container.exec_run("find . -maxdepth 2 -not -path '*/.*'", workdir="/workspace")
        return res.output.decode("utf-8")
    except Exception as e:
        return f"Error listing files: {e}"

@mcp.tool
def read_file(filename: str) -> str:
    """Reads the full content of a file."""
    try:
        target_path = _sanitize_path(filename)
        if not target_path.exists():
            return "Error: File not found."
        return target_path.read_text(encoding="utf-8")
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool
def write_file(filename: str, content: str) -> str:
    """Writes content to a file."""
    try:
        target_path = _sanitize_path(filename)
        target_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {filename}"
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error writing file: {e}"

@mcp.tool
def run_shell_command(command: str) -> str:
    """Executes a shell command in the sandbox."""
    container = _get_container()
    try:
        result = container.exec_run(f"bash -c '{command}'", workdir="/workspace")
        output = result.output.decode('utf-8')
        return f"Exit Code {result.exit_code}:\n{output}"
    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()