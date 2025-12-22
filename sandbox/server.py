import os
from dotenv import load_dotenv
import docker
from fastmcp import FastMCP
from pathlib import Path
import ast
import textwrap
import requests

load_dotenv()

# Load env vars
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USER = os.getenv("GITHUB_USERNAME")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")

mcp = FastMCP("DockerSandbox")
client = docker.from_env()

# --- CONFIGURATION ---
CONTAINER_NAME = "agent_sandbox"
IMAGE_NAME = "agent_sandbox_image"

# FIX: Use a dedicated subfolder to avoid polluting the agent's own code
# AND ensure Docker mounts this specific folder.
MOUNT_DIR = (Path.cwd() / "workspace_mount").resolve()

# Ensure the mount directory exists
if not MOUNT_DIR.exists():
    MOUNT_DIR.mkdir(parents=True)
    print(f"Created workspace directory at: {MOUNT_DIR}")

def _get_container():
    try:
        container = client.containers.get(CONTAINER_NAME)
        if container.status != "running":
            container.start()
        return container
    except docker.errors.NotFound:
        print(f"Starting container with mount: {MOUNT_DIR} -> /workspace")
        client.images.build(path="sandbox", tag=IMAGE_NAME)
        return client.containers.run(
            IMAGE_NAME, 
            name=CONTAINER_NAME, 
            detach=True, 
            auto_remove=True,
            # CRITICAL: This binds the local folder to the docker folder
            volumes={str(MOUNT_DIR): {'bind': '/workspace', 'mode': 'rw'}}
        )

def _sanitize_path(filename: str) -> Path:
    """Ensures file operations happen INSIDE the workspace_mount."""
    clean_name = filename.strip()
    if clean_name.startswith("/workspace/"):
        clean_name = clean_name[len("/workspace/"):]
    elif clean_name.startswith("/"):
        clean_name = clean_name[1:]
    
    target_path = (MOUNT_DIR / clean_name).resolve()
    
    # Security: Prevent escaping the workspace_mount
    if not str(target_path).startswith(str(MOUNT_DIR)):
        raise ValueError(f"Access denied: {filename} is outside the workspace.")
        
    return target_path

def _clean_content(content: str) -> str:
    if not content: return content
    if "\\u" in content:
        try: content = content.encode('utf-8').decode('unicode_escape')
        except: pass
    content = content.replace("\\n", "\n").replace("\\t", "\t")
    lines = content.split('\n')
    if lines and lines[0].strip().startswith("```"): lines = lines[1:]
    if lines and lines[-1].strip() == "```": lines = lines[:-1]
    return "\n".join(lines)

def _is_valid_python(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError on line {e.lineno}: {e.msg}"
    except Exception as e:
        return True, ""

def _setup_git_config(container):
    """Ensures git is usable inside the container."""
    container.exec_run("git config --global --add safe.directory /workspace")
    container.exec_run(f"git config --global user.email '{GITHUB_USER}@bot.com'")
    container.exec_run(f"git config --global user.name '{GITHUB_USER}'")

# --- TOOLS ---

@mcp.tool
def list_files() -> str:
    """Lists all files in the workspace."""
    container = _get_container()
    try:
        res = container.exec_run("find . -maxdepth 2 -not -path '*/.*'", workdir="/workspace")
        return res.output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def read_file(filename: str) -> str:
    """Reads the full content of a file."""
    try:
        target_path = _sanitize_path(filename)
        if not target_path.exists():
            return "Error: File not found."
        return target_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def write_file(filename: str, content: str) -> str:
    """Writes content to a file. Overwrites existing content."""
    try:
        content = _clean_content(content)
        target_path = _sanitize_path(filename)
        # Check syntax if python
        if filename.endswith(".py"):
            valid, error = _is_valid_python(content)
            if not valid:
                return f"Error: Invalid Python syntax. {error}"
        
        target_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def replace_in_file(filename: str, find: str, replace: str) -> str:
    """
    Replaces the first occurrence of the 'find' string with the 'replace' string.
    Useful for fixing bugs without overwriting the whole file.
    Constraint: The result must be valid Python code.
    """
    try:
        target_path = _sanitize_path(filename)
        if not target_path.exists(): return "Error: File not found."
        
        content = target_path.read_text(encoding="utf-8")
        find_clean = textwrap.dedent(find).strip()
        replace_clean = textwrap.dedent(replace).strip()
        
        # Fuzzy search logic
        file_lines = content.split('\n')
        find_lines = find_clean.split('\n')
        start_index = -1
        detected_indent = ""
        
        for i in range(len(file_lines)):
            if file_lines[i].strip() == find_lines[0].strip():
                match = True
                current_indent = file_lines[i][:len(file_lines[i]) - len(file_lines[i].lstrip())]
                for j in range(len(find_lines)):
                    if i + j >= len(file_lines) or file_lines[i+j].strip() != find_lines[j].strip():
                        match = False
                        break
                if match:
                    start_index = i
                    detected_indent = current_indent
                    break
        
        if start_index == -1: return "Error: Text to replace not found."
        
        replace_lines = replace_clean.split('\n')
        indented_replace = [detected_indent + line for line in replace_lines]
        
        new_file_lines = file_lines[:start_index] + indented_replace + file_lines[start_index + len(find_lines):]
        new_content = "\n".join(new_file_lines)
        
        if filename.endswith(".py"):
            valid, error = _is_valid_python(new_content)
            if not valid: return f"Error: Resulting code has syntax errors: {error}"
            
        target_path.write_text(new_content, encoding="utf-8")
        return "Success: File updated."
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def run_shell_command(command: str) -> str:
    """Executes a shell command in the sandbox."""
    container = _get_container()
    try:
        result = container.exec_run(f"bash -c '{command}'", workdir="/workspace")
        return f"Exit Code {result.exit_code}:\n{result.output.decode('utf-8')}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def list_outdated_packages(package_name: str = "") -> str:
    """Lists outdated packages. If package_name is specified, returns only that package."""
    container = _get_container()
    try:
        # Ignore package_name arg to prevent crashes
        res = container.exec_run("pip list --outdated --format=json", workdir="/workspace")
        return res.output.decode("utf-8")
    except Exception as e:
        return f"Error: {e}"

@mcp.tool
def git_clone(repo_url: str = "") -> str:
    """
    Clones the repository into the current workspace.
    WARNING: This deletes all existing files in the workspace to ensure a clean clone.
    
    Args:
        repo_url: Optional. If not provided, constructs URL from env vars.
    """
    if not GITHUB_TOKEN: return "Error: GITHUB_TOKEN not set."
    
    # 1. Construct URL if missing
    if not repo_url:
        # Use the env vars defined at the top of server.py
        repo_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}.git"
    
    # 2. Inject Token for Authentication
    if "https://" in repo_url:
        auth_url = repo_url.replace("https://", f"https://{GITHUB_TOKEN}@")
    else:
        auth_url = repo_url # Fallback, likely won't work for private repos without https

    container = _get_container()
    _setup_git_config(container)
    
    try:
        # 3. CLEANUP: Delete existing files to allow 'git clone .' to work
        # Safety: We strictly perform this inside /workspace
        clean_cmd = "find . -mindepth 1 -delete" 
        container.exec_run(f"bash -c '{clean_cmd}'", workdir="/workspace")

        # 4. CLONE
        # We clone into '.' (current dir) because the mount is the root of the project
        clone_cmd = f"git clone {auth_url} ." 
        res = container.exec_run(f"bash -c '{clone_cmd}'", workdir="/workspace")
        
        if res.exit_code != 0:
            return f"Git Clone Failed: {res.output.decode('utf-8')}"
            
        return f"Success: Repository cloned. \nOutput: {res.output.decode('utf-8')}"

    except Exception as e:
        return f"Error during clone: {e}"
    
@mcp.tool
def git_create_branch(branch_name: str) -> str:
    """Creates and switches to a new git branch. Verifies the switch."""
    container = _get_container()
    _setup_git_config(container)
    try:
        # Auto-initialize if missing
        if not (MOUNT_DIR / ".git").exists():
            container.exec_run("git init", workdir="/workspace")
            container.exec_run("git checkout -b main", workdir="/workspace")
            
        # 1. Try create (-b)
        res = container.exec_run(f"git checkout -b {branch_name}", workdir="/workspace")
        
        # 2. If fail, try switch (maybe it exists)
        if res.exit_code != 0:
            res = container.exec_run(f"git checkout {branch_name}", workdir="/workspace")

        # 3. VERIFY: Are we actually on the branch?
        status = container.exec_run("git branch --show-current", workdir="/workspace")
        current = status.output.decode('utf-8').strip()
        
        if current != branch_name:
            return f"Error: Failed to switch. Git is still on '{current}'. Output: {res.output.decode('utf-8')}"

        return f"Git: Successfully switched to '{branch_name}'"
    except Exception as e:
        return f"Git Error: {e}"

@mcp.tool
def git_commit(message: str) -> str:
    container = _get_container()
    _setup_git_config(container)
    try:
        container.exec_run("git add .", workdir="/workspace")
        res = container.exec_run(f"git commit -m '{message}'", workdir="/workspace")
        return res.output.decode('utf-8')
    except Exception as e:
        return f"Git Commit Error: {e}"

@mcp.tool
def git_push(branch_name: str) -> str:
    """Pushes the current branch to GitHub. Does not push to main/master."""
    if not GITHUB_TOKEN: return "Error: GITHUB_TOKEN missing."

    if branch_name.lower() in ["main", "master"]:
        return (
            "Error: PROTECTED BRANCH VIOLATION. "
            "You are attempting to push to 'main' or 'master'. THIS IS FORBIDDEN. "
            "You must push to your feature branch (e.g., 'feat/upgrade-deps')."
        )
    
    container = _get_container()
    _setup_git_config(container)
    remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
    try:
        # Force push support for testing
        cmd = f"git push {remote_url} {branch_name}"
        res = container.exec_run(f"bash -c '{cmd}'", workdir="/workspace")
        return f"Push Result: {res.output.decode('utf-8')}"
    except Exception as e:
        return f"Git Push Error: {e}"

@mcp.tool
def create_github_pr(title: str, body: str, head_branch: str, base_branch: str = "main") -> str:
    """
    Creates a Pull Request. Handles duplicates gracefully.
    
    Args:
        repo: The name of the repository to create the PR in
        title: The title of the PR
        body: Description of the PR
        head_branch: The branch to merge from
        base_branch: The branch to merge into
    """
    if not GITHUB_TOKEN: return "Error: GITHUB_TOKEN not set."
    
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"title": title, "body": body, "head": head_branch, "base": base_branch}
    
    try:
        resp = requests.post(url, json=data, headers=headers)
        
        # 1. Success (201 Created)
        if resp.status_code == 201:
            return f"Success! PR Created: {resp.json().get('html_url')}"
            
        # 2. Handle 422 (Validation Failed)
        elif resp.status_code == 422:
            error_json = resp.json()
            error_msg = str(error_json).lower() # Lowercase for easier matching
            
            # BROAD MATCH for duplicates
            # GitHub error messages for duplicates usually contain "exists" or "already exists"
            if "exist" in error_msg:
                 return f"Success! PR already exists for {head_branch}. (Ignoring 422 error)"

            # Check if it's the "No commits between" error (also a form of success/done)
            if "no commits" in error_msg:
                return "Success: No new commits to merge. PR not needed."

            return f"Error creating PR (422): {error_json}"
            
        else:
            return f"Error creating PR ({resp.status_code}): {resp.text}"
            
    except Exception as e:
        return f"Request Error: {e}"

if __name__ == "__main__":
    mcp.run()