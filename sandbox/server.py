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

def _clean_content(content: str) -> str:
    """Standardizes input from the agent to prevent encoding errors."""
    if not content: return content
    
    # 1. Unescape Unicode (fixes \u0027 -> ')
    if "\\u" in content:
        try: content = content.encode('utf-8').decode('unicode_escape')
        except: pass
        
    # 2. Fix literal newline injections
    content = content.replace("\\n", "\n").replace("\\t", "\t")
    
    # 3. Remove common Markdown wrappers if the agent forgot to strip them
    lines = content.split('\n')
    if lines and lines[0].strip().startswith("```"): lines = lines[1:]
    if lines and lines[-1].strip() == "```": lines = lines[:-1]
    
    return "\n".join(lines)

def _is_valid_python(content: str) -> tuple[bool, str]:
    """
    Parses content to check for syntax errors before saving.
    Returns (True, "") if valid, or (False, error_message) if invalid.
    """
    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError on line {e.lineno}: {e.msg}"
    except Exception as e:
        # If it's not python (e.g. txt, md), we might skip this, 
        # but here we assume mostly python work.
        return True, ""
    
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
    """Writes content to a file. Overwrites existing content."""
    try:
        # --- CLEAN INPUT ---
        content = _clean_content(content)

        # --- LAZY GUARD ---
        # Detects common placeholders that agents use when being lazy
        forbidden_phrases = [
            "your_fixed_code_here",
            "# ... existing code",
            "# ... rest of the code",
            "// ... existing code",
            "TODO: implement",
            "pass  # implementation"
        ]
        
        for phrase in forbidden_phrases:
            if phrase in content:
                return (
                    f"Error: Your code contains a placeholder ('{phrase}'). "
                    "You must write the FULL, EXECUTABLE code. "
                    "Do not abbreviate. Do not use placeholders."
                )
            
        target_path = _sanitize_path(filename)

        # --- SYNTAX GUARD ---
        if filename.endswith(".py"):
            valid, error = _is_valid_python(content)
            if not valid:
                return f"Error: Your code has a syntax error and was NOT saved.\n{error}"
            
        target_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {filename}"
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return f"Error writing file: {e}"
    
@mcp.tool
def replace_in_file(filename: str, find: str, replace: str) -> str:
    """
    Replaces the first occurrence of the 'find' string with the 'replace' string.
    Useful for fixing bugs without overwriting the whole file.
    Constraint: The result must be valid Python code.
    """
    try:
        # --- CLEAN INPUTS ---
        find = _clean_content(find)
        replace = _clean_content(replace)

        # --- RECURSION GUARD ---
        # This stops the agent from fixing "A" with "if X: B else: A", which causes infinite loops.
        if find.strip() in replace and len(replace) > len(find):
             return (
                 "Error: The text you are trying to find is present inside your replacement text.\n"
                 "Risk: This will cause an infinite loop where you fix the code repeatedly.\n"
                 "Solution: Use 'write_file' to overwrite the ENTIRE function instead of patching it."
             )
        
        target_path = _sanitize_path(filename)
        if not target_path.exists():
            return "Error: File not found."
        
        content = target_path.read_text(encoding="utf-8")
        
        # Verify the text exists first
        if find not in content:
            return (
                f"Error: 'find' text not found.\n"
                f"Your input: {find[:50]}...\n"
                f"File content: {content[:50]}...\n"
                "Check for mismatch in whitespace or newlines."
            )
            
        # Perform replacement (only the first occurrence to be safe)
        new_content = content.replace(find, replace, 1)

        # --- SYNTAX GUARD ---
        if filename.endswith(".py"):
            valid, error = _is_valid_python(new_content)
            if not valid:
                return (
                    f"Error: This replacement creates invalid Python syntax (e.g., indentation error).\n"
                    f"Details: {error}\n"
                    "Suggestion: Try replacing the WHOLE function/block instead of just the inner line."
                )
        target_path.write_text(new_content, encoding="utf-8")
        
        return f"Successfully replaced block in {filename}."
    except Exception as e:
        return f"Error: {e}"

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