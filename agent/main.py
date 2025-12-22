import asyncio
import os
import time
import json
import logging
import ast
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

LLM_MODEL = "llama3.1"
MAX_RETRIES = 30 

# --- Add all common aliases ---
TOOL_MAPPING = {
    "run_command": "run_shell_command",
    "cmd": "run_shell_command",
    "read": "read_file",
    "write": "write_file",
    "save": "write_file",
    "create": "write_file",  
    "replace": "replace_in_file",
    "edit": "replace_in_file",
    "edit_file": "replace_in_file", 
    "update": "replace_in_file",
    "branch": "git_create_branch",
    "commit": "git_commit",
    "push": "git_push",
    "pr": "create_github_pr",
    "outdated": "list_outdated_packages",
    "check_updates": "list_outdated_packages"
}

ALLOWED_TOOLS = {
    "run_shell_command", "read_file", "write_file", "list_files", "replace_in_file", 
    "git_create_branch", "git_commit", "git_push", "create_github_pr", "list_outdated_packages",
    "run_command", "cmd", "read", "write", "save", "replace", "edit", "update", "edit_file", "create",
    "branch", "commit", "push", "pr", "outdated", "check_updates"
}

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    retry_count: int

def parse_tool_calls(text):
    """Hybrid Parser: Matches Python calls AND JSON blocks."""
    tool_calls = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line.startswith("//"): continue 
        if line.startswith("```") or "(" not in line or not line.endswith(")"): continue
        try:
            tree = ast.parse(line)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = node.func.id if isinstance(node.func, ast.Name) else None
                    if func_name and func_name in ALLOWED_TOOLS:
                        args = {}
                        for keyword in node.keywords:
                            val = keyword.value
                            if isinstance(val, ast.Constant): args[keyword.arg] = val.value
                            elif isinstance(val, ast.Str): args[keyword.arg] = val.s
                        if node.args:
                            def get_val(n): return getattr(n, 'value', getattr(n, 's', ''))
                            if func_name in ["replace_in_file", "replace", "edit", "update", "edit_file"]:
                                if len(node.args) >= 3: args.update({"filename": get_val(node.args[0]), "find": get_val(node.args[1]), "replace": get_val(node.args[2])})
                            elif func_name in ["write_file", "write", "save", "create"]:
                                if len(node.args) >= 2: args.update({"filename": get_val(node.args[0]), "content": get_val(node.args[1])})
                            elif func_name in ["run_shell_command", "cmd", "run_command"]:
                                if len(node.args) >= 1: args["command"] = get_val(node.args[0])
                            elif func_name in ["read_file", "read"]:
                                if len(node.args) >= 1: args["filename"] = get_val(node.args[0])
                        tool_calls.append({"name": func_name, "args": args, "id": f"py_{int(time.time())}_{len(tool_calls)}"})
        except: pass

    if not tool_calls:
        clean_text = text.replace(r"\'", "'")
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(clean_text):
            start = clean_text.find('{', pos)
            if start == -1: break
            try:
                obj, end = decoder.raw_decode(clean_text, start)
                if isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                    name = obj.get("name") or obj.get("tool")
                    args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
                    tool_calls.append({"name": name, "args": args, "id": f"json_{int(time.time())}_{len(tool_calls)}"})
                pos = end
            except json.JSONDecodeError: pos = start + 1

    return tool_calls

async def main():
    logger.info("--- INITIALIZING AUTONOMOUS AGENT ---")
    
    required_vars = ["GITHUB_TOKEN", "GITHUB_USERNAME", "REPO_OWNER", "REPO_NAME"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"CRITICAL: Missing environment variables: {missing}")
        logger.error("Please add them to your .env file before running.")
        return

    client = MultiServerMCPClient({
        "sandbox": {
            "transport": "stdio",
            "command": "python", 
            "args": ["sandbox/server.py"],
            "env": {
                "FASTMCP_LOG_LEVEL": "WARNING",
                "GITHUB_TOKEN": os.getenv("GITHUB_TOKEN"),
                "GITHUB_USERNAME": os.getenv("GITHUB_USERNAME"),
                "REPO_OWNER": os.getenv("REPO_OWNER"),
                "REPO_NAME": os.getenv("REPO_NAME")
            } 
        }
    })

    try:
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} tools.")
        
        llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=8192)
        llm_with_tools = llm.bind_tools(tools)

        def reasoner(state: AgentState):
            logger.info("--- BRAIN: Thinking... ---")
            if state["retry_count"] >= MAX_RETRIES:
                return {"messages": [AIMessage(content="FATAL: Max retries exceeded.")], "retry_count": state["retry_count"]}

            msg = llm_with_tools.invoke(state["messages"])
            if not msg.tool_calls:
                extracted = parse_tool_calls(msg.content)
                if extracted:
                    logger.warning(f">>> Rescued {len(extracted)} VALID tool calls!")
                    msg.tool_calls = extracted
            return {"messages": [msg], "retry_count": state["retry_count"] + 1}

        def reflector(state: AgentState):
            last_msg = state["messages"][-1]

            if "```" in last_msg.content and not last_msg.tool_calls:
                return {
                    "messages": [HumanMessage(content="Error: You wrote code in the chat but didn't execute it. Use the 'write_file' tool.")],
                    "retry_count": state["retry_count"]
                }

            if last_msg.tool_calls:
                for tool in last_msg.tool_calls:
                    if tool["name"] in ["write_file", "write"]:
                        content = tool["args"].get("content", "").lower()
                        suspicious_keywords = ["calculate_total", "price * quantity", "foo", "bar", "baz", "john doe"]
                        if any(k in content for k in suspicious_keywords):
                            return {
                                "messages": [HumanMessage(content=
                                    "STOP. You are hallucinating generic tutorial code (e.g., 'calculate_total', 'price'). "
                                    "This has nothing to do with the actual file content. "
                                    "READ the file again and fix the ACTUAL code present on disk."
                                )],
                                "retry_count": state["retry_count"]
                            }

                    if tool["name"] in ["git_commit", "commit"]:
                        has_edited = False
                        for msg in state["messages"][-5:]: 
                             if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for past_tool in msg.tool_calls:
                                    if past_tool["name"] in ["write_file", "replace_in_file", "edit", "save"]:
                                        has_edited = True
                        if not has_edited:
                            return {
                                "messages": [HumanMessage(content=
                                    "STOP. You are trying to commit, but you haven't edited any files yet! "
                                    "1. You must update 'requirements.txt' (or similar) to apply the upgrade. "
                                    "2. You must fix any code broken by the upgrade. "
                                    "Use 'write_file' or 'replace_in_file' FIRST."
                                )],
                                "retry_count": state["retry_count"]
                            }
            return {"messages": []}

        async def executor(state: AgentState):
            last_message = state["messages"][-1]
            if not last_message.tool_calls: return {"messages": []}

            results = []
            for tool_call in last_message.tool_calls:
                original_name = tool_call["name"]
                tool_name = TOOL_MAPPING.get(original_name, original_name)
                tool_args = tool_call["args"] or {}

                # =========================================================
                # 1. ARGUMENT NORMALIZATION (The "Translation Layer")
                # =========================================================
                
                # --- General Aliases ---
                if "cmd" in tool_args: tool_args["command"] = tool_args.pop("cmd")
                if "path" in tool_args: tool_args["filename"] = tool_args.pop("path")
                if "file_path" in tool_args: tool_args["filename"] = tool_args.pop("file_path")
                if "file" in tool_args: tool_args["filename"] = tool_args.pop("file")

                # --- Git Tool Fixes ---
                if "branch" in tool_args: tool_args["branch_name"] = tool_args.pop("branch")
                if "name" in tool_args and "branch" in tool_name: tool_args["branch_name"] = tool_args.pop("name")

                # --- Find/Replace Aliases ---
                if "search" in tool_args: tool_args["find"] = tool_args.pop("search")
                if "pattern" in tool_args: tool_args["find"] = tool_args.pop("pattern")
                if "old_string" in tool_args: tool_args["find"] = tool_args.pop("old_string")
                if "old_code" in tool_args: tool_args["find"] = tool_args.pop("old_code")
                if "old" in tool_args: tool_args["find"] = tool_args.pop("old")
                
                if "new_string" in tool_args: tool_args["replace"] = tool_args.pop("new_string")
                if "new_code" in tool_args: tool_args["replace"] = tool_args.pop("new_code")
                if "new" in tool_args: tool_args["replace"] = tool_args.pop("new")
                if "new_line" in tool_args: tool_args["replace"] = tool_args.pop("new_line")
                if "replacement" in tool_args: tool_args["replace"] = tool_args.pop("replacement")

                # --- PR Aliases (THE FIX) ---
                # Fixes pr_title -> title
                if "pr_title" in tool_args: tool_args["title"] = tool_args.pop("pr_title")
                if "name" in tool_args and tool_name == "create_github_pr": tool_args["title"] = tool_args.pop("name")
                
                # Fixes pr_body -> body
                if "pr_body" in tool_args: tool_args["body"] = tool_args.pop("pr_body")
                if "description" in tool_args: tool_args["body"] = tool_args.pop("description")
                if "desc" in tool_args: tool_args["body"] = tool_args.pop("desc")

                # Fixes branch names
                if "source_branch" in tool_args: tool_args["head_branch"] = tool_args.pop("source_branch")
                if "head" in tool_args: tool_args["head_branch"] = tool_args.pop("head")
                if "target_branch" in tool_args: tool_args["base_branch"] = tool_args.pop("target_branch")
                if "base" in tool_args: tool_args["base_branch"] = tool_args.pop("base")

                # INJECT DEFAULTS (Safety Net)
                if tool_name == "create_github_pr":
                    if "title" not in tool_args:
                        tool_args["title"] = "Automated Library Upgrade"
                    if "body" not in tool_args:
                        tool_args["body"] = "This PR was created automatically by the Auto-Upgrader Agent."
                    if "head_branch" not in tool_args:
                        # Fallback: assume the agent pushed to the correct branch previously
                        tool_args["head_branch"] = "feat/upgrade-deps" 

                # Cleanup line numbers
                if "line_number" in tool_args: tool_args.pop("line_number")
                if "line" in tool_args: tool_args.pop("line")

                # =========================================================
                # 2. EXECUTION
                # =========================================================
                logger.info(f"> Action: {tool_name}")
                
                if tool_name == "write_file":
                    content = tool_args.get("content", "").strip()
                    if len(content) < 10 or "..." in content:
                        results.append({"role": "tool", "name": original_name, "tool_call_id": tool_call["id"], "content": "Error: You wrote placeholder text ('...'). Write the FULL code."})
                        continue

                selected_tool = next((t for t in tools if t.name == tool_name), None)
                if selected_tool:
                    try:
                        result = await selected_tool.ainvoke(tool_args)
                        content = str(result)
                        logger.info(f"  - Result: {content[:100]}...")
                    except Exception as e:
                        content = f"Error invoking tool {tool_name}: {e}\n(Tip: Check arguments. You passed: {list(tool_args.keys())})"
                else: content = f"Tool '{tool_name}' not found."
                
                results.append({"role": "tool", "name": original_name, "tool_call_id": tool_call["id"], "content": content})
            return {"messages": results}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", reasoner)
        workflow.add_node("reflector", reflector)
        workflow.add_node("tools", executor)
        
        workflow.set_entry_point("agent")
        
        def should_continue(state):
            last_msg = state["messages"][-1]
            if state["retry_count"] > MAX_RETRIES: return "end"
            if last_msg.tool_calls: return "tools"
            return "reflector"

        def after_reflector(state):
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage): return "agent"
            return "end"

        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "reflector": "reflector", "end": END})
        workflow.add_conditional_edges("reflector", after_reflector, {"agent": "agent", "end": END})
        workflow.add_edge("tools", "agent") 
        
        app = workflow.compile() 

        # --- 3. SYSTEM PROMPT FIX: Handle Missing Requirements ---
        system_instruction = (
            "You are an expert Python Auto-Upgrader Agent. "
            "You cannot talk. You can ONLY execute tools. "

            "CORE BEHAVIOR:"
            "1. SELF-CORRECTION: If you upgrade a library and 'pytest' fails, fix the code."
            "2. MANDATORY CLOSURE: You are NOT done until you have generated a GitHub PR URL."
            "   - If tests pass, you MUST immediately Commit, Push, and Create PR."
            "   - DO NOT just say 'A PR can be created'. DO IT."

            "TROUBLESHOOTING PROTOCOLS:"
            "1. MISSING FILE: If 'Could not open requirements file', ACTION: write_file('requirements.txt', content='requests==2.32.0')"
            "2. MISSING TOOLS: If 'pytest' is missing, ACTION: run_shell_command('pip install pytest')"
            "3. DEPENDENCY CONFLICTS: If pip install fails, loosen the version pins."

            "WORKFLOW PROTOCOL:"
            "1. SETUP: Create branch 'feat/upgrade-deps' using 'git_create_branch'."
            "2. DISCOVER: Use 'list_outdated_packages'."
            "3. UPGRADE: Update 'requirements.txt' (Create it if it doesn't exist!)."
            "4. INSTALL: Run 'pip install -r requirements.txt'."
            "5. TEST: Run 'run_shell_command' (e.g., 'pytest'). "
            "   - IF FAIL: Fix code or dependencies until tests pass."
            "6. PUBLISH: Use 'git_commit' (ONLY if tests pass) -> 'git_push' -> 'create_github_pr'."
            
            "CRITICAL RULES:"
            "1. HANDLE TEST FAILURES: If 'run_shell_command' returns 'Exit Code 1', DO NOT COMMIT. Fix first."
            "2. NO GHOST UPGRADES: You must physically edit/create 'requirements.txt'."
            "3. IGNORE GIT PUSH OUTPUT: The 'git_push' tool returns a URL ending in '/pull/new/...'. "
            "   THIS IS NOT A VALID PR. It is just a suggestion."
            "   YOU MUST EXECUTE 'create_github_pr' to actually create the PR."
            "   Your final output must be the URL returned by 'create_github_pr', NOT 'git_push'."
            "3. IDEMPOTENCY: If 'create_github_pr' says 'PR already exists', treat it as Success."
            "4. NEVER push to 'main'."

            "CRITICAL ABORT CONDITIONS:\n"
            "1. IF you see 'fatal: not a git repository': STOP.\n"
            "2. IF you see 'Recursion limit reached': STOP."
        )
        
        user_task = (
            "Project mounted at '/workspace'. "
            "GOAL: Upgrade the 'requests' library to the latest version and ensure the app is stable."
            
            "EXECUTION PLAN:"
            "1. DISCOVER: Check which version is currently installed."
            "2. UPGRADE: Edit 'requirements.txt' to pin the NEW version. If the file is missing, CREATE it."
            "3. STABILITY LOOP: Run tests ('pytest')."
            "   - IF FAIL: Fix the application code."
            "   - REPEAT until tests pass."
            "4. FINALIZE: Create a PR only when tests pass."

            "SUCCESS CRITERIA:"
            "1. 'requests' version is updated in requirements.txt."
            "2. 'pytest' passes with Exit Code 0."
            "3. A Pull Request is created on GitHub."

            "IMPORTANT: Do not stop at Step 2. You must execute Step 3."
        )

        logger.info("--- STARTING AUTONOMOUS UPGRADER WORKFLOW ---")
        
        final_state = await app.ainvoke(
            {"messages": [SystemMessage(content=system_instruction), HumanMessage(content=user_task)], "retry_count": 0},
            {"recursion_limit": 50} 
        )
        
        print("\n--- FINAL OUTPUT ---")
        print(final_state["messages"][-1].content)

    finally:
        if hasattr(client, "close"):
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())