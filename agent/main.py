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
MAX_RETRIES = 30  # Increased for difficult debugging

TOOL_MAPPING = {
    "run_command": "run_shell_command",
    "cmd": "run_shell_command",
    "read": "read_file",
    "write": "write_file",
    "save": "write_file",
    "replace": "replace_in_file",
    "edit": "replace_in_file"
}

ALLOWED_TOOLS = {
    "run_shell_command", "read_file", "write_file", "list_files", "replace_in_file", 
    "run_command", "cmd", "read", "write", "save", "replace", "edit"
    }

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    retry_count: int

def parse_tool_calls(text):
    """Hybrid Parser: Matches Python calls AND JSON blocks."""
    tool_calls = []
    
    # 1. Python Syntax Parser
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line.startswith("//"): continue # Skip comments
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
                        # Handle positional args if any (simple cases)
                        if node.args:
                            def get_val(n): return getattr(n, 'value', getattr(n, 's', ''))
                            if func_name in ["replace_in_file", "replace"]:
                                if len(node.args) >= 3: args.update({"filename": get_val(node.args[0]), "find": get_val(node.args[1]), "replace": get_val(node.args[2])})
                            elif func_name in ["write_file", "write"]:
                                if len(node.args) >= 2: args.update({"filename": get_val(node.args[0]), "content": get_val(node.args[1])})
                            elif func_name in ["run_shell_command", "cmd"]:
                                if len(node.args) >= 1: args["command"] = get_val(node.args[0])
                            elif func_name in ["read_file", "read"]:
                                if len(node.args) >= 1: args["filename"] = get_val(node.args[0])
                        tool_calls.append({"name": func_name, "args": args, "id": f"py_{int(time.time())}_{len(tool_calls)}"})
        except: pass

    # 2. JSON Parser (Fallback)
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
    
    client = MultiServerMCPClient({
        "sandbox": {
            "transport": "stdio",
            "command": "python", 
            "args": ["sandbox/server.py"],
            "env": {"FASTMCP_LOG_LEVEL": "WARNING"} 
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

            # 1. Check for Lazy Placeholders
            if "```" in last_msg.content and not last_msg.tool_calls:
                return {
                    "messages": [HumanMessage(content="Error: You wrote code in the chat but didn't execute it. Use the 'write_file' tool.")],
                    "retry_count": state["retry_count"]
                }

            # 2. Check for "Tutorial Hallucinations" in tool calls
            if last_msg.tool_calls:
                for tool in last_msg.tool_calls:
                    if tool["name"] in ["write_file", "write"]:
                        content = tool["args"].get("content", "").lower()

                        # Generic keywords that rarely appear in production bugs unless the app IS a calculator
                        suspicious_keywords = ["calculate_total", "price * quantity", "foo", "bar", "baz", "john doe"]

                        # If these appear, but weren't in the conversation history, it's likely a hallucination
                        if any(k in content for k in suspicious_keywords):
                            # We check if these keywords were actually relevant to the user's task
                            # (A simple heuristic: if the user didn't mention them, block them)
                            return {
                                "messages": [HumanMessage(content=
                                    "STOP. You are hallucinating generic tutorial code (e.g., 'calculate_total', 'price'). "
                                    "This has nothing to do with the actual file content. "
                                    "READ the file again and fix the ACTUAL code present on disk."
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

                if "cmd" in tool_args: tool_args["command"] = tool_args.pop("cmd")
                if "path" in tool_args: tool_args["filename"] = tool_args.pop("path")
                if "old" in tool_args: tool_args["find"] = tool_args.pop("old")
                if "new" in tool_args: tool_args["replace"] = tool_args.pop("new")

                logger.info(f"> Action: {tool_name}")
                
                # Ellipsis Guard
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
                    except Exception as e: content = f"Error: {e}"
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

        system_instruction = (
            "You are an expert Python Debugging Agent. "
            "You cannot talk. You can ONLY execute tools. "

            "CORE PHILOSOPHY:"
            "1. PRESERVE CONTEXT: You are fixing specific files, not writing new ones from scratch."
            "   - Before editing 'app.py', you MUST read 'app.py'."
            "   - Your code MUST use the same variable names, libraries, and logic style as the existing file."
            "   - DO NOT hallucinate generic examples (like 'calculate_total', 'foo', 'bar') if they are not in the file."

            "2. ATOMIC EDITS: Prefer 'replace_in_file' for small fixes. Use 'write_file' only if re-writing a messy function."
            "   - If you use 'write_file', you must output the FULL, VALID code."

            "3. VERIFICATION: After every fix, run the test script immediately."

            "DEBUGGING LOOP:"
            "1. EXPLORE (list_files) -> 2. TEST (run test) -> 3. READ (read_file) -> 4. PLAN & FIX -> 5. VERIFY"
        )
        
        user_task = (
            "Project mounted at '/workspace'. "
            "1. List files. "
            "2. Run 'test_app.py'. "
            "3. Fix 'app.py' based on the errors. "
            "4. Verify the fix."
        )

        logger.info("--- STARTING AUTONOMOUS WORKFLOW ---")
        
        # --- FIX: Set explicit recursion limit here ---
        final_state = await app.ainvoke(
            {"messages": [SystemMessage(content=system_instruction), HumanMessage(content=user_task)], "retry_count": 0},
            {"recursion_limit": 30} 
        )
        
        print("\n--- FINAL OUTPUT ---")
        print(final_state["messages"][-1].content)

    finally:
        if hasattr(client, "close"):
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())