import asyncio
import os
import time
import json
import logging
import ast
import re
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
MAX_RETRIES = 20

# --- CONFIG ---
TOOL_MAPPING = {
    "run_command": "run_shell_command",
    "cmd": "run_shell_command",
    "read": "read_file",
    "write": "write_file",
    "save": "write_file"
}

ALLOWED_TOOLS = {"run_shell_command", "read_file", "write_file", "list_files", "run_command", "cmd", "read", "write", "save"}

class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    retry_count: int

def parse_tool_calls(text):
    """
    Hybrid Parser: Tries to extract tools from BOTH Python syntax and JSON.
    """
    tool_calls = []
    
    # 1. Try Python Function Calls (The preferred method for complex code)
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("```"): continue
        if "(" in line and line.endswith(")"):
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
                            
                            # Positional args support
                            if node.args:
                                if func_name in ["write_file", "write"]:
                                    if len(node.args) >= 1: args["filename"] = getattr(node.args[0], 'value', getattr(node.args[0], 's', ''))
                                    if len(node.args) >= 2: args["content"] = getattr(node.args[1], 'value', getattr(node.args[1], 's', ''))
                                elif func_name in ["run_shell_command", "run_command"]:
                                    if len(node.args) >= 1: args["command"] = getattr(node.args[0], 'value', getattr(node.args[0], 's', ''))
                                elif func_name in ["read_file", "read"]:
                                    if len(node.args) >= 1: args["filename"] = getattr(node.args[0], 'value', getattr(node.args[0], 's', ''))
                            
                            tool_calls.append({"name": func_name, "args": args, "id": f"py_{int(time.time())}_{len(tool_calls)}"})
            except: pass

    # 2. Try JSON (Fallback for when model reverts to training habits)
    if not tool_calls:
        clean_text = text.replace(r"\'", "'") # Fix escaping
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(clean_text):
            start = clean_text.find('{', pos)
            if start == -1: break
            try:
                obj, end = decoder.raw_decode(clean_text, start)
                if isinstance(obj, dict) and ("name" in obj or "tool" in obj):
                    # Extract fields
                    name = obj.get("name") or obj.get("tool")
                    args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
                    tool_calls.append({"name": name, "args": args, "id": f"json_{int(time.time())}_{len(tool_calls)}"})
                pos = end
            except json.JSONDecodeError:
                pos = start + 1

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

        # --- REASONER ---
        def reasoner(state: AgentState):
            logger.info("--- BRAIN: Thinking... ---")
            if state["retry_count"] >= MAX_RETRIES:
                return {"messages": [AIMessage(content="FATAL: Max retries exceeded.")], "retry_count": state["retry_count"]}

            msg = llm_with_tools.invoke(state["messages"])
            
            # Rescue tool calls using Hybrid Parser
            if not msg.tool_calls:
                extracted = parse_tool_calls(msg.content)
                if extracted:
                    logger.warning(f">>> Rescued {len(extracted)} VALID tool calls!")
                    msg.tool_calls = extracted
            
            return {"messages": [msg], "retry_count": state["retry_count"] + 1}

        # --- REFLECTOR (STRICT ENFORCER) ---
        def reflector(state: AgentState):
            last_msg = state["messages"][-1]
            content = last_msg.content.lower()
            
            # Reject lazy code blocks without execution
            if "```" in last_msg.content and not last_msg.tool_calls:
                logger.warning(">>> Agent hallucinated action. Forcing retry...")
                return {
                    "messages": [HumanMessage(content="Error: You wrote code but didn't execute it. Call 'write_file' now.")],
                    "retry_count": state["retry_count"]
                }

            if "fixed" in content or "passed" in content or "success" in content:
                return {"messages": []}
            
            return {"messages": []}

        # --- EXECUTOR ---
        async def executor(state: AgentState):
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                return {"messages": []}

            results = []
            for tool_call in last_message.tool_calls:
                original_name = tool_call["name"]
                tool_name = TOOL_MAPPING.get(original_name, original_name)
                tool_args = tool_call["args"] or {}

                # Map args
                if "cmd" in tool_args: tool_args["command"] = tool_args.pop("cmd")
                if "path" in tool_args: tool_args["filename"] = tool_args.pop("path")

                logger.info(f"> Action: {tool_name}")
                
                # --- ELLIPSIS GUARD ---
                # Stop "..." lazy write
                if tool_name == "write_file":
                    content = tool_args.get("content", "").strip()
                    if len(content) < 10 or "..." in content:
                        logger.error("!!! REJECTED LAZY WRITE !!!")
                        results.append({
                            "role": "tool", "name": original_name, "tool_call_id": tool_call["id"],
                            "content": "Error: You wrote placeholder text ('...'). You must write the FULL, complete code block."
                        })
                        continue

                selected_tool = next((t for t in tools if t.name == tool_name), None)
                if selected_tool:
                    try:
                        result = await selected_tool.ainvoke(tool_args)
                        content = str(result)
                        logger.info(f"  - Result: {content[:100]}...")
                    except Exception as e:
                        content = f"Error: {e}"
                else:
                    content = f"Tool '{tool_name}' not found."
                
                results.append({"role": "tool", "name": original_name, "tool_call_id": tool_call["id"], "content": content})
            return {"messages": results}

        # --- GRAPH ---
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

        # --- PROMPT ---
        system_instruction = (
            "You are an expert Python Debugging Agent. "
            "You cannot talk. You can ONLY execute tools. "
            
            "YOUR GOAL: Fix 'app.py' so it passes the tests in 'test_app.py'. "
            
            "DEBUGGING PROTOCOL (Follow strictly):"
            "1. EXPLORE: Run 'list_files()' to see the structure."
            "2. TEST: Run the tests first to see the error: run_shell_command(command='python -m unittest test_app.py')"
            "3. INVESTIGATE: "
            "   - Read the failing test file ('test_app.py') to understand EXPECTED output."
            "   - Read the implementation file ('app.py') to understand ACTUAL output."
            "4. ANALYZE: Compare the Error Message against the Code."
            "   - If Error is 'b'...' != '...', you have a Bytes vs String mismatch."
            "   - If Error is 'Module not found', check your imports."
            "5. FIX: Overwrite 'app.py' with the corrected code using 'write_file'."
            "   - IMPORTANT: Write clean, readable, multi-line code. Do NOT put everything on one line."
            "6. VERIFY: Run tests again."
            
            "CRITICAL RULES:"
            "- DO NOT Modify 'test_app.py' (The test is the source of truth)."
            "- Always output Python Function Calls (e.g. write_file(filename='...', content='...'))"
        )
        
        user_task = (
            "Project mounted at '/workspace'. "
            "1. List files. "
            "2. Run 'test_app.py'. "
            "3. Fix 'app.py' based on the errors. "
            "4. Verify the fix."
        )

        logger.info("--- STARTING AUTONOMOUS WORKFLOW ---")
        final_state = await app.ainvoke({
            "messages": [SystemMessage(content=system_instruction), HumanMessage(content=user_task)],
            "retry_count": 0
        })
        
        print("\n--- FINAL OUTPUT ---")
        print(final_state["messages"][-1].content)

    finally:
        if hasattr(client, "close"):
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())