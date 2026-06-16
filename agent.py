import os
import json
from openai import OpenAI

# Initialize the client (it automatically looks for the OPENAI_API_KEY env variable)
client = OpenAI()

# 1. Define the tools the agent can use
def calculate(expression: str) -> str:
    """A safe calculator tool to evaluate basic math expressions."""
    try:
        # Using a restricted dict for basic safety in this simple example
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Error: Invalid characters in expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Map tool names to the actual Python functions
AVAILABLE_TOOLS = {
    "calculate": calculate
}

# 2. Define the tool metadata for the LLM
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression string (e.g., '2 + 2' or '5 * (10 - 3)').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to solve.",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

# 3. The Agent Loop
def run_agent(user_prompt: str):
    print(f"\n[User]: {user_prompt}")
    
    # Initialize the conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when necessary to provide accurate answers."},
        {"role": "user", "content": user_prompt}
    ]
    
    # First LLM call: Present the prompt and the available tools
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Using a fast, smart, tool-capable model
        messages=messages,
        tools=tools_definition,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    # Check if the model decided it needs to use a tool
    if response_message.tool_calls:
        print("[Agent]: Thinking... I need to use a tool.")
        
        # Add the model's tool call request to the history
        messages.append(response_message)
        
        # Process each tool call requested by the model
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"[Agent]: Calling tool '{tool_name}' with arguments: {tool_args}")
            
            # Execute the tool
            tool_function = AVAILABLE_TOOLS[tool_name]
            tool_result = tool_function(expression=tool_args.get("expression"))
            
            print(f"[Tool Result]: {tool_result}")
            
            # Feed the tool's result back to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_result
            })
            
        # Second LLM call: Provide the tool results so the LLM can generate the final answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        print(f"[Agent Final Answer]: {final_response.choices[0].message.content}")
        
    else:
        # If no tool was needed, just print the direct response
        print(f"[Agent Direct Answer]: {response_message.content}")

# --- Test the Agent ---
if __name__ == "__main__":
    # Test Case 1: Doesn't require a tool
    run_agent("What is the capital of France?")
    
    print("-" * 40)
    
    # Test Case 2: Requires a tool
    run_agent("What is 1452 multiplied by 43, plus 12? Use your calculator.")