from model_example_query import query_vlm, query_llm
from search_frame_captions import search_captions
from prompts import initial_prompt, followup_prompt, finish_prompt
from captions.global_summary import global_summary

import json
import os
from together import Together
from google import genai
import asyncio

with open("/resource/claude_runs/1qtest/env.json", "r") as f:
    env_data = json.load(f)
    together_key_PRIV = env_data["together_key"]
    gemini_key_PRIV = env_data["gemini_key"]

os.environ['TOGETHER_API_KEY'] = together_key_PRIV
os.environ['GEMINI_API_KEY'] = gemini_key_PRIV
client_together = Together()
client = genai.Client(api_key=gemini_key_PRIV)

# Define the tools available for the model
tools = [
    {
        "type": "function",
        "function": {
            "name": "vlm_query",
            "description": "Query a Vision Language Model to analyze specific frames from the video",
            "parameters": {
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of frame paths to analyze (e.g., ['frames/frame_0100.jpg', 'frames/frame_0200.jpg'])"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/question to ask about the frames"
                    }
                },
                "required": ["frames", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "caption_search",
            "description": "Search through video frame captions to find relevant scenes",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant frame captions"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer to the question",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the question"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]

class Pipeline:
    def __init__(self, llm_model_name, vlm_model_name, max_num_iterations=10):
        # Store model names
        self.llm_model_name = llm_model_name
        self.vlm_model_name = vlm_model_name
        
        # Create client objects with model names embedded
        self.llm = llm_model_name
        self.vlm = vlm_model_name
        
        self.max_num_iterations = max_num_iterations
        self.scratchpad = []
        self.messages = []
    
    def llm_query_with_tools(self, messages, tools=None, tool_choice=None):
        """Query LLM with tool calling support"""
        try:
            kwargs = {
                "model": self.llm,
                "messages": messages,
                "stream": False
            }
            
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
                
            response = client_together.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            print(f"Error querying {self.llm}: {e}")
            return None
    
    def llm_query(self, prompt):
        return query_llm(self.llm, prompt)
    
    async def vlm_query(self, image_paths, prompt):
        return await query_vlm(self.vlm, image_paths, prompt)
        

async def query_model_iterative(model, question):
    """Iteratively query any open-source model to answer questions about video
    
    Args:
        question: The question to answer
        model: Pipeline object with LLM and VLM models
        max_num_iterations: Maximum iterations for reasoning
    """
    question = question.strip()
    
    # Initialize conversation with system and user messages
    conversation_messages = [
        {
            "role": "system", 
            "content": "You are an expert at reasoning and tool-using, with the goal of answering this question about a long video. You should be able to extract detailed frame-information from videos, do caption searches, and use your findings to answer the question. You should be SUPER PICKY about your findings, NOT make assumptions, and always bias towards gathering more evidence before executing a final answer. Use EXACT evidence only."
        },
        {
            "role": "user", 
            "content": f"\nHere is a global summary of the video for general context: {global_summary}\n\nYour question is this: {question}\n\n{initial_prompt(question)}"
        }
    ]
    
    model.messages = conversation_messages.copy()
    
    for i in range(model.max_num_iterations):
        print("="*60 + f" Iteration {i} " + "="*60)
        
        # Query the model with tools
        response = model.llm_query_with_tools(
            messages=conversation_messages,
            tools=tools,
            tool_choice="auto"  # Let the model decide which tool to use
        )
        
        if not response:
            print(f"Failed to get response at iteration {i+1}")
            continue
        
        # Get the assistant's message
        assistant_message = response.choices[0].message
        conversation_messages.append(assistant_message.model_dump())
        model.messages.append(assistant_message.model_dump())
        
        # Check if the model wants to call a tool
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"Tool call: {function_name}")
                print(f"Arguments: {function_args}")
                
                # Execute the appropriate tool
                if function_name == "vlm_query":
                    print("="*60 + " Querying VLM " + "="*60)
                    frames = function_args.get("frames", [])
                    prompt = f"Here is a global summary of the video for general context: {global_summary}\n{function_args.get('prompt', '')}"
                    print(f"PROMPT: {prompt}")
                    
                    retrieved_info = await model.vlm_query(frames, prompt)
                    tool_result = json.dumps(retrieved_info) if retrieved_info else "Failed to query VLM"
                    
                    # Add tool response to conversation
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    }
                    conversation_messages.append(tool_message)
                    model.messages.append(tool_message)
                    
                elif function_name == "caption_search":
                    search_query = function_args.get("query", "")
                    print(f"Caption search query: {search_query}")
                    
(search_query, "captions/frame_captions_sorted_embeddings.jsonl", 30)
                    
                    # Convert list results to string
                    if isinstance(retrieved_info, list):
                        retrieved_info = json.dumps(retrieved_info, indent=2)
                    
                    # Add tool response to conversation
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": retrieved_info
                    }
                    conversation_messages.append(tool_message)
                    model.messages.append(tool_message)
                    
                elif function_name == "final_answer":
                    # Return the final answer
                    final_answer = function_args.get("answer", "")
                    print(f"Final answer: {final_answer}")
                    
                    with open("scratchpad.txt", "w") as f:
                        f.write(str(model.messages))
                    
                    return final_answer
                
                # Add a follow-up prompt after tool execution
                if function_name in ["vlm_query", "caption_search"]:
                    followup_msg = {
                        "role": "user",
                        "content": followup_prompt(model.messages, question)
                    }
                    conversation_messages.append(followup_msg)
                    model.messages.append(followup_msg)
        
        else:
            # No tool calls, model just responded with text
            print(f"Model response: {assistant_message.content}")
    
    # If we've exhausted iterations, get a final answer
    final_prompt_msg = {
        "role": "user",
        "content": finish_prompt(model.messages)
    }
    conversation_messages.append(final_prompt_msg)
    
    final_response = model.llm_query_with_tools(
        messages=conversation_messages,
        tools=[tools[2]],  # Only provide final_answer tool
        tool_choice={"type": "function", "function": {"name": "final_answer"}}
    )
    
    if final_response and final_response.choices[0].message.tool_calls:
        tool_call = final_response.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        final_answer = function_args.get("answer", "Could not determine answer")
    else:
        final_answer = model.llm_query(finish_prompt(model.messages))
    
    with open("scratchpad.txt", "w") as f:
        f.write(str(model.messages))
    
    return final_answer


def query_model_simple(prompt, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo", **kwargs):
    """Simple wrapper to query any open-source model directly
    
    Args:
        prompt: The prompt to send
        model_name: The model to use. Options include:
            - "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            - "deepseek/deepseek-r1-distill-llama-70b"
            - "Qwen/Qwen2.5-72B-Instruct-Turbo"
            - "mistralai/Mixtral-8x7B-Instruct-v0.1"
            - "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
            - "cognitivecomputations/dolphin-2.5-mixtral-8x7b"
            - "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        **kwargs: Additional arguments (max_tokens, temperature)
    """
    return query_llm(model_name, prompt, **kwargs)


if __name__ == "__main__":
    model = Pipeline("Qwen/Qwen2.5-7B-Instruct-Turbo", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    question = "What is the protagonist primary objective at the spring?\n(A) Sleep\n(B) Drink some water\n(C) Wash clothes\n(D) Take a bath"
    answer = asyncio.run(query_model_iterative(model, question))
    print(answer)
    with open("messages.txt", "w") as f:
        f.write(str(model.messages))