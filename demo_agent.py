"""
Demo Llama 3/4 Agent for the GridMind Benchmark.

This lightweight script connects to the GridMind OpenEnv server, gets the structured
observation, formats it as a prompt, and queries standard Chat Completions API
to get the next move.
"""
import os
import json
import time

# Deferred openai import for when a real API is used

from client import MyEnv, MyAction


# Fallback test mock API client if api key is not set
class MockLlamaClient:
    def __init__(self, *args, **kwargs):
        pass
    
    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                class Msg:
                    @property
                    def content(self):
                        return json.dumps({"reasoning": "Mock reasoning", "action": "south"})
                
                class Choice:
                    def __init__(self):
                        self.message = Msg()

                class Resp:
                    def __init__(self):
                        self.choices = [Choice()]
                return Resp()

API_KEY = os.environ.get("LLAMA_API_KEY", "dummy-key")
BASE_URL = os.environ.get("LLAMA_BASE_URL", "https://api.llama.com/v1")

# Use real OpenAI-compatible client if a key is provided
if API_KEY != "dummy-key":
    try:
        from openai import OpenAI
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    except ImportError:
        print("Please pip install openai to use real models.")
        exit(1)
else:
    print("WARNING: Using mock logic. Set LLAMA_API_KEY environment variable to use real Llama models.")
    client = MockLlamaClient()

SYSTEM_PROMPT = """You are a smart agent navigating a grid-world.
You must reach the goal while avoiding traps and walls.
Your output must be EXCLUSIVELY a JSON object with two keys:
1. 'reasoning': A short explanation of your logic (max 2 sentences).
2. 'action': Your chosen action. MUST be one of: 'up', 'down', 'left', 'right'.
"""

def play_episode():
    # Connect to local environment server (make sure `uvicorn server.app:app` is running)
    with MyEnv(base_url="http://localhost:8000").sync() as env:
        print("Starting Llama Agent Demo...")
        # Reset and get initial structured observation
        result = env.reset()
        
        while not result.done:
            obs = result.observation.structured
            
            # Format the state cleanly for the LLM
            prompt = f"""
Current State:
- Grid Size: {obs['grid_size']}x{obs['grid_size']}
- Map Type: {obs['map_type']}
- Agent Position: {obs['agent']}
- Goal Position: {obs['goal']}
- Distance to goal: {obs['manhattan_dist_to_goal']}
- Direction to goal: {obs['direction_to_goal']}

Valid Actions (those not immediately resulting in a wall bump):
"""
            for a in obs['valid_actions']:
                 prompt += f" - {a['label']} -> moves to {a['leads_to']}\n"
                 
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-70b-instruct",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                )
                
                # Parse JSON output from the model
                content = response.choices[0].message.content
                # Strip markdown codeblocks if model wraps output
                if content.startswith("```json"):
                    content = content[7:-3]
                    
                resp_json = json.loads(content)
                reasoning = resp_json.get("reasoning", "")
                action_str = resp_json.get("action", "up")
                
                print(f"🧐 CoT: {reasoning}")
                print(f"🤖 Action: {action_str} -> Map Update:")
                
                result = env.step(MyAction(message=action_str))
                print(result.observation.grid_state)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Agent error or JSON parse failure: {e}")
                # Fallback to safe valid action
                safe_act = "up"
                if obs['valid_actions']:
                    safe_act = obs['valid_actions'][0]['label']
                result = env.step(MyAction(message=safe_act))
        
        print("\n=== Episode Finished ===")
        print(f"Reward: {result.reward}")
        if result.reward >= 10.0:
            print("🎉 Success! Goal reached!")
        else:
            print("💀 Failure! Agent hit a trap.")
            
if __name__ == "__main__":
    play_episode()
