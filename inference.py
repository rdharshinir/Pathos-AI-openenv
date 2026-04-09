import os
import json
import time
import signal
import sys

from openai import OpenAI
from client import PathosEnv, PathosAction


API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-url>")
MODEL_NAME   = os.getenv("MODEL_NAME",   "<your-active-model>")
HF_TOKEN     = os.getenv("HF_TOKEN",     "dummy")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Timeouts ─────────────────────────────────────────────────────────────────
# Hard limit for the entire inference.py run (seconds); leave headroom under 30 min.
WALL_CLOCK_LIMIT  = 25 * 60          # 25 minutes total
# Per-LLM-call timeout (seconds)
LLM_TIMEOUT       = 20               # give up on a single LLM call after 20 s
# Max environment steps per episode (safety net against infinite loops)
MAX_STEPS_PER_EPISODE = 50
# Connection check
CONNECT_MAX_RETRIES = 8
CONNECT_DELAY       = 1.0            # seconds between retries
# ─────────────────────────────────────────────────────────────────────────────

# Record the wall-clock start so we can bail out early if needed
_START_TIME = time.monotonic()

def _elapsed() -> float:
    return time.monotonic() - _START_TIME

def _time_remaining() -> float:
    return WALL_CLOCK_LIMIT - _elapsed()


# Initialize OpenAI client with a default per-request timeout
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
    timeout=LLM_TIMEOUT,   # applied to every request automatically
)

SYSTEM_PROMPT = """You are a Pathos Rescue Drone navigating a disaster zone.
You must reach the extraction zone while avoiding hazards and moving fires.
Your output must be EXCLUSIVELY a JSON object with two keys:
1. 'reasoning': A short explanation of your logic (max 2 sentences).
2. 'action': Your chosen action. MUST be one of: 'up', 'down', 'left', 'right'.
"""


def check_connection(url: str, max_retries: int = CONNECT_MAX_RETRIES,
                     delay: float = CONNECT_DELAY) -> bool:
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme in ("https", "wss") else 80)

    for attempt in range(max_retries):
        # Abort connection check if we're already running low on time
        if _time_remaining() < 60:
            return False
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, socket.timeout):
            if attempt < max_retries - 1:
                time.sleep(delay)
    return False


def _pick_safe_action(obs: dict) -> str:
    """Fallback: return the first valid non-hazard action, else first valid action."""
    paths = obs.get("valid_flight_paths", [])
    if not paths:
        return "up"
    safe = [p["label"] for p in paths if not p.get("is_hazard")]
    return safe[0] if safe else paths[0]["label"]


def _call_llm(prompt: str) -> str | None:
    """
    Call the LLM with a hard timeout. Returns the action string or None on failure.
    Respects the wall-clock budget.
    """
    if _time_remaining() < LLM_TIMEOUT + 5:
        # Not enough time left for a full LLM round-trip
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            timeout=LLM_TIMEOUT,
        )
        content = response.choices[0].message.content or ""
        # Strip markdown fences if the model wraps output
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        resp_json = json.loads(content.strip())
        return resp_json.get("action", "up")
    except Exception:
        return None


def play_episode() -> None:
    # Stdout logs follow the required structured format (START/STEP/END) exactly
    print("START", flush=True)

    base_url = os.environ.get("OPENENV_URL", "http://localhost:7860")

    if not check_connection(base_url):
        print("END", flush=True)
        sys.exit(0)

    try:
        with PathosEnv(base_url=base_url).sync() as env:
            try:
                result = env.reset()
            except Exception:
                print("END", flush=True)
                sys.exit(0)

            step_count = 0

            while not result.done:
                # ── Wall-clock guard ─────────────────────────────────
                if _time_remaining() < 30:
                    # Fewer than 30 s left globally — stop gracefully
                    break
                # ── Per-episode step cap ─────────────────────────────
                if step_count >= MAX_STEPS_PER_EPISODE:
                    break
                # ────────────────────────────────────────────────────

                print("STEP", flush=True)
                obs = result.observation.structured
                step_count += 1

                # Format the state for the LLM
                prompt = (
                    f"Current State:\n"
                    f"- Zone Size: {obs['zone_size']}x{obs['zone_size']}\n"
                    f"- Disaster Type: {obs['disaster_type']}\n"
                    f"- Drone Position: {obs['drone_position']}\n"
                    f"- Extraction Zone: {obs['extraction_zone']}\n"
                    f"- Distance to extraction: {obs['manhattan_dist_to_extraction']}\n"
                    f"- Direction to extraction: {obs['direction_to_extraction']}\n"
                    f"\nValid Flight Paths (those not immediately resulting in a crash):\n"
                )
                for a in obs.get("valid_flight_paths", []):
                    prompt += f" - {a['label']} -> flies to {a['leads_to']}\n"

                # Try LLM; fall back to heuristic if it times out or errors
                action_str = _call_llm(prompt)
                if action_str is None:
                    action_str = _pick_safe_action(obs)

                try:
                    result = env.step(PathosAction(message=action_str))
                except Exception:
                    # Step call failed — exit cleanly
                    break

    except Exception:
        pass

    print("END", flush=True)


if __name__ == "__main__":
    play_episode()
