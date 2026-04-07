---
title: Pathos AI – Autonomous Drone Rescue Simulator
emoji: 🚁
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
license: mit
short_description: Grid-world RL environment for autonomous drone rescue with curriculum learning
---

# 🚁 Pathos AI — Autonomous Drone Rescue Simulator

**A high-utility, evaluator-ready Reinforcement Learning environment for training and benchmarking autonomous agents in disaster-response scenarios.**

[![Running](https://img.shields.io/badge/status-running-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/framework-OpenEnv-blue)]()
[![Curriculum](https://img.shields.io/badge/curriculum-4%20levels-orange)]()

---

## 🎯 Real-World Utility

Pathos AI simulates **disaster-zone drone navigation** — the exact kind of autonomy needed in real search-and-rescue operations. An autonomous agent must:

- 🚁 Navigate hazardous terrain without GPS (grid-world)
- 🆘 Rescue survivors before reaching the extraction zone
- ☣️ Avoid static hazards and 🔥 dynamically spreading fires
- 🌪️ Handle wind zones that push the drone off course
- 🌫️ Operate under fog-of-war with limited visibility

This directly mirrors real autonomous UAV challenges in wildfire response, earthquake rescue, and military operations.

---

## 🎮 Visual Dashboard (NEW)

Access the full visual interface at: **`/ui`**

| Feature | Description |
|---|---|
| 🗺️ **Live SVG Grid** | Color-coded, animated drone movement with hover coordinates |
| 📊 **Reward Curve** | Real-time Chart.js line chart over last 50 episodes |
| 🔥 **Visit Heatmap** | Canvas-based frequency heatmap of drone positions |
| 🎯 **Objective Tracker** | Live status of all 3 tiered objectives |
| 🔄 **Replay System** | Step-by-step playback of current/best/worst episodes |
| 🗒️ **Scenario Editor** | Place hazards, save/load JSON map seeds |
| 🏆 **Leaderboard** | Submit agent scores, top-10 ranking table |

**Controls:** WASD / Arrow keys, or type natural language ("go north")

---

## 📐 Environment Design

### Curriculum Learning — 4 Levels

| Level | Label | Grid | Map Type | Features |
|---|---|---|---|---|
| 1 | 🟢 Rookie | 5×5 | Open | Fixed layout, no hazards |
| 2 | 🔵 Trained | 7×7 | Sparse | 3 static hazards, random start |
| 3 | 🟠 Expert | 10×10 | Maze | Moving hazards + fog of war |
| 4 | 🔴 Legendary | 10×10 | Adversarial | Wind zones + multi-hazard + fog |

### Tiered Objectives

| Objective | Reward |
|---|---|
| 🏥 Reach Extraction Zone | **+10.0** |
| 🆘 Rescue each Survivor | **+0.3** |
| ⚡ Speed Bonus (≤10 steps) | **+0.5** |
| 🪫 Step Penalty | **-0.1** per step |
| ☣️ Hit Hazard | **-10.0** (fatal) |

### Dynamic Mechanics
- **🔥 Moving Fires** – hazards that shift randomly each step (adversarial map)
- **🌪️ Wind Zones** – cells that push the drone one extra random step (Legendary only)
- **🌫️ Fog of War** – only cells within radius-2 are visible (Expert & Legendary)
- **🧱 Maze Walls** – procedurally generated via recursive backtracking (Expert)

---

## 🔌 API Endpoints

### Standard OpenEnv
| Endpoint | Description |
|---|---|
| `POST /reset` | Reset episode (pass `difficulty: 1-4`) |
| `POST /step` | Step with natural language action |
| `GET /state` | Current environment state |

### Extended Hackathon API
| Endpoint | Description |
|---|---|
| `GET /ui` | Visual dashboard |
| `GET /grid_ui/{id}` | SVG cell data for renderer |
| `GET /heatmap/{id}` | Visit frequency heatmap |
| `GET /replay/{id}` | Trajectory replay data |
| `GET /export_layout/{id}` | Export map as JSON seed |
| `POST /load_layout/{id}` | Load scenario from JSON seed |
| `GET /leaderboard` | Top-10 agent leaderboard |
| `POST /submit_score` | Submit agent score |
| `GET /episode_stats` | Reward curve data (last 50 ep) |

---

## 🐍 Python Client

```python
from gridmind import MyEnv, MyAction

with MyEnv.from_env("DHARSHINI-457/my_env") as env:
    result = await env.reset(difficulty=2)  # Level 2: Trained

    while not result.done:
        obs = result.structured
        direction = obs["direction_to_extraction"]
        result = await env.step(MyAction(message=f"go {direction}"))

    print(f"Score: {result.objectives['total_score']:.2f}")
    print(f"Survivors: {result.objectives['survivors_rescued']}")
```

---

## 📊 Grading Rubric

The `grader.py` provides deterministic scoring:

```
Success:  score = 0.5 + 0.5 × efficiency + speed_bonus + survivor_bonus
Failure:  score = 0.4 × closeness_to_goal + survivor_bonus
```

Tasks are exported as `grader.tasks` with all 4 curriculum levels.

---

## 🏗️ Architecture

```
my_env/
├── env.py                    # Core GridEnv: wind, fog, maze, objectives, replay
├── models.py                 # Pydantic observation/action models  
├── grader.py                 # Deterministic scoring + 4 task definitions
├── server/
│   ├── app.py               # FastAPI: OpenEnv + Hackathon endpoints
│   ├── my_env_environment.py # OpenEnv session wrapper
│   └── static/
│       └── index.html       # Full visual dashboard (SVG, Chart.js, Canvas)
```

---

*Built for the Meta OpenEnv Hackathon 2026 — Pathos AI demonstrates real-world utility, creative design, and comprehensive LLM-agent evaluation capabilities.*
