#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

# reuse your function from gstar_astar_proto.py
from gstar_astar_proto import choose_action_with_gstar

def _extract_goal(info: dict, obs_text: str) -> str:
    for k in ("goal", "goal_desc", "task_desc", "mission"):
        v = info.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def run_episode(max_steps: int = 12):
    load_dotenv(override=True)

    # Create TextWorld env (English)
    cfg = generic.load_config()
    env = get_environment("AlfredTWEnv")(cfg, train_eval="eval").init_env(batch_size=1)

    obs_list, info = env.reset()
    obs = obs_list[0]
    goal = _extract_goal(info, obs)

    done, step = False, 0
    while not done and step < max_steps:
        admiss = info.get("admissible_commands", set())
        if isinstance(admiss, dict):
            admiss = set(admiss.get(0, [])) or set(admiss.get("0", [])) or set()
        admiss = sorted(set(admiss)) if admiss else ["look", "inventory"]

        action = choose_action_with_gstar(
            obs=obs, admissible=admiss, goal=goal,
            k=8, tau=1.05, budget_nodes=3, edge_alpha=0.0
        )

        obs_list, scores, dones, info = env.step([action])
        obs = obs_list[0]
        done = bool(dones[0]); step += 1
        print(f"[{step}] action={action!r} | score={scores[0]} | done={done}")

if __name__ == "__main__":
    run_episode()
