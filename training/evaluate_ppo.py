"""
Evaluate a trained PPO agent against the fixed-time baseline.

Usage:
    python training/evaluate_ppo.py --model-name ppo_baltycka --episodes 3
    python training/evaluate_ppo.py --baseline
    python training/evaluate_ppo.py --model-name ppo_baltycka --episodes 3 --gui
    python training/evaluate_ppo.py --model-name ppo_baltycka --episodes 5 --seed 42
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from envs import BaltyckaIntersectionEnv
from envs.baltycka_intersection_env import EPISODE_STEPS, DELTA_TIME

MODELS_DIR = ROOT / 'models'


def _run_episode(env: BaltyckaIntersectionEnv, action_fn) -> tuple[float, float]:
    obs, _ = env.reset()
    total_reward = 0.0
    total_halting = 0.0
    steps = 0
    done = False

    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_halting += info['total_halting']
        steps += 1
        done = terminated or truncated

    avg_halting_per_step = total_halting / steps
    return total_reward, avg_halting_per_step


def evaluate_agent(model_name: str, episodes: int, use_gui: bool, seed: int = None) -> float:
    model_path = MODELS_DIR / f'{model_name}.zip'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    env = BaltyckaIntersectionEnv(use_gui=use_gui, seed=seed)
    model = PPO.load(model_path)
    model.verbose = 0  # get rid of wrapper info logs
    model.set_env(env)

    def action_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    rewards, haltings = [], []
    for episode in range(episodes):
        reward, avg_halting = _run_episode(env, action_fn)
        rewards.append(reward)
        haltings.append(avg_halting)
        print(f"[RL agent]  episode {episode + 1:>2}: reward={reward:>8.1f}  avg_halting={avg_halting:.2f}")

    env.close()

    mean, std = np.mean(rewards), np.std(rewards)
    print(f"[RL agent]  mean reward: {mean:.1f} ± {std:.1f}")
    return float(mean)


def evaluate_baseline(episodes: int, cycle_steps: int = 12, seed: int = None) -> float:
    env = BaltyckaIntersectionEnv(use_gui=False, seed=seed)

    def action_fn(obs):
        return 1 if env._step_count % cycle_steps == 0 and env._step_count > 0 else 0

    rewards, haltings = [], []
    for episode in range(episodes):
        reward, avg_halting = _run_episode(env, action_fn)
        rewards.append(reward)
        haltings.append(avg_halting)
        print(f"[baseline]  episode {episode + 1:>2}: reward={reward:>8.1f}  avg_halting={avg_halting:.2f}")

    env.close()

    mean, std = np.mean(rewards), np.std(rewards)
    print(f"[baseline]  mean reward: {mean:.1f} ± {std:.1f}\n")
    return float(mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO agent vs fixed-time baseline')
    parser.add_argument('--model-name', type=str, default='ppo_baltycka')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    parser.add_argument('--baseline', action='store_true', help='Run baseline only (no model needed)')
    parser.add_argument('--seed', type=int, default=None, help='Fixed seed for reproducibility (disabled by default)')
    args = parser.parse_args()

    print(f"\nEpisode length: {EPISODE_STEPS} steps x {DELTA_TIME}s = {EPISODE_STEPS * DELTA_TIME}s simulated\n")

    if args.baseline:
        evaluate_baseline(episodes=args.episodes, seed=args.seed)
    else:
        baseline_mean = evaluate_baseline(episodes=args.episodes, seed=args.seed)
        agent_mean = evaluate_agent(model_name=args.model_name, episodes=args.episodes, use_gui=args.gui, seed=args.seed)
        if baseline_mean != 0:
            improvement = (agent_mean - baseline_mean) / abs(baseline_mean) * 100
            print(f"\nImprovement over baseline: {improvement:+.1f}%\n")
