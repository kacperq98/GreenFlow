"""
Train a PPO agent on the Baltycka intersection environment.

Usage:
    python training/train_ppo.py
    python training/train_ppo.py --timesteps 500_000 --model-name ppo_baltycka_v2
    python training/train_ppo.py --gui
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from envs import BaltyckaIntersectionEnv
from envs.baltycka_intersection_env import DELTA_TIME, EPISODE_STEPS

MODELS_DIR = ROOT / 'models'
LOGS_DIR = ROOT / 'logs'


def train(model_name: str, total_timesteps: int):
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    print(f"Training PPO for {total_timesteps:_} timesteps...")
    print(f"Episode length: {EPISODE_STEPS} steps x {DELTA_TIME}s = {EPISODE_STEPS * DELTA_TIME}s simulated\n")

    env = BaltyckaIntersectionEnv(use_gui=False)
    check_env(env, warn=True)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        ent_coef=0.05,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(MODELS_DIR / model_name)

    env.close()
    print(f'Training complete. Model saved to models/{model_name}.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='ppo_baltycka',
        help='Name of the saved model file (default: ppo_baltycka)',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=200_000,
        help='Total training timesteps (default: 200_000)',
    )
    args = parser.parse_args()

    train(model_name=args.model_name, total_timesteps=args.timesteps)
