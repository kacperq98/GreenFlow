import argparse
import logging
import os
import numpy as np
from utils import setup_sumo_home
setup_sumo_home()

from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("evaluate.log"),
                        logging.StreamHandler()
                    ])


def environment_setup(use_gui=False, seed=None):
    logging.info("Setting up SUMO environment.")
    route_files = (
            "../simulation/demand/car_ev.rou.xml,"
            "../simulation/demand/car.rou.xml,"
            "../simulation/demand/emergency.rou.xml,"
            "../simulation/demand/motorcycle.rou.xml,"
            "../simulation/demand/bus.rou.xml,"
            "../simulation/demand/tram.rou.xml,"
            "../simulation/demand/truck.rou.xml")

    env = SumoEnvironment(
        net_file='../simulation/network/osm-rl-agent.net.xml',
        route_file=route_files,
        out_csv_name='../models/eval_single_agent',
        additional_sumo_cmd='--collision.action remove --ignore-route-errors',
        single_agent=True,
        ts_ids=['Glowny_wezel'],
        use_gui=use_gui,
        num_seconds=3600,
        sumo_seed=seed if seed is not None else 'random',
    )
    logging.info("SUMO environment created.")
    return env


def run_episode(env, action_fn):
    obs, _ = env.reset()
    total_reward = 0.0
    total_halting = 0
    steps = 0

    while True:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_halting += env.traffic_signals['Glowny_wezel'].get_total_queued()
        steps += 1
        if terminated or truncated:
            break

    return total_reward, total_halting / steps if steps > 0 else 0.0


def evaluate_agent(model_name, episodes, use_gui, seed=None):
    model_path = f'../models/{model_name}.zip'
    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logging.info(f"Evaluating agent: {model_name}")
    model = PPO.load(model_path)

    def action_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    rewards, haltings = [], []
    for episode in range(episodes):
        env = environment_setup(use_gui=use_gui, seed=seed)
        reward, avg_halting = run_episode(env, action_fn)
        env.close()
        rewards.append(reward)
        haltings.append(avg_halting)
        logging.info(f"[RL agent]  episode {episode + 1:>2}: reward={reward:>8.1f}  avg_halting={avg_halting:.2f}")

    mean, std = np.mean(rewards), np.std(rewards)
    logging.info(f"[RL agent]  mean reward: {mean:.1f} ± {std:.1f}")
    return float(mean)


def evaluate_baseline(episodes, cycle_seconds=30, seed=None):
    logging.info("Evaluating fixed-time baseline.")

    rewards, haltings = [], []
    for episode in range(episodes):
        env = environment_setup(use_gui=False, seed=seed)
        step_count = 0

        def action_fn(obs):
            nonlocal step_count
            action = 1 if step_count > 0 and step_count % cycle_seconds == 0 else 0
            step_count += 1
            return action

        reward, avg_halting = run_episode(env, action_fn)
        env.close()
        rewards.append(reward)
        haltings.append(avg_halting)
        logging.info(f"[baseline]  episode {episode + 1:>2}: reward={reward:>8.1f}  avg_halting={avg_halting:.2f}")

    mean, std = np.mean(rewards), np.std(rewards)
    logging.info(f"[baseline]  mean reward: {mean:.1f} +/- {std:.1f}")
    return float(mean)


def main(model_name, episodes, use_gui, baseline_only, seed=None):
    logging.info("--- Starting Evaluation ---")

    if baseline_only:
        evaluate_baseline(episodes=episodes, seed=seed)
    else:
        baseline_mean = evaluate_baseline(episodes=episodes, seed=seed)
        agent_mean = evaluate_agent(model_name=model_name, episodes=episodes, use_gui=use_gui, seed=seed)
        if baseline_mean != 0:
            improvement = (agent_mean - baseline_mean) / abs(baseline_mean) * 100
            logging.info(f"Improvement over baseline: {improvement:+.1f}%")

    logging.info("--- Evaluation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate PPO agent vs fixed-time baseline')
    parser.add_argument('--model-name', type=str, default='ppo_galeria_baltycka_v1',
                        help='Name of the saved model file (default: ppo_galeria_baltycka_v1)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of evaluation episodes (default: 3)')
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode')
    parser.add_argument('--baseline', action='store_true', help='Run baseline only (no model needed)')
    parser.add_argument('--seed', type=int, default=None, help='Fixed seed for reproducibility')
    args = parser.parse_args()

    main(model_name=args.model_name, episodes=args.episodes, use_gui=args.gui,
         baseline_only=args.baseline, seed=args.seed)
