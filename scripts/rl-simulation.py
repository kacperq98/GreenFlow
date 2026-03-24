import argparse
import logging
import json
import numpy as np
from utils import setup_sumo_home
setup_sumo_home()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from sumo_rl import SumoEnvironment, env

# Constants for reward function
PT_VEHICLE_TYPES = {'bus', 'tram_gdansk'}
PT_WAIT_CAP = 60.0        # maximum PT waiting time considered in penalty (seconds)
PT_WAIT_MULTIPLIER = 2.0  # multiplier for PT waiting time penalty (added not to spoil the function composition that sums up to 1)
PT_WAIT_NORM = 100.0      # normalization factor for total PT waiting time
WAIT_NORM = 100.0         # expected max waiting time change per step (seconds)


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("rl-simulation.log"),
                        logging.StreamHandler()
                    ])

# sumo-rl documentation: https://lucasalegre.github.io/sumo-rl/

def baltycka_reward_fn(ts):
    # 1. Waiting time difference between current and previous step
    current_wait = sum(ts.get_accumulated_waiting_time_per_lane())
    last_wait = getattr(ts, '_last_wait', current_wait)
    waiting_time_delta = np.clip((last_wait - current_wait) / WAIT_NORM, -1, 1)  # clipped so it won't dominate the rest
    ts._last_wait = current_wait

    # 2. Quadratic queue penalty (normalized by number of lanes)
    queues = ts.get_lanes_queue()
    queue_penalty = -sum(q ** 2 for q in queues) / len(queues)

    # 3. Average vehicle speed (already normalized by sumo-rl)
    avg_speed = ts.get_average_speed()

    # 4. Public transport priority (penalty for waiting time of buses/trams - averaged, capped and normalized)
    pt_waits = [
        min(ts.sumo.vehicle.getAccumulatedWaitingTime(veh_id), PT_WAIT_CAP)
        for veh_id in ts._get_veh_list()
        if ts.sumo.vehicle.getTypeID(veh_id) in PT_VEHICLE_TYPES
    ]
    pt_penalty = -np.mean(pt_waits) * PT_WAIT_MULTIPLIER / PT_WAIT_NORM if pt_waits else 0.0

    # 5. Phase switching penalty (punishes hard rapid phase changes, gentle to rare switches)
    switch_penalty = 0.0
    now = ts.env.sim_step
    if getattr(ts, '_last_phase', None) != ts.green_phase:
        dt = now - getattr(ts, '_last_switch_time', now)
        switch_penalty = -1.0 / (dt + 1)
        ts._last_switch_time = now
    ts._last_phase = ts.green_phase

    return (
        0.30 * waiting_time_delta +
        0.25 * queue_penalty +
        0.20 * avg_speed +
        0.15 * pt_penalty +
        0.10 * switch_penalty
    )

def environment_setup(use_gui=False):
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
        net_file='../simulation/network/osm-rl-agent.net.xml',  # Net file
        route_file=route_files,
        out_csv_name='../models/ppo_single_agent',
        additional_sumo_cmd="--collision.action remove --ignore-route-errors ",
        single_agent=True,
        ts_ids =['Glowny_wezel'],
        use_gui=use_gui,
        num_seconds=3600,
        reward_fn=baltycka_reward_fn)
    logging.info("SUMO environment created.")
    return env

def load_best_hyperparameters(params):
    try:
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
            logging.info(f"Loaded hyperparameters from best_params.json: {best_params}")

            params['learning_rate'] = best_params.get('learning_rate', params['learning_rate'])
            params['n_steps'] = best_params.get('n_steps', params['n_steps'])
            params['gamma'] = best_params.get('gamma', params['gamma'])
            params['ent_coef'] = best_params.get('ent_coef', params['ent_coef'])
            params['batch_size'] = best_params.get('batch_size', params['batch_size'])

            net_arch_dict = {
                'tiny': [64, 64],
                'small': [256, 128],
                'medium': [256, 128, 64]
            }
            net_arch_str = best_params.get('net_arch', 'medium')
            params['net_arch'] = net_arch_dict.get(net_arch_str, params['net_arch'])

    except FileNotFoundError:
        logging.info("best_params.json not found. Using default hyperparameters.")

    return params

def create_model(env):
    logging.info("Initializing PPO model...")

    default_params = {
        'learning_rate': 0.001,
        'n_steps': 2048,
        'gamma': 0.99,
        'ent_coef': 0.05,
        'batch_size': 128,
        'net_arch': [256, 128, 64]}

    params = load_best_hyperparameters(default_params)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=params['learning_rate'],
        ent_coef=params['ent_coef'],
        gamma=params['gamma'],
        n_steps=params['n_steps'],
        batch_size=params['batch_size'],
        device='auto',
        policy_kwargs=dict(net_arch=params['net_arch']),
        tensorboard_log="../models/ppo_traffic_tensorboard/")

    logging.info("PPO model initialized.")
    return model

def model_learn(model, callback=None, total_timesteps=200_000):
    logging.info("Starting model training.")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    logging.info("Training finished!")

def model_save(model, model_name):
    model.save(f"../models/{model_name}")
    logging.info(f"Model saved to ../models/{model_name}")

def close_environment(env):
    env.close()
    logging.info("Environment closed.")

def evaluate_model(eval_env):
    logging.info("Setting up model evaluation.")
    eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='../models/best_model/',
            log_path='../models/ppo_traffic_tensorboard/eval_logs/', 
            eval_freq=10_000,
            deterministic=True, 
            render=False)
    logging.info("Evaluation callback created.")
    return eval_callback, eval_env

def main(model_name, total_timesteps, use_gui):
    logging.info("--- Starting RL Simulation ---")
    training_env = environment_setup(use_gui=use_gui)
    check_env(training_env, warn=True)
    eval_env = environment_setup()
    model = create_model(training_env)

    eval_callback, eval_env = evaluate_model(eval_env)
    model_learn(model, callback=eval_callback, total_timesteps=total_timesteps)

    model_save(model, model_name)
    close_environment(training_env)
    close_environment(eval_env)
    logging.info("--- RL Simulation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on Baltycka intersection')
    parser.add_argument('--model-name', type=str, default='ppo_galeria_baltycka_v1',
                        help='Name of the saved model file (default: ppo_galeria_baltycka_v1)')
    parser.add_argument('--timesteps', type=int, default=200_000,
                        help='Total training timesteps (default: 200_000)')
    parser.add_argument('--gui', action='store_true', help='Run simulation with SUMO GUI')
    args = parser.parse_args()

    main(model_name=args.model_name, total_timesteps=args.timesteps, use_gui=args.gui)
