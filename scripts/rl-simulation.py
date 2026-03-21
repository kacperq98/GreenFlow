import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sumo_rl import SumoEnvironment, env

#sumo-rl documentation: https://lucasalegre.github.io/sumo-rl/

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Error: SUMO_HOME environment variable not set. Please set it to your SUMO installation directory.")

def environment_setup():
    route_files = (
            "../simulation/demand/car_ev.rou.xml,"
            "../simulation/demand/car.rou.xml,"
            "../simulation/demand/emergency.rou.xml,"
            "../simulation/demand/motorcycle.rou.xml,"
            "../simulation/demand/bus.rou.xml,"
            "../simulation/demand/tram.rou.xml,"
            "../simulation/demand/truck.rou.xml")

    env = SumoEnvironment(
        net_file='../simulation/network/osm.net.xml',  # Net file
        route_file=route_files,
        out_csv_name='../models/ppo_single_agent',
        additional_sumo_cmd="--collision.action remove --ignore-route-errors ",
        single_agent=True,
        ts_ids =['Glowny_wezel'],
        use_gui=False,
        num_seconds=3600,
        reward_fn='diff-waiting-time')
    return env

def create_model(env):
    print("Initializing PPO model...")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        ent_coef=0.05,
        gamma=0.99,
        n_steps=2048,
        batch_size=128,
        device='auto',
        policy_kwargs=dict(net_arch=[256, 128, 64]),
        tensorboard_log="../models/ppo_traffic_tensorboard/")
    
    return model

def model_learn(model, callback=None):
    model.learn(total_timesteps=100_000, callback=callback)
    print("Training finished!")


def model_save(model):
    model.save("../models/ppo_galeria_baltycka_v1")
    print("Model saved")

def close_environment(env):
    env.close()

def evaluate_model(eval_env):
    eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='../models/best_model/',
            log_path='../models/ppo_traffic_tensorboard/eval_logs/', 
            eval_freq=5_000, 
            deterministic=True, 
            render=False)
    return eval_callback, eval_env

def main():
    check_sumo_home()
    training_env = environment_setup()
    eval_env = environment_setup()
    model = create_model(training_env)

    eval_callback, eval_env = evaluate_model(eval_env)
    model_learn(model, callback=eval_callback)
    
    model_save(model)
    close_environment(training_env)
    close_environment(eval_env)

if __name__ == "__main__":
    main()