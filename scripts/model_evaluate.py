import os
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment
import numpy as np

# Constants for reward function
PT_VEHICLE_TYPES = {'bus', 'tram_gdansk'}
PT_WAIT_CAP = 60.0        # maximum PT waiting time considered in penalty (seconds)
PT_WAIT_MULTIPLIER = 2.0  # multiplier for PT waiting time penalty (added not to spoil the function composition that sums up to 1)
PT_WAIT_NORM = 100.0      # normalization factor for total PT waiting time
WAIT_NORM = 100.0         # expected max waiting time change per step (seconds)

def baltycka_reward_fn(ts) -> float:
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
        0.10 * switch_penalty)

def run_evaluation():
    print("IInitializing evaluation environment...")

    route_files = (
        "../simulation/demand/netedit_cfg.rou.xml,"
        "../simulation/demand/car_ev.rou.xml,"
        "../simulation/demand/car.rou.xml,"
        "../simulation/demand/emergency.rou.xml,"
        "../simulation/demand/motorcycle.rou.xml,"
        "../simulation/demand/bus.rou.xml,"
        "../simulation/demand/tram.rou.xml,"
        "../simulation/demand/truck.rou.xml"
    )

    sumo_cmd = (
        "--collision.action remove "
        "--ignore-route-errors "
        "--additional-files ../simulation/osm.add.xml "
        "--emission-output ../simulation/results/evaluate_results/emissions.xml "
        "--tripinfo-output ../simulation/results/evaluate_results/tripinfos.xml "
        "--tripinfo-output.write-unfinished true "
        "--stop-output ../simulation/results/evaluate_results/stopinfos.xml "
        "--statistic-output ../simulation/results/evaluate_results/stats.xml "
        "--duration-log.statistics true"
    )

    env = SumoEnvironment(
        net_file='../simulation/network/osm-rl-agent.net.xml',
        route_file=route_files,
        out_csv_name='../models/evaluate_results/ppo_eval',
        additional_sumo_cmd=sumo_cmd,
        single_agent=True,
        ts_ids=['Glowny_wezel'],
        use_gui=True,
        num_seconds=3600,
        reward_fn=baltycka_reward_fn,
        time_to_teleport=500
    )

    model_path = "../models/best_model/best_model.zip" 
    
    if not os.path.exists(model_path):
        print(f"Error: Missing path {model_path}")
        return

    print(f"loading model from {model_path}...")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    step = 0

    print("Started simulation!")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print("Simluation finished! Saved results.")
    env.close()

if __name__ == "__main__":
    run_evaluation()