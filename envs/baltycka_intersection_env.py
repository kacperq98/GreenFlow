import gymnasium as gym
import numpy as np
import os
import traci

# Path to simulation config
SUMOCFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'simulation', 'osm.sumocfg')

# The main traffic light on the Baltycka intersection -> controlled by the RL agent
TL_ID = 'Glowny_wezel'

# Lanes linked to the intersection (from detectors/osm.add.xml)
DETECTOR_LANES = [
    '161297931#0_0',
    '161297931#0_1',
    '182045481#4.51_0',
    '187938710#0_0',
    '187938710#0_1',
    '187938710#0_2',
    '187938710#0_3',
    '236811148_0',
    '236811148_1',
    '236811148_2',
    '236811148_3',
    '236811148_4',
    '236811148_5',
]

NUM_GREEN_PHASES = 7               # number of green phases in the intersection
YELLOW_TIME = 3                    # seconds spent in yellow phase during phase transition
MIN_GREEN_TIME = 10                # minimum seconds before a phase switch is allowed (to avoid phase flickering)
DELTA_TIME = 5                     # simulation seconds per agent step
EPISODE_STEPS = 720                # max steps per episode (720 × 5s = 3600s = 1 simulated hour)
MAX_HALTING = 50                   # normalization cap for halting vehicles per lane
MAX_PHASE_DURATION = 90            # normalization cap for elapsed phase time
PHASE_LOCK_DURATION = 1_000_000    # large value to prevent SUMO from advancing phase automatically


class BaltyckaIntersectionEnv(gym.Env):
    """
    RL environment for the Galeria Bałtycka intersection in Gdańsk.

    Observations (21):
        [0:13]  halting vehicles per detector lane, normalized to [0, 1]
        [13:20] one-hot encoding of current green phase (for NUM_GREEN_PHASES = 7)
        [21]    elapsed time in current phase, normalized to [0, 1]

    Actions (2):
        0 — keep current green phase
        1 — change phase (ignored if elapsed time < MIN_GREEN_TIME)

    Reward:
        negative sum of halting vehicles across all detector lanes
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, use_gui: bool = False, seed: int = 42):
        super().__init__()

        self.use_gui = use_gui
        self._seed = seed
        self._sumo_binary = 'sumo-gui' if use_gui else 'sumo'

        self._green_phases: list[int] = []
        self._yellow_phases: dict[int, int] = {}
        self._num_green_phases: int = 0
        self._current_green_idx: int = 0
        self._green_timer: int = 0
        self._step_count: int = 0

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(DETECTOR_LANES) + NUM_GREEN_PHASES + 1,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        try:
            traci.close()
        except Exception:
            pass

        self._start_sumo()
        traci.simulationStep()

        self._current_green_idx = 0
        self._green_timer = 0
        self._step_count = 0
        self._set_phase(self._green_phases[0])  # start with the first green phase

        return self._get_observations(), {}  # return (obs, info)

    def step(self, action: int):
        switched = self._apply_action(action)
        yellow_used = YELLOW_TIME if switched else 0

        for _ in range(DELTA_TIME - yellow_used):
            traci.simulationStep()

        self._green_timer += DELTA_TIME - yellow_used
        self._step_count += 1

        obs = self._get_observations()
        reward = self._get_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self._step_count >= EPISODE_STEPS
        info = {
            'sim_time': traci.simulation.getTime(),
            'total_halting': -reward,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _start_sumo(self):
        sumo_cmd = [
            self._sumo_binary,
            '-c', os.path.abspath(SUMOCFG_PATH),
            '--no-warnings',
            '--no-step-log',
            '--seed', str(self._seed),
        ]
        if self.use_gui:
            sumo_cmd += ['--start', '--delay', '100']
        traci.start(sumo_cmd)

        phases = traci.trafficlight.getAllProgramLogics(TL_ID)[0].phases
        self._green_phases = [
            i for i, p in enumerate(phases)
            if 'G' in p.state or 'g' in p.state
        ]
        self._num_green_phases = len(self._green_phases)
        assert self._num_green_phases == NUM_GREEN_PHASES, (
            f'Expected {NUM_GREEN_PHASES} green phases, got {self._num_green_phases}. '
            f'Update NUM_GREEN_PHASES constant.'
        )
        self._yellow_phases = self._detect_yellow_phases(phases)

    def _set_phase(self, phase_idx: int):
        traci.trafficlight.setPhase(TL_ID, phase_idx)
        traci.trafficlight.setPhaseDuration(TL_ID, PHASE_LOCK_DURATION)

    def _detect_yellow_phases(self, phases: list) -> dict[int, int]:
        """Map each green phase index to its following yellow phase index."""
        n = len(phases)
        mapping = {}
        for g_idx in self._green_phases:
            y_idx = (g_idx + 1) % n
            mapping[g_idx] = y_idx if 'y' in phases[y_idx].state else g_idx
        return mapping

    def _apply_action(self, action: int) -> bool:
        if action == 1 and self._green_timer >= MIN_GREEN_TIME:
            yellow_idx = self._yellow_phases[self._green_phases[self._current_green_idx]]
            traci.trafficlight.setPhase(TL_ID, yellow_idx)
            for _ in range(YELLOW_TIME):
                traci.simulationStep()

            self._current_green_idx = (self._current_green_idx + 1) % self._num_green_phases
            self._set_phase(self._green_phases[self._current_green_idx])
            self._green_timer = 0
            return True
        return False

    def _get_observations(self) -> np.ndarray:
        halting = np.array(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in DETECTOR_LANES],
            dtype=np.float32,
        )
        halting_norm = np.clip(halting / MAX_HALTING, 0.0, 1.0)
        phase_one_hot = np.zeros(NUM_GREEN_PHASES, dtype=np.float32)
        phase_one_hot[self._current_green_idx] = 1.0
        elapsed_norm = np.array([min(self._green_timer / MAX_PHASE_DURATION, 1.0)], dtype=np.float32)
        return np.concatenate([halting_norm, phase_one_hot, elapsed_norm])

    def _get_reward(self) -> float:
        return -float(sum(
            traci.lane.getLastStepHaltingNumber(lane) for lane in DETECTOR_LANES
        ))
