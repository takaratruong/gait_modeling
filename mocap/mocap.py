import numpy as np
from pathlib import Path
import pprint

from typing import List
from typing import Dict
from typing import Tuple
from scipy.interpolate import interp1d

class MoCap:
    def __init__(self, train_path, val_path, env_frame_skip, env_step_size) -> None:
        self._hz = 100
        self._frame_skip = env_frame_skip
        self._step_size = env_step_size
        self._upper_limit = 21 - 1/100 - env_frame_skip*env_step_size -1# time lim of all mocap files (experimental detail), single frame removed for vel calc, and a couple more in case

        self.train_traj = self.load_traj(train_path)
        self.val_traj = self.load_traj(val_path)

    def load_traj(self, path) -> Dict:
        trajectories = {}
        for np_name in Path(path).glob('*.np[yz]'):
            data = np.load(np_name)
            trajectories[np_name.stem] = interp1d(np.arange(0, data.shape[0]) / self._hz, data, axis=0)
        return trajectories

    def sample_expert(self, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        rand_key = np.random.choice(self.get_keys('train'))
        curr_traj_func = self.train_traj[rand_key]

        curr_state_time = np.round(np.random.default_rng().uniform(low=0.0, high=self._upper_limit, size=batch_size), 2)
        next_state_time = curr_state_time + self._frame_skip * self._step_size

        curr_state = curr_traj_func(curr_state_time)
        next_state = curr_traj_func(next_state_time)

        return curr_state, next_state

    def get_keys(self, dataset: str) -> List[str]:
        if dataset == 'train':
            return list(self.train_traj.keys())
        if dataset == 'val':
            return list(self.val_traj.keys())

if __name__ == "__main__":
    t_path = '/home/takaraet/gait_modeling/mocap/SubjectData_1/train_traj'
    v_path = '/home/takaraet/gait_modeling/mocap/SubjectData_1/val_traj'

    mocap = MoCap(t_path, v_path, 8, .01)

    train_keys = mocap.get_keys('train')

    print(mocap.train_traj[train_keys[0]](12))


    print(train_keys)
    batch = mocap.sample_expert(100000)
    # print(batch)
