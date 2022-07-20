import nimblephysics as nimble
import numpy as np
from pathlib import Path
import pprint

from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
from typing import Any
from scipy.interpolate import interp1d

class MoCap:
    def __init__(self, train_path, val_path, env_frame_skip, env_step_size) -> None:
        self._hz = 100
        self._frame_skip = env_frame_skip
        self._step_size = env_step_size
        self._upper_limit = 21 - 1/100 - env_frame_skip*env_step_size # time lim of all mocap files, single frame removed for vel calc,

        self.train_traj = self.load_traj(train_path)
        self.val_traj = self.load_traj(val_path)

    def load_traj(self, path) -> Dict:
        trajectories = {}
        for np_name in Path(path).glob('*.np[yz]'):
            data = np.load(np_name)
            trajectories[np_name.stem] = interp1d(np.arange(0, data.shape[0]) / self._hz, data, axis=0)
        return trajectories

    def sample_expert(self, batch_size: int = 100) -> np.ndarray:
        rand_key = np.random.choice(self.get_keys('train'))
        curr_traj_func = self.train_traj[rand_key]

        curr_state_time = np.round(np.random.default_rng().uniform(low=0.0, high=self._upper_limit, size=batch_size), 2)
        next_state_time = curr_state_time + self._frame_skip * self._step_size

        curr_state = curr_traj_func(curr_state_time)
        next_state = curr_traj_func(next_state_time)

        return np.hstack((curr_state, next_state))

    def get_keys(self, dataset: str) -> List[str]:
        if dataset == 'train':
            return list(self.train_traj.keys())
        if dataset == 'val':
            return list(self.val_traj.keys())

if __name__ == "__main__":
    t_path = '/home/takaraet/gait_modeling/mocap/SubjectData_1/train_traj'
    v_path = '/home/takaraet/gait_modeling/mocap/SubjectData_1/val_traj'

    mocap = MoCap(t_path, v_path, 1, .01)

    train_keys = mocap.get_keys('train')

    print(mocap.sample_expert(10).shape)    # pprint.pprint(mocap.train_traj)
