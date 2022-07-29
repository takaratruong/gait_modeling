# import nimblephysics as nimble
import pprint
import ipdb
import time
import numpy as np
import glob

from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scipy.interpolate import interp1d

import sys
sys.path.append('..')
from environments.skeleton.skeleton_env import SkeletonEnv
from utils.config_loader import load_args


if __name__ == "__main__":
    args = load_args()

    name = 'S02DN101'

    # df = pd.read_table("S02DN101_v5.mot", skiprows=10)
    df = pd.read_table("S01DN201_ik.mot", skiprows=10)

    df['ground_pelvis_trans_x'] -= df['ground_pelvis_trans_x'].mean()
    df['ground_pelvis_trans_z'] -= df['ground_pelvis_trans_z'].mean()

    env = SkeletonEnv(args=args)
    jnt2adr = env.model.get_joint_qpos_addr

    mot = np.zeros(36)
    for index, row in df.iterrows():
        qpos = np.zeros(36)
        qvel = np.zeros(36)

        qpos[jnt2adr('treadmill')] = 1.25
        for joint in env.model.joint_names:
            if joint != 'treadmill':
                addr = env.model.get_joint_qpos_addr(joint)
                qpos[addr] = row[joint]

        qpos[1] += .3

        env.set_state(qpos, np.zeros(36))

        mot = np.vstack((mot, qpos))

        # env.step(np.zeros(30))
        env.render()

    mot = np.delete(mot, 0, axis=0)

    # np.save(name, mot)

