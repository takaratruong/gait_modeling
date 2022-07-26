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
from environments.humanoid.humanoid_env import HumanoidEnv
from environments.rajagopal.rajagopal_env import RajagopalEnv
from environments.skeleton.skeleton_env import SkeletonEnv

from utils.config_loader import load_args

#
# def flip_ball_joint(x, y, z):
#     new_x = row[x]
#     new_y = -row[z]
#     new_z = row[y]
#     return new_x, new_y, new_z
#

if __name__ == "__main__":
    args = load_args()
    df = pd.read_table("S02DN101_v5.mot", skiprows=10)

    env = SkeletonEnv(args=args, xml_file='skeleton.xml')
    env.reset()
    qpos = np.zeros(36)  # last one is treadmill
    qpos[1] = 1.5
    env.set_state(qpos, np.zeros(36))
    while True:
        env.reset()

        done = False
        while not done:
            next_state, rew, done, _ = env.step(np.zeros(30))
            env.render()



    assert False

    """ GET WALK REFERENCE """
    jnt2adr = env.model.get_joint_qpos_addr

    df = df.drop([x for x in range(0, len(df)) if x < 105 or x > 210])

    data = np.loadtxt("S02DN101_v5.mot", skiprows=11)
    smooth = interp1d(np.arange(0, 2), np.vstack((data[211], data[104])), axis=0)
    df_smooth = pd.DataFrame(smooth([.5]), columns=df.columns)

    df = df.append(df_smooth)

    df = df.drop(columns='time')

    # jnt2adr = env.model.get_joint_qpos_addr
    mot = np.zeros(36)
    # while True:
    for index, row in df.iterrows():
        # if index in range(106, 211): #900 1300

        qpos = np.zeros(36)  # last one is treadmill
        qvel = np.zeros(36)

        qpos[jnt2adr('treadmill')] = 0
        for joint in env.model.joint_names:
            if joint != 'treadmill':
                addr = env.model.get_joint_qpos_addr(joint)
                qpos[addr] = row[joint]

        qpos[0] = 0
        qpos[1] += .3
        qpos[2] = 0

        qpos[-1] = 1.25

        mot = np.vstack((mot, qpos))

        # env.set_state(qpos, np.zeros(36))

        # env.step(np.zeros(30))
        # env.render()
        # ipdb.set_trace()

    mot = np.delete(mot, 0, axis=0)

    np.save('skeleton_walk_ref', mot)

