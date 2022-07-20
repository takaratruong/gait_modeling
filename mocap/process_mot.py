import nimblephysics as nimble
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

import sys
sys.path.append('..')
from environments.humanoid.humanoid_env import HumanoidEnv
from environments.rajagopal.rajagopal_env import RajagopalEnv

from utils.config_loader import load_args


if __name__ == "__main__":
    args = load_args()
    df = pd.read_table("S02DN101_keenon.mot", skiprows=10)

    env = HumanoidEnv(args=args)
    env.reset()
    while True:
        for index, row in df.iterrows():
            qpos = np.zeros(35)
            qvel = np.zeros(34)

            rot = R.from_euler('xyz', [row['root_rot_x'], -row['root_rot_z'], row['root_rot_y']])  # +.5
            qpos[0:3] = [row['root_pos_x'], -row['root_pos_z'], row['root_pos_y'] +.85]  # root ps
            qpos[3:7] = rot.as_quat()[[3, 0, 1, 2]]  # orient [1,0,0,0 ] #
            qpos[7:10] = [0, 0, 0]  # chest -1*rot.as_euler('xyz') #
            qpos[10:13] = [0, 0, 0]  # neck

            qpos[13:16] = [row['acromial_r_x'], -row['acromial_r_z'], row['acromial_r_y']]
            qpos[16] = row['elbow_r']

            qpos[17:20] = [row['acromial_l_x'], -row['acromial_l_z'], row['acromial_l_y']]
            qpos[20] = row['elbow_l']

            qpos[21:24] = [row['hip_r_x'], -row['hip_r_z'], -row['hip_r_y']]
            qpos[24] = -row['walker_knee_r']
            qpos[25:28] = [0, -row['ankle_r'], 0]

            qpos[28:31] = [row['hip_l_x'], -row['hip_l_z'], row['hip_l_y']]
            qpos[31] = -row['walker_knee_l']
            qpos[32:35] = [0, -row['ankle_l'], 0]

            env.set_state(qpos, qvel)
            env.render()


"""


    # df = pd.read_table( "results_ik_right.mot", skiprows=10)
    # # df = pd.read_table( "SubjectData_1/IK/S01DN201/output/results_ik.mot", skiprows=10)
    # df = df.drop(columns='time')
    # df = df.apply(lambda x: np.deg2rad(x) if x.name not in ['pelvis_tz', 'pelvis_ty', 'pelvis_tz'] else x)

    # data_joints = set(df.keys())
    # env = RajagopalEnv(args=args)
    # env.reset()
    #
    # model_joints = set(env.model.joint_names)
    #
    # shared_joints = model_joints.intersection(data_joints)
    # print(len(shared_joints))
    # print(shared_joints)
    #
    # while True:
    #     for index, row in df.iterrows():
    #         qpos = np.zeros(51)
    #         qvel = np.zeros(51)
    #
    #         for joint_name in shared_joints:
    #             addr = env.model.get_joint_qpos_addr(joint_name)
    #             qpos[addr] = row[joint_name]
    #
    #         qpos[1] = -.175
    #
    #         env.set_state(qpos, qvel)
    #
    #         env.render()
    #
"""
