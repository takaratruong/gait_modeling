import numpy as np
import glob

from scipy.spatial.transform import Rotation as R
import pandas as pd
import sys

sys.path.append('..')
from environments.humanoid.humanoid_treadmill_env import HumanoidTreadmillEnv
from utils.config_loader import load_args

"""
Process data to have same shape and values of observation space (aside from phase) 
"""


if __name__ == "__main__":
    args = load_args()

    # name = 'S02DN101'

    df = pd.read_table("S02DN101_v5.mot", skiprows=10)
    df['ground_pelvis_trans_x'] -= df['ground_pelvis_trans_x'].mean()
    df['ground_pelvis_trans_z'] -= df['ground_pelvis_trans_z'].mean()

    env = HumanoidTreadmillEnv(args=args)
    jnt2adr = env.model.get_joint_qpos_addr

    mot = np.zeros(36 + 35 - 2) # remove treadmill
    qpos_old = np.zeros(36)
    rot_old = R.from_euler('xyz', [0, 0, 0])

    for index, row in df.iterrows():
        qpos = np.zeros(36)
        qvel = np.zeros(35)

        qpos[0] = row['ground_pelvis_trans_x']
        qpos[1] = -row['ground_pelvis_trans_z']
        qpos[2] = row['ground_pelvis_trans_y'] + .18  # treadmill height offset

        rot = R.from_euler('xyz', [row['ground_pelvis_rot_x'], -row['ground_pelvis_rot_z'], row['ground_pelvis_rot_y']])

        quat = rot.as_quat()
        qpos[3] = quat[3]
        qpos[4] = quat[0]
        qpos[5] = quat[1]
        qpos[6] = quat[2]

        qpos[jnt2adr('chest_x')] = row['lumbar_bending']
        qpos[jnt2adr('chest_y')] = -row['lumbar_extension']
        qpos[jnt2adr('chest_z')] = row['lumbar_rotation']

        qpos[jnt2adr('right_shoulder_x')] = row['arm_add_r']
        qpos[jnt2adr('right_shoulder_y')] = -row['arm_flex_r']
        qpos[jnt2adr('right_shoulder_z')] = row['arm_rot_r']
        qpos[jnt2adr('right_elbow')] = row['elbow_flex_r']

        qpos[jnt2adr('left_shoulder_x')] = -row['arm_add_l']
        qpos[jnt2adr('left_shoulder_y')] = -row['arm_flex_l']
        qpos[jnt2adr('left_shoulder_z')] = -row['arm_rot_l']
        qpos[jnt2adr('left_elbow')] = row['elbow_flex_l']

        qpos[jnt2adr('right_hip_x')] = row['hip_adduction_r']
        qpos[jnt2adr('right_hip_y')] = -row['hip_flexion_r']
        qpos[jnt2adr('right_hip_z')] = row['hip_rotation_r']
        qpos[jnt2adr('right_knee')] = -row['walker_knee_r']

        qpos[jnt2adr('right_ankle_y')] = -row['ankle_angle_r']
        qpos[jnt2adr('right_ankle_z')] = row['subtalar_angle_r']

        qpos[jnt2adr('left_hip_x')] = -row['hip_adduction_l']
        qpos[jnt2adr('left_hip_y')] = -row['hip_flexion_l']
        qpos[jnt2adr('left_hip_z')] = -row['hip_rotation_l']
        qpos[jnt2adr('left_knee')] = -row['walker_knee_l']

        qpos[jnt2adr('left_ankle_y')] = -row['ankle_angle_l']
        qpos[jnt2adr('left_ankle_z')] = -row['subtalar_angle_l']

        env.set_state(qpos, qvel)
        # env.step(np.zeros(30))
        env.render()

        if index > 0:
            vel = ((qpos - qpos_old) / .01) / 10 # 100 hz, divide 10 to match obs
            rot_vel = ((rot.as_euler('xyz') - rot_old.as_euler('xyz')) / .01) / 10

            qvel[0:3] = vel[0:3]
            qvel[3:6] = rot_vel
            qvel[6:-1] = vel[7:-1]

        mot = np.vstack((mot, np.hstack((qpos[:-1], qvel[:-1]))))
        qpos_old = qpos
        rot_old = rot

    mot = np.delete(mot, 0, axis=0)

    np.save('S02DN101_humanoid', mot)
