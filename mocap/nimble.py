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

import sys
sys.path.append('..')
from environments.humanoid.human_env_test2 import Humanoid_test_env2
from utils.config_loader import load_args

joint2idx = {
    'pelvis_tilt': 0,
    'pelvis_list': 1,
    'pelvis_rotation': 2,
    'pelvis_tx': 3,
    'pelvis_ty': 4,
    'pelvis_tz': 5,
    'hip_flexion_r': 6,
    'hip_abuction_r': 7,
    'hip_rotation_r': 8,
    'knee_angle_r': 9,
    'ankle_angle_r': 10,
    'subtalar_angle_r': 11,
    'mtp_angle_r': 12,
    'hip_flexion_l': 13,
    'hip_adduction_l': 14,
    'hip_rotation_l': 15,
    'knee_angle_l': 16,
    'ankle_angle_l': 17,
    'subtalar_angle_l': 18,
    'mtp_angle_l': 19,
    'lumbar_extension': 20,
    'lumbar_bending': 21,
    'lumbar_rotation': 22,
    'arm_flex_r': 23,
    'arm_add_r': 24,  # abd?
    'arm_rot_r': 25,
    'elbow_flex_r': 26,
    'pro_sup_r': 27,
    'wrist_flex_r': 28,
    'wrist_dev_r': 29,
    'arm_flex_l': 30,
    'arm_add_l': 31,
    'arm_rot_l': 32,
    'elbow_flex_l': 33,
    'pro_sup_l': 34,
    'wrist_flex_l': 35,
    'wrist_dev_l': 36,
}

def dist(x, y):
    return np.linalg.norm(x - y)

class Mocap():
    def __init__(self, sample_batch_size):
        self.file: nimble.biomechanics.OpensSimFile = None  # subject
        self.motion: nimble.biomechanics.OpenSimMot = None  # impact trial

        self.sample_batch_size = sample_batch_size

    def load_open_sim_file(self, path_osim: str) -> None:
        self.file = nimble.biomechanics.OpenSimParser.parseOsim(path_osim)

    def load_motion_file(self, path_mot: str) -> None:
        assert self.file is not None, 'load open sim file first'
        self.motion = nimble.biomechanics.OpenSimParser.loadMot(self.file.skeleton, path_mot)

    def visualize_motion(self) -> None:
        assert self.file or self.motion is not None, 'load open sim file and motion file'

        world = nimble.simulation.World()
        world.addSkeleton(self.file.skeleton)
        gui = nimble.NimbleGUI(world)

        gui.serve(8080)
        while True:
            for i in range(self.motion.poses.shape[1]):
                self.file.skeleton.setPositions(self.motion.poses[:, i])
                gui.nativeAPI().renderSkeleton(self.file.skeleton)
                # time.sleep(.01)
                # time.sleep(.005)
            # time.sleep(1)
        #gui.blockWhileServing()

    def sample_recovery(self):
        pass

    def sample(self):
        pass

if __name__ == "__main__":
    args = load_args()

    # mcp = Mocap()
    # mcp.load_open_sim_file('/home/takaraet/gait_modeling/mocap/Rajagopal_Scaled_v2.osim')
    # mcp.load_motion_file('/home/takaraet/gait_modeling/mocap/results_ik.mot')
    # mcp.visualize_motion()
    
#    path_osim = '/home/takaraet/gait_modeling/mocap/Rajagopal_Scaled_v2.osim'
    path_osim = '/home/takaraet/gait_modeling/mocap/Rajagopal_Scaled.osim'
    file: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(path_osim)

    name = 'S01DN203'
    path_mot = '/home/takaraet/gait_modeling/mocap/SubjectData_1/IK/' + name + '/output/results_ik.mot'
    mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(file.skeleton, path_mot)


    #pprint.pprint(mot.poses[:, 100])

    if args.visualize:
        world = nimble.simulation.World()
        world.addSkeleton(file.skeleton)
        gui = nimble.NimbleGUI(world)

        gui.serve(8080)

        while True:
            for i in range(mot.poses.shape[1]):
                file.skeleton.setPositions(mot.poses[:, i])
                gui.nativeAPI().renderSkeleton(file.skeleton)
                time.sleep(.02)

    # gui.blockWhileServing()
    #
    # print('\''+ file.skeleton.getDofs()[num].getName()+'\':', str(num)+',')
    #

    file.skeleton.setPositions(mot.poses[:, 0])
    xpos = file.skeleton.getJointWorldPositionsMap()

    """
    CALC BODY GEOM 
    """
    # pprint.pprint(xpos)
    femur_r = dist(xpos['hip_r'], xpos['walker_knee_r'])  # thigh
    tibia_r = dist(xpos['walker_knee_r'], xpos['ankle_r']) #shin
    foot_r = dist(xpos['ankle_r'], xpos['mtp_r'])

    femur_l = dist(xpos['hip_l'], xpos['walker_knee_l'])  # thigh
    tibia_l = dist(xpos['walker_knee_l'], xpos['ankle_l']) # shin
    foot_l = dist(xpos['ankle_l'], xpos['mtp_l'])

    # print(femur_l, femur_r)
    # print(tibia_l, tibia_r)
    # print(foot_l, foot_r)


    b = dist(xpos['ground_pelvis'], xpos['acromial_l'])
    a = dist(xpos['acromial_l'], xpos['acromial_r'])
    torso = b**2 * (1 - (a**2 / (2*b)**2))
    # print(torso)
    # assert False

    """
    RUN TRAJECTORY ON MUJOCO
    """

    # env = Humanoid_test_env2(args=args,xml_file='subject1.xml')
    env = Humanoid_test_env2(args=args,xml_file='humanoid.xml')

    env.reset()

    state = env.observe()

    # quart = state[3:7]
    # init_rot = R.from_quat(quart)
    # print(quart)
    # print(init_rot.as_euler('xyz',degrees=True))
    # print('---------------------------------')
    #
    #
    # frame = mot.poses[:,0]
    #
    # rot = R.from_euler('xyz',  [-1*frame[joint2idx['pelvis_tx']], frame[joint2idx['pelvis_ty']], frame[joint2idx['pelvis_tz']]])
    # rot2 = R.from_euler('xyz', [0,0,0])
    #
    # print("rot", rot.as_euler('xyz', degrees=True))
    # print("rot2", rot2.as_euler('xyz', degrees=True))
    #
    # print("rot", rot.as_quat())
    # print("rot2", rot2.as_quat())

    compiled_results = np.zeros(34+35)

    """VISUALIZE"""
    frame = mot.poses[:, 0]
    file.skeleton.setPositions(frame)
    xpos_old = file.skeleton.getJointWorldPositionsMap()

    rot = R.from_euler('xyz', [frame[joint2idx['lumbar_bending']] * -1 - .1, (frame[joint2idx['lumbar_extension']]) * -1 - .4, frame[joint2idx['lumbar_rotation']] - .1])  # +.5
    rot_old = rot.as_euler('xyz')
    qpos_old = np.concatenate((
        np.array(xpos['ground_pelvis'][[0, 2, 1]] - [0, 0, .15]),  # [0,2,1]
        # np.array([1, 0, 0, 0]),  # root

        np.array(rot.as_quat()[[3, 0, 1, 2]]),  # root
        np.array([0, 0, 0]),  # chest
        np.array([0, 0, 0]),  # neck

        np.array([frame[joint2idx['arm_add_r']], frame[joint2idx['arm_flex_r']], frame[joint2idx['arm_rot_r']]]),
        # right shoulder
        np.array([frame[joint2idx['elbow_flex_r']]]),  # right elbow
        np.array(
            [-1 * frame[joint2idx['arm_add_l']], frame[joint2idx['arm_flex_l']], -1 * frame[joint2idx['arm_rot_l']]]),
        # left shoulder
        np.array([frame[joint2idx['elbow_flex_l']]]),  # left elbow

        np.array([frame[joint2idx['hip_abuction_r']], -1 * frame[joint2idx['hip_flexion_r']] - .3,
                  frame[joint2idx['hip_rotation_r']]]),  # right hip -.52
        np.array([-1 * frame[joint2idx['knee_angle_r']]]),  # right knee
        np.array([frame[joint2idx['subtalar_angle_r']], frame[joint2idx['ankle_angle_r']], 0]),  # right ankle

        np.array([-1 * frame[joint2idx['hip_adduction_l']], -1 * frame[joint2idx['hip_flexion_l']] - .3,
                  -1 * frame[joint2idx['hip_rotation_l']]]),  # left hip -.52
        np.array([-1 * frame[joint2idx['knee_angle_l']]]),  # left knee
        np.array([-1 * frame[joint2idx['subtalar_angle_l']], frame[joint2idx['ankle_angle_l']], 0]),  # left ankle
    ))


    file.skeleton.getBodyNode('pelvis').getWorldTransform() # <-3x3 rot .linear()  and 3x3 offset  .offset()
    #while True:
    for i in range(0, mot.poses.shape[1]): #range(900, 1200):#mot.poses.shape[1]):
        frame = mot.poses[:, i]
        file.skeleton.setPositions(frame)
        xpos = file.skeleton.getJointWorldPositionsMap()

        # print(frame[joint2idx['pelvis_tx']], frame[joint2idx['pelvis_ty']], frame[joint2idx['pelvis_tz']])
        # print(frame[joint2idx['pelvis_tilt']], frame[joint2idx['pelvis_list']], frame[joint2idx['pelvis_rotation']])

        rot = R.from_euler('xyz', [frame[joint2idx['lumbar_bending']]*-1-.1, (frame[joint2idx['lumbar_extension']])*-1 -.4, frame[joint2idx['lumbar_rotation']]-.1]) #+.5
        rot_euler = rot.as_euler('xyz')

        """ POSITION """
        qpos = np.concatenate((
            np.array(xpos['ground_pelvis'][[0, 2, 1]] - [0, 0, .15]), #[0,2,1]
            # np.array([1, 0, 0, 0]),  # root

            np.array(rot.as_quat()[[3, 0, 1, 2]]),  # root
            np.array([0, 0, 0]),  # chest
            np.array([0, 0, 0]),  # neck

            np.array([frame[joint2idx['arm_add_r']], frame[joint2idx['arm_flex_r']], frame[joint2idx['arm_rot_r']]]),  # right shoulder
            np.array([frame[joint2idx['elbow_flex_r']]]),  # right elbow
            np.array([-1 * frame[joint2idx['arm_add_l']], frame[joint2idx['arm_flex_l']], -1*frame[joint2idx['arm_rot_l']]]),  # left shoulder
            np.array([frame[joint2idx['elbow_flex_l']]]),  # left elbow

            np.array([frame[joint2idx['hip_abuction_r']], -1 * frame[joint2idx['hip_flexion_r']] -.3, frame[joint2idx['hip_rotation_r']]]),  # right hip -.52
            np.array([-1 * frame[joint2idx['knee_angle_r']]]),  # right knee
            np.array([frame[joint2idx['subtalar_angle_r']], frame[joint2idx['ankle_angle_r']], 0]),  # right ankle

            np.array([-1 * frame[joint2idx['hip_adduction_l']], -1 * frame[joint2idx['hip_flexion_l']]-.3, -1*frame[joint2idx['hip_rotation_l']]]),  # left hip -.52
            np.array([-1 * frame[joint2idx['knee_angle_l']]]),  # left knee
            np.array([-1 * frame[joint2idx['subtalar_angle_l']], frame[joint2idx['ankle_angle_l']], 0]),  # left ankle
        ))

        """ VELOCITY """
        q_vel = (qpos - qpos_old)/.01
        q_vel[0] += 1.25
        q_vel = np.delete(q_vel, 3, 0) # delete element of quart and set to rot v in euler
        rot_vel = (rot_euler - rot_old) / .01
        q_vel[3:6] = rot_vel

        compiled_results = np.vstack((compiled_results, np.concatenate((qpos,q_vel))))
        env.set_state(qpos, q_vel)
        env.render()
        time.sleep(.01)
        rot_old = rot_euler
        qpos_old = qpos

    compiled_results = np.delete(compiled_results, [0, 1], 0)

    # np.save('SubjectData_1/Processed/' + name, compiled_results)

