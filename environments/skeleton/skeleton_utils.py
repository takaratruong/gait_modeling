import numpy as np


def reflect_sagital(pos, jnt2addr):
    pos_cpy = pos.copy()

    pos[jnt2addr('ground_pelvis_trans_z')] = -pos_cpy[jnt2addr('ground_pelvis_trans_z')]
    pos[jnt2addr('ground_pelvis_rot_x')] = -pos_cpy[jnt2addr('ground_pelvis_rot_x')]
    pos[jnt2addr('ground_pelvis_rot_y')] = -pos_cpy[jnt2addr('ground_pelvis_rot_y')]

    pos[jnt2addr('hip_flexion_r')] = pos_cpy[jnt2addr('hip_flexion_l')]
    pos[jnt2addr('hip_adduction_r')] = pos_cpy[jnt2addr('hip_adduction_l')]
    pos[jnt2addr('hip_rotation_r')] = pos_cpy[jnt2addr('hip_rotation_l')]
    pos[jnt2addr('walker_knee_r')] = pos_cpy[jnt2addr('walker_knee_l')]
    pos[jnt2addr('ankle_angle_r')] = pos_cpy[jnt2addr('ankle_angle_l')]
    pos[jnt2addr('subtalar_angle_r')] = pos_cpy[jnt2addr('subtalar_angle_l')]
    pos[jnt2addr('mtp_angle_r')] = pos_cpy[jnt2addr('mtp_angle_l')]

    pos[jnt2addr('hip_flexion_l')] = pos_cpy[jnt2addr('hip_flexion_r')]
    pos[jnt2addr('hip_adduction_l')] = pos_cpy[jnt2addr('hip_adduction_r')]
    pos[jnt2addr('hip_rotation_l')] = pos_cpy[jnt2addr('hip_rotation_r')]
    pos[jnt2addr('walker_knee_l')] = pos_cpy[jnt2addr('walker_knee_r')]
    pos[jnt2addr('ankle_angle_l')] = pos_cpy[jnt2addr('ankle_angle_r')]
    pos[jnt2addr('subtalar_angle_l')] = pos_cpy[jnt2addr('subtalar_angle_r')]
    pos[jnt2addr('mtp_angle_l')] = pos_cpy[jnt2addr('mtp_angle_r')]

    pos[jnt2addr('lumbar_bending')] = -pos_cpy[jnt2addr('lumbar_bending')]
    pos[jnt2addr('lumbar_rotation')] = -pos_cpy[jnt2addr('lumbar_rotation')]

    pos[jnt2addr('arm_flex_r')] = pos_cpy[jnt2addr('arm_flex_l')]
    pos[jnt2addr('arm_add_r')] = pos_cpy[jnt2addr('arm_add_l')]
    pos[jnt2addr('arm_rot_r')] = pos_cpy[jnt2addr('arm_rot_l')]
    pos[jnt2addr('elbow_flex_r')] = pos_cpy[jnt2addr('elbow_flex_l')]
    pos[jnt2addr('pro_sup_r')] = pos_cpy[jnt2addr('pro_sup_l')]
    pos[jnt2addr('wrist_flex_r')] = pos_cpy[jnt2addr('wrist_flex_l')]

    pos[jnt2addr('arm_flex_l')] = pos_cpy[jnt2addr('arm_flex_r')]
    pos[jnt2addr('arm_add_l')] = pos_cpy[jnt2addr('arm_add_r')]
    pos[jnt2addr('arm_rot_l')] = pos_cpy[jnt2addr('arm_rot_r')]
    pos[jnt2addr('elbow_flex_l')] = pos_cpy[jnt2addr('elbow_flex_r')]
    pos[jnt2addr('pro_sup_l')] = pos_cpy[jnt2addr('pro_sup_r')]
    pos[jnt2addr('wrist_flex_l')] = pos_cpy[jnt2addr('wrist_flex_r')]

    return pos


def reflect_action(action, act2id):
    act_cpy = action.copy()

    action[act2id('hip_flexion_r')] = act_cpy[act2id('hip_flexion_l')]
    action[act2id('hip_adduction_r')] = act_cpy[act2id('hip_adduction_l')]
    action[act2id('hip_rotation_r')] = act_cpy[act2id('hip_rotation_l')]
    action[act2id('walker_knee_r')] = act_cpy[act2id('walker_knee_l')]
    action[act2id('ankle_angle_r')] = act_cpy[act2id('ankle_angle_l')]
    action[act2id('subtalar_angle_r')] = act_cpy[act2id('subtalar_angle_l')]
    action[act2id('mtp_angle_r')] = act_cpy[act2id('mtp_angle_l')]

    action[act2id('hip_flexion_l')] = act_cpy[act2id('hip_flexion_r')]
    action[act2id('hip_adduction_l')] = act_cpy[act2id('hip_adduction_r')]
    action[act2id('hip_rotation_l')] = act_cpy[act2id('hip_rotation_r')]
    action[act2id('walker_knee_l')] = act_cpy[act2id('walker_knee_r')]
    action[act2id('ankle_angle_l')] = act_cpy[act2id('ankle_angle_r')]
    action[act2id('subtalar_angle_l')] = act_cpy[act2id('subtalar_angle_r')]
    action[act2id('mtp_angle_l')] = act_cpy[act2id('mtp_angle_r')]

    action[act2id('lumbar_bending')] = -act_cpy[act2id('lumbar_bending')]
    action[act2id('lumbar_rotation')] = -act_cpy[act2id('lumbar_rotation')]

    action[act2id('arm_flex_r')] = act_cpy[act2id('arm_flex_l')]
    action[act2id('arm_add_r')] = act_cpy[act2id('arm_add_l')]
    action[act2id('arm_rot_r')] = act_cpy[act2id('arm_rot_r')]
    action[act2id('elbow_flex_r')] = act_cpy[act2id('elbow_flex_l')]
    action[act2id('pro_sup_r')] = act_cpy[act2id('pro_sup_l')]
    action[act2id('wrist_flex_r')] = act_cpy[act2id('wrist_flex_l')]

    action[act2id('arm_flex_l')] = act_cpy[act2id('arm_flex_r')]
    action[act2id('arm_add_l')] = act_cpy[act2id('arm_add_r')]
    action[act2id('arm_rot_l')] = act_cpy[act2id('arm_rot_r')]
    action[act2id('elbow_flex_l')] = act_cpy[act2id('elbow_flex_r')]
    action[act2id('pro_sup_l')] = act_cpy[act2id('pro_sup_r')]
    action[act2id('wrist_flex_l')] = act_cpy[act2id('wrist_flex_r')]

    return action
