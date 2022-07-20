# """
# frame = mot.poses[:, 0]
#     file.skeleton.setPositions(frame)
#     xpos_old = file.skeleton.getJointWorldPositionsMap()
#
#     rot = R.from_euler('xyz', [frame[joint2idx['lumbar_bending']] * -1 - .1, (frame[joint2idx['lumbar_extension']]) * -1 - .4, frame[joint2idx['lumbar_rotation']] - .1])  # +.5
#     rot_old = rot.as_euler('xyz')
#     qpos_old = np.concatenate((
#         np.array(xpos['ground_pelvis'][[0, 2, 1]] - [0, 0, .15]),  # [0,2,1]
#         # np.array([1, 0, 0, 0]),  # root
#
#         np.array(rot.as_quat()[[3, 0, 1, 2]]),  # root
#         np.array([0, 0, 0]),  # chest
#         np.array([0, 0, 0]),  # neck
#
#         np.array([frame[joint2idx['arm_add_r']], frame[joint2idx['arm_flex_r']], frame[joint2idx['arm_rot_r']]]),
#         # right shoulder
#         np.array([frame[joint2idx['elbow_flex_r']]]),  # right elbow
#         np.array(
#             [-1 * frame[joint2idx['arm_add_l']], frame[joint2idx['arm_flex_l']], -1 * frame[joint2idx['arm_rot_l']]]),
#         # left shoulder
#         np.array([frame[joint2idx['elbow_flex_l']]]),  # left elbow
#
#         np.array([frame[joint2idx['hip_abuction_r']], -1 * frame[joint2idx['hip_flexion_r']] - .3,
#                   frame[joint2idx['hip_rotation_r']]]),  # right hip -.52
#         np.array([-1 * frame[joint2idx['knee_angle_r']]]),  # right knee
#         np.array([frame[joint2idx['subtalar_angle_r']], frame[joint2idx['ankle_angle_r']], 0]),  # right ankle
#
#         np.array([-1 * frame[joint2idx['hip_adduction_l']], -1 * frame[joint2idx['hip_flexion_l']] - .3,
#                   -1 * frame[joint2idx['hip_rotation_l']]]),  # left hip -.52
#         np.array([-1 * frame[joint2idx['knee_angle_l']]]),  # left knee
#         np.array([-1 * frame[joint2idx['subtalar_angle_l']], frame[joint2idx['ankle_angle_l']], 0]),  # left ankle
#     ))
#
#
#     file.skeleton.getBodyNode('pelvis').getWorldTransform() # <-3x3 rot .linear()  and 3x3 offset  .offset()
#     while True:
#         for i in range(0, mot.poses.shape[1]): #range(900, 1200):#mot.poses.shape[1]):
#             frame = mot.poses[:, i]
#             file.skeleton.setPositions(frame)
#             xpos = file.skeleton.getJointWorldPositionsMap()
#
#             # print(frame[joint2idx['pelvis_tx']], frame[joint2idx['pelvis_ty']], frame[joint2idx['pelvis_tz']])
#             # print(frame[joint2idx['pelvis_tilt']], frame[joint2idx['pelvis_list']], frame[joint2idx['pelvis_rotation']])
#
#             rot = R.from_euler('xyz', [frame[joint2idx['lumbar_bending']]*-1-.1, (frame[joint2idx['lumbar_extension']])*-1 -.4, frame[joint2idx['lumbar_rotation']]-.1]) #+.5
#             rot_euler = rot.as_euler('xyz')
#
#             """ POSITION """
#             qpos = np.concatenate((
#                 np.array(xpos['ground_pelvis'][[0, 2, 1]] - [0, 0, .15]), #[0,2,1]
#                 # np.array([1, 0, 0, 0]),  # root
#
#                 np.array(rot.as_quat()[[3, 0, 1, 2]]),  # root
#                 np.array([0, 0, 0]),  # chest
#                 np.array([0, 0, 0]),  # neck
#
#                 np.array([frame[joint2idx['arm_add_r']], frame[joint2idx['arm_flex_r']], frame[joint2idx['arm_rot_r']]]),  # right shoulder
#                 np.array([frame[joint2idx['elbow_flex_r']]]),  # right elbow
#                 np.array([-1 * frame[joint2idx['arm_add_l']], frame[joint2idx['arm_flex_l']], -1*frame[joint2idx['arm_rot_l']]]),  # left shoulder
#                 np.array([frame[joint2idx['elbow_flex_l']]]),  # left elbow
#
#                 np.array([frame[joint2idx['hip_abuction_r']], -1 * frame[joint2idx['hip_flexion_r']] -.3, frame[joint2idx['hip_rotation_r']]]),  # right hip -.52
#                 np.array([-1 * frame[joint2idx['knee_angle_r']]]),  # right knee
#                 np.array([frame[joint2idx['subtalar_angle_r']], frame[joint2idx['ankle_angle_r']], 0]),  # right ankle
#
#                 np.array([-1 * frame[joint2idx['hip_adduction_l']], -1 * frame[joint2idx['hip_flexion_l']]-.3, -1*frame[joint2idx['hip_rotation_l']]]),  # left hip -.52
#                 np.array([-1 * frame[joint2idx['knee_angle_l']]]),  # left knee
#                 np.array([-1 * frame[joint2idx['subtalar_angle_l']], frame[joint2idx['ankle_angle_l']], 0]),  # left ankle
#             ))
#
#             """ VELOCITY """
#             q_vel = (qpos - qpos_old)/.01
#             q_vel[0] += 1.25
#             q_vel = np.delete(q_vel, 3, 0) # delete element of quart and set to rot v in euler
#             rot_vel = (rot_euler - rot_old) / .01
#             q_vel[3:6] = rot_vel
#
#             # compiled_results = np.vstack((compiled_results, np.concatenate((qpos,q_vel))))
#             env.set_state(qpos, q_vel)
#
#             env.render()
#             print("HELLO")
#
#             time.sleep(.01)
#             rot_old = rot_euler
#             qpos_old = qpos
#
#         # compiled_results = np.delete(compiled_results, [0, 1], 0)
#
#         # np.save('SubjectData_1/Processed/' + name, compiled_results)
#
# # print(frame[joint2idx['pelvis_tx']], frame[joint2idx['pelvis_ty']], frame[joint2idx['pelvis_tz']])
# # print(frame[joint2idx['pelvis_tilt']], frame[joint2idx['pelvis_list']], frame[joint2idx['pelvis_rotation']])
#
# rot = R.from_euler('xyz', [frame[joint2idx['pelvis_tilt']], frame[joint2idx['pelvis_list']],
#                            frame[joint2idx['pelvis_rotation']]])  # +.5
# rot_euler = rot.as_euler('xyz')
# print(frame[joint2idx['pelvis_tx']])
# """ POSITION """
# qpos = np.concatenate((
#     np.array([frame[joint2idx['pelvis_tx']], frame[joint2idx['pelvis_tz']], frame[joint2idx['pelvis_ty']]]),
#     np.array(rot.as_quat()[[3, 0, 1, 2]]),  # root
#     np.array([0, 0, 0]),  # chest
#     np.array([0, 0, 0]),  # neck
#
#     np.array([frame[joint2idx['arm_add_r']], frame[joint2idx['arm_flex_r']], frame[joint2idx['arm_rot_r']]]),
#     # right shoulder
#     np.array([frame[joint2idx['elbow_flex_r']]]),  # right elbow
#     np.array([-1 * frame[joint2idx['arm_add_l']], frame[joint2idx['arm_flex_l']], -1 * frame[joint2idx['arm_rot_l']]]),
#     # left shoulder
#     np.array([frame[joint2idx['elbow_flex_l']]]),  # left elbow
#
#     np.array([frame[joint2idx['hip_abuction_r']], -1 * frame[joint2idx['hip_flexion_r']] - .3,
#               frame[joint2idx['hip_rotation_r']]]),  # right hip -.52
#     np.array([-1 * frame[joint2idx['knee_angle_r']]]),  # right knee
#     np.array([frame[joint2idx['subtalar_angle_r']], frame[joint2idx['ankle_angle_r']], 0]),  # right ankle
#
#     np.array([-1 * frame[joint2idx['hip_adduction_l']], -1 * frame[joint2idx['hip_flexion_l']] - .3,
#               -1 * frame[joint2idx['hip_rotation_l']]]),  # left hip -.52
#     np.array([-1 * frame[joint2idx['knee_angle_l']]]),  # left knee
#     np.array([-1 * frame[joint2idx['subtalar_angle_l']], frame[joint2idx['ankle_angle_l']], 0]),  # left ankle
# ))
#
#
# """
#         qpos[0:3] = [row['root_pos_x'], -row['root_pos_z'], row['root_pos_y'] +1]  # root ps
#         qpos[3:7] = rot.as_quat()[[3, 0, 1, 2]] # orient [1,0,0,0 ] #
#         qpos[7:10] = [0, 0, 0]  # chest
#         qpos[10:13] = [0, 0, 0]  # neck
#
#         qpos[13:16] = [row['acromial_r_x'], -row['acromial_r_z'], row['acromial_r_y']]
#         qpos[16] = row['elbow_r']
#
#         qpos[17:20] = [row['acromial_l_x'], -row['acromial_l_z'], row['acromial_l_y']]
#         qpos[20] = row['elbow_l']
#
#         qpos[21:24] = [row['hip_r_x'], -row['hip_r_z'], -row['hip_r_y']]
#         qpos[24] = -row['walker_knee_r']
#         qpos[25:28] = [0, -row['ankle_r'], 0]
#
#         qpos[28:31] = [row['hip_l_x'], -row['hip_l_z'], row['hip_l_y']]
#         qpos[31] = -row['walker_knee_l']
#         qpos[32:35] = [0, -row['ankle_l'], 0]
#         qpos = np.zeros(35)
#         qvel = np.zeros(34)
#
