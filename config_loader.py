import configargparse

p = configargparse.ArgParser()

""" Logistics (Naming/Loading/Saving/etc..) """
p.add('-n', '--exp_name', required=True)
p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path')

p.add('--log_dir', type=str, default='results/')
p.add('--eval_freq', type=int, default=10)
p.add('--vid_freq', type=int, default=10)

""" Experiment Parameters """
# Visualizing
p.add('-v', '--visualize', action='store_true')  # Visualize policy
p.add('--vis_ref', action='store_true')  # Visualize reference motion

# Training parameters
p.add('-t', '--train_mode', required=True, type=str)  # either "fine_tune" or "standard"
p.add('-t_env', '--train_env', required=True, type=str)

p.add('--num_envs', type=int, default=20)
p.add('--num_steps', type=int, default=300)
p.add('--time_steps', type=int, default=10e8)
p.add('--num_epochs', type=int, default=10)


p.add('--frame_skip', type=int, default=20)
p.add('--max_ep_time', type=float, default=10.0)

# Policy Parameters
p.add('--gait_policy', type=str)  # Experiment folder name defining the gait policy (loaded from results)

# Perturbation Parameters
p.add('-rp',  '--rand_perturbation', action='store_true')  # Use random perturbation
p.add('-cp',  '--const_perturbation', action='store_true')  # Use constant perturbation (if neither then no impacts)
p.add('-p_frc', '--perturbation_force', type=float, default=90)
p.add('-p_frc_min', '--min_perturbation_force_mag', type=float, default=70)
p.add('-p_dur', '--perturbation_duration', type=float, default=.2)
p.add('-p_del', '--perturbation_delay', type=float, default=3.5)

# Reward/Action weights
p.add('--phase_action_mag', type=float, default=50)

p.add('--orient_weight', type=float, default=.3)
p.add('--joint_weight', type=float, default=.4)
p.add('--pos_weight', type=float, default=.3)


def load_args():
    return p.parse_args()


if __name__ == "__main__":
    options = p.parse_args()

    print(options)
    print("----------")
    print(p.format_help())
    print("----------")
    print(p.format_values())