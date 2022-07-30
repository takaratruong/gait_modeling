import wandb
from configs.config_loader import load_args
from rl_algs.amp_ppo import RL
from utils.video_callback import AMPVideoCallback
from utils.load_envs import load_environments

if __name__ == '__main__':
    args = load_args()

    print("Exp Name: ", args.exp_name)
    print("Config File:", args.config)
    print("Algorithm:", args.alg)

    # Initialize wandb
    run = wandb.init(project=args.project_name, config=args, name=args.exp_name, monitor_gym=True, dir=args.log_dir) if args.wandb else None

    # Load environments
    train_env, _, vid_env = load_environments(args, run)
    assert train_env is not None, 'Train env is None'

    # Create callbacks
    vid_callback = AMPVideoCallback(vid_env) if vid_env is not None else None

    # Instantiate algorithm
    ppo = RL(train_env, vid_callback, args, [128, 128])

    # Train
    ppo.collect_samples_multithread()





