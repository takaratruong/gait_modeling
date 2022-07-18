import wandb
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from utils.config_loader import load_args, POLICY_KWARGS
from utils.load_envs import load_envs
from utils.video_callback import VideoCallback


if __name__ == '__main__':

    args = load_args()

    run = wandb.init(project="gait-modeling", config=args, name=args.exp_name, sync_tensorboard=True, monitor_gym=True, save_code=True, dir=args.log_dir)

    train_env, eval_env, vid_env = load_envs(args, run)

    eval_callback = EvalCallback(eval_env, best_model_save_path=args.log_dir + args.exp_name, log_path=args.log_dir + 'models/' + args.exp_name, eval_freq=args.eval_freq * args.num_epochs * args.num_envs, deterministic=True, render=False)
    vid_callback = VideoCallback(vid_env, eval_freq=6000)

    model = PPO("MlpPolicy", train_env, batch_size=(args.num_envs*args.num_steps)//4, use_sde=False, n_steps= args.num_steps, n_epochs=args.num_epochs, gamma=.99, gae_lambda=.95,
                target_kl=.01, verbose=True,  clip_range=.1, ent_coef=.000585045, vf_coef=0.871923, max_grad_norm=10, learning_rate=1e-4,
                tensorboard_log=args.log_dir + 'tf/', policy_kwargs=POLICY_KWARGS)

    model.load('results/fg/best_model')

    for _ in range(4):
        obs = vid_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _ = vid_env.step(action)

    vid_env.close()

    #model.learn(total_timesteps=args.time_steps, callback=[eval_callback, vid_callback], tb_log_name=args.exp_name)

    #run.finish()
