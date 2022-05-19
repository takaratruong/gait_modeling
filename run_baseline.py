from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from my_pd_walker import PD_Walker2dEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-log_dir', type=str, default='gait_modeling_results/')
parser.add_argument('-exp_name', type=str, default='default_exp')
parser.add_argument('-time_steps', type=int, default=5e7)
parser.add_argument('-imp', action='store_true')  # use random impulses
parser.add_argument('-c_imp', action='store_true')  # use constant impulses

parser.add_argument('-f', type=float, default=300)  # define force
parser.add_argument('-ft', action='store_true')
parser.add_argument('-vis_ref', action='store_true')

parser.add_argument('-t', action='store_true')
parser.add_argument('-v', action='store_true')


if __name__ == "__main__":
    args = parser.parse_args()

    num_cpu = 8
    env = PD_Walker2dEnv(args=args)

    # Train model
    if args.t or args.ft:

        train_env = make_vec_env(lambda:env, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)
        eval_env = make_vec_env(lambda:env, n_envs=1)

        eval_callback = EvalCallback(eval_env, best_model_save_path=args.log_dir + args.exp_name, log_path=args.log_dir + args.exp_name, eval_freq=1000, deterministic=True, render=False)
        model = PPO("MlpPolicy", train_env, batch_size=32, use_sde=False, n_steps=512, n_epochs=20, gamma=.99, gae_lambda=.95, target_kl=.01, verbose=True,  clip_range=.1, ent_coef=.000585045, vf_coef=0.871923, max_grad_norm=1, learning_rate=5.05041e-5, tensorboard_log = args.log_dir + 'tf/')#, policy_kwargs=policy_kwargs)

        if args.ft:
            model.set_parameters(args.log_dir + args.exp_name + '/best_model')

        model.learn(total_timesteps=args.time_steps, callback=eval_callback, tb_log_name=args.exp_name)

    # Visualize best policy
    if args.v:
        model = PPO.load(args.log_dir + args.exp_name + '/best_model')
        env = make_vec_env(lambda:env, n_envs=1)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

            if dones:
                obs = env.reset()