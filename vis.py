import numpy as np
import torch.utils.data
from configs.config_loader import load_args
from rl_algs.amp_models import ActorCriticNet
from utils.load_envs import load_environments
import ipdb

if __name__ == '__main__':
    args = load_args()
    _, env, _ = load_environments(args)

    if args.policy_path is None:
        print('---------------------------------------------------------------------------------------')
        print('Specify policy path via argument command flag: --gait_policy relative/path/to/policy.pt')
        print('Running zero-action policy')
        print('---------------------------------------------------------------------------------------')

    # Recreate model and load trained weights
    model = None
    if args.policy_path:
        model = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], [128, 128])
        model.load_state_dict(torch.load(args.gait_policy))  # relative file path
        model.cuda()

    # Loop Policy
    while True:
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                if model is not None:
                    act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
                else:
                    act = np.zeros(env.action_space.shape[0]).reshape(1, -1)

            next_state, reward, done, info = env.step(act)
            env.render()

            state = next_state
