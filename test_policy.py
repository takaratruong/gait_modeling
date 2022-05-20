import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-log_dir', type=str, default='results/')
parser.add_argument('-exp_name', type=str, default='default_exp')
parser.add_argument('-time_steps', type=int, default=5e7)
parser.add_argument('-imp', action='store_true')  # use random impulses
parser.add_argument('-c_imp', action='store_true')  # use constant impulses

parser.add_argument('-f', type=float, default=300)  # define force
parser.add_argument('-ft', action='store_true')
parser.add_argument('-vis_ref', action='store_true')

parser.add_argument('-t', action='store_true')
parser.add_argument('-v', action='store_true')

if __name__ == '__main__':
   import json
 
   import torch
   import torch.optim as optim
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Variable
   import torch.utils.data
   from model import ActorCriticNet
   import os
   import numpy as np
   from my_pd_walker import PD_Walker2dEnv
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
   seed = 3#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # create environment from the configuration file
   args = parser.parse_args()
   env = PD_Walker2dEnv(args=args)
   print("env_created")

   num_inputs = env.observation_space.shape[0]
   num_outputs = env.action_space.shape[0]
   model = ActorCriticNet(num_inputs, num_outputs, [128, 128])
   model.load_state_dict(torch.load("stats/test/iter1999.pt"))
   model.cuda()

   env.reset()
   obs = env.observe()
   print(obs)
   average_gating = np.zeros(8)
   average_gating_sum = 0
   for i in range(10000):
      with torch.no_grad():
         act = model.sample_best_actions(torch.from_numpy(obs).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()

      obs, rew, done, _ = env.step(act)
      #env.reset_time_limit() 
      env.render()
      print(rew)
      if done:
         env.reset()

      import time; time.sleep(0.02)