from stable_baselines3.common.callbacks import BaseCallback, sync_envs_normalization

class VideoCallback(BaseCallback):
    def __init__(self, vid_env, eval_freq, verbose=0):
        super(VideoCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.vid_env = vid_env
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for _ in range(4):
                obs = self.vid_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, done, _ = self.vid_env.step(action)

            self.vid_env.close()
        return True
