import numpy as np

class OrnsteinUhlenbeckActionNoise:
    
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1, x0=None):
        """
        Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
        based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        """
        
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalActionNoise:

    def __init__(self, size, sigma=0.2):

        self.size = size
        self.sigma = sigma

    def __call__(self):
        return np.random.randn(self.size) * self.sigma

    def reset(self):
        # added reset function because OUNoise uses it
        pass
