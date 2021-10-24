import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


    def get_logprob(self,obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        
        # log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.log_std_layer(net_out)
        # log_std = self.min_log_std + log_std * (
        #                 self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        # import ipdb;ipdb.set_trace()
        actions_u = torch.log1p(2*actions/(1-actions)+1e-7) / 2
        # torch.atanh(actions)
        logp_pi = pi_distribution.log_prob(actions_u).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - actions_u - F.softplus(-2*actions_u))).sum(axis=1)
        return logp_pi
    #     return logp_pi
        # def get_logprob(self,obs, actions):
        #     net_out = self.net(obs)
        #     mu = self.mu_layer(net_out)
        #     log_std = self.log_std_layer(net_out)
        #     log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #     std = torch.exp(log_std)
        #     pi_distribution = Normal(mu, std)
        #     logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        #     logp_pi -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=1)

        #     return logp_pi


class BC_Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.min_log_std=-20
        self.max_log_std=2


    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None


        
        return pi_action, logp_pi


    def get_logprob(self,obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        # log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi
        # def get_logprob(self,obs, actions):
        #     net_out = self.net(obs)
        #     mu = self.mu_layer(net_out)
        #     log_std = self.log_std_layer(net_out)
        #     log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        #     std = torch.exp(log_std)
        #     pi_distribution = Normal(mu, std)
        #     logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        #     logp_pi -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=1)

        #     return logp_pi

class awacMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # print("Using the special policy")
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)
        
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None


        return pi_action, logp_pi

    def get_logprob(self,obs, actions):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi


class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1) # Critical to ensure q has right shape.


class MLPQRankFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_size=1):
        super().__init__()
        self.q_i = mlp([obs_dim + act_dim] + list(hidden_sizes), activation)
        self.q_j = mlp([hidden_sizes[-1]]+[1],activation)
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q_i_out = self.q_i(torch.cat([obs, act], dim=-1))
        q = self.q_j(q_i_out)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

    def get_features(self,obs,act):
        q_i_out = self.q_i(torch.cat([obs, act], dim=-1))
        return q_i_out

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(32,32),special_policy=None,
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # self.special_policy = special_policy
        # build policy and value functions
        if special_policy is 'awac':
            self.pi = awacMLPActor(obs_dim, act_dim, (32,32), activation, act_limit).to(device)
            # self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
            # self.pi = awac_core.GaussianPolicy([256,256,256,256],obs_dim, act_dim, max_log_std=0, min_log_std=-6, std_architecture="values")
        elif special_policy is 'bc':
            self.pi = BC_Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        else:
            print("Initializing actor for SAC")
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        
        
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.v = MLPVFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act_batch(self, obs, deterministic=False,with_logprob=False):
        with torch.no_grad():
            # if self.special_policy is 'two_timescale':
            #     zeros_t = torch.zeros((obs.shape[0],1)).float().to(device)
            #     a, _ =self.pi(torch.cat((obs,zeros_t),dim=1), deterministic, False)
            # else:
            a, logp = self.pi(obs, deterministic, True)
            if with_logprob:
                return a,logp
            return a

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().data.numpy().flatten()