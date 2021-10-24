import numpy as np
import torch
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
class PredictEnv:
    def __init__(self, model,replay_buffer, env_name, model_type):
        self.model = model
        self.env_name = env_name
        self.model_type = model_type
        self.replay_buffer=replay_buffer

    def _termination_fn(self, env_name, obs, act, next_obs):
        # TODO
        if env_name == "Hopper-v2" or env_name == 'MBRLHopper-v0':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = np.isfinite(next_obs).all(axis=-1) \
                       * np.abs(next_obs[:, 1:] < 100).all(axis=-1) \
                       * (height > .7) \
                       * (np.abs(angle) < .2)

            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "Walker2d-v2" or env_name == 'MBRLWalker-v0':
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif env_name == "AntTruncatedObs-v2":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            x = next_obs[:, 0]
            not_done = 	np.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)

            done = ~not_done
            done = done[:,None]
            return done
        elif env_name == "MBRLAnt-v0":
            assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
            x = next_obs[:, 0]
            not_done = 	np.isfinite(next_obs).all(axis=-1) \
                        * (x >= 0.2) \
                        * (x <= 1.0)

            done = ~not_done
            done = done[:,None]
            return done
        elif 'walker_' in env_name:
            torso_height =  next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in env_name or 'walker_5' in env_name:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            done = done[:, None]
            return done
        elif 'Stoch' in env_name:
            angle1 =  next_obs[:, 1]
            angle2 = next_obs[:, 2]
            not_done = (abs(angle1) < np.pi/2) \
                       * (abs(angle2) < 2)
            done = ~not_done
            done = done[:, None]
            return done

    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))

        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means, 0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        model_means = ensemble_model_means[model_idxes, batch_idxes]
        model_stds = ensemble_model_stds[model_idxes, batch_idxes]

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:, :1], samples[:, 1:]
        terminals = self._termination_fn(self.env_name, obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            rewards = rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
        return next_obs, rewards, terminals, info



    def get_forward_prediction_random_ensemble_t(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = torch.cat((obs,act),dim=-1)
        # inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict_t(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict_t(inputs, factored=True)
        
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + torch.randn(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        
        return samples

    def get_forward_prediction_random_ensemble(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False
        if(torch.is_tensor(obs)):
            obs = obs.detach().cpu().numpy()
        if(torch.is_tensor(act)):
            act = act.detach().cpu().numpy()
        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[model_idxes, batch_idxes]
        
        return samples

    def get_forward_prediction(self, obs, act, deterministic=False):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        if(torch.is_tensor(obs)):
            obs = obs.detach().cpu().numpy()
        if(torch.is_tensor(act)):
            act = act.detach().cpu().numpy()
        # obs = obs[0,:,:]
        # act = np.expand_dims(act, axis=0)
        # act = np.tile(act,(self.model.elite_size,1,1))



        inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        num_models, batch_size, _ = ensemble_model_means.shape
        if self.model_type == 'pytorch':
            model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        else:
            model_idxes = self.model.random_inds(batch_size)
        batch_idxes = np.arange(0, batch_size)

        samples = ensemble_samples[self.model.elite_model_idxes, :]
        return samples

    def get_forward_prediction_t(self, obs, act, deterministic=False):
        # import ipdb;ipdb.set_trace()
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False


        inputs = torch.cat((obs,act),dim=-1)
        # inputs = np.concatenate((obs, act), axis=-1)
        if self.model_type == 'pytorch':
            ensemble_model_means, ensemble_model_vars = self.model.predict_batch_t(inputs)
        else:
            ensemble_model_means, ensemble_model_vars = self.model.predict_batch_t(inputs, factored=True)
        
        ensemble_model_means[:, :, 1:] += obs
        ensemble_model_stds = torch.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + torch.randn(size=ensemble_model_means.shape).to(device) * ensemble_model_stds

        
        # num_models, batch_size, _ = ensemble_model_means.shape
        # if self.model_type == 'pytorch':
        #     model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        # else:
        #     model_idxes = self.model.random_inds(batch_size)
        # batch_idxes = np.arange(0, batch_size)

        # samples = ensemble_samples[self.model.elite_model_idxes, :]
        return ensemble_samples

    def train(self):
        state = self.replay_buffer.state[:self.replay_buffer.size,:]
        action = self.replay_buffer.action[:self.replay_buffer.size,:]
        reward = self.replay_buffer.reward[:self.replay_buffer.size]
        next_state = self.replay_buffer.next_state[:self.replay_buffer.size,:]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)       

        return self.model.train(inputs,labels,batch_size=256, holdout_ratio=0.2)

    def train_low_mem(self,max_epochs=None):
        state = self.replay_buffer.state[:self.replay_buffer.size,:]
        action = self.replay_buffer.action[:self.replay_buffer.size,:]
        reward = self.replay_buffer.reward[:self.replay_buffer.size]
        next_state = self.replay_buffer.next_state[:self.replay_buffer.size,:]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)       

        return self.model.train_low_mem(inputs,labels,batch_size=256, holdout_ratio=0.2,max_epochs=max_epochs)