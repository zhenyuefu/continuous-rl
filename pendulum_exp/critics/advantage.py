import copy
from abstract import Critic, Arrayable, ParametricFunction, Tensorable
import torch
from torch import Tensor
import numpy as np
from convert import arr_to_th, check_array, check_tensor
from optimizer import setup_optimizer
from gym.spaces import Box, Discrete
from models import MLP, ContinuousAdvantageMLP, NormalizedMLP
from stateful import CompoundStateful
from nn import soft_update

class AdvantageCritic(CompoundStateful, Critic):
    def __init__(self,
                 dt: float, gamma: float, lr: float, tau: float, optimizer: str,
                 val_function: ParametricFunction, adv_function: ParametricFunction) -> None:
        CompoundStateful.__init__(self)
        self._reference_obs: Tensor = None
        self._val_function = val_function
        self._adv_function = adv_function
        self._target_val_function = copy.deepcopy(val_function)
        self._target_adv_function = copy.deepcopy(adv_function)

        self._adv_optimizer = \
            setup_optimizer(self._adv_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=1,
                            weight_decay=0)
        self._val_optimizer = \
            setup_optimizer(self._val_function.parameters(),
                            opt_name=optimizer, lr=lr, dt=dt,
                            inverse_gradient_magnitude=dt,
                            weight_decay=0)

        self._dt = dt
        self._gamma = gamma
        self._tau = tau

        self._device = 'cpu'

    def optimize(self, obs: Arrayable, action: Arrayable, max_action: Tensor,
                 next_obs: Arrayable, max_next_action: Tensor, reward: Arrayable,
                 done: Arrayable, time_limit: Arrayable, weights: Arrayable) -> Tensor:
        if self._reference_obs is None:
            self._reference_obs = arr_to_th(obs, self._device)

        action = arr_to_th(action, self._device).type_as(max_action)
        reward = arr_to_th(reward, self._device)
        weights = arr_to_th(check_array(weights), self._device)
        done = arr_to_th(check_array(done).astype('float'), self._device)

        v = self._val_function(obs).squeeze()
        reference_v = self._val_function(self._reference_obs).squeeze()
        mean_v = reference_v.mean()
        next_v = (1 - done) * (
            self._target_val_function(next_obs).squeeze() - self._dt * self._gamma ** (1 - self._dt) * mean_v) - \
            done * self._gamma * mean_v / max(1 - self._gamma, 1e-5)

        obs = check_array(obs)
        batch_size = obs.shape[0]
        advs = self.critic(
            np.concatenate([obs, obs], axis=0),
            torch.cat([action, max_action], dim=0))
        adv, max_adv = advs[:batch_size], advs[batch_size:]

        expected_v = (reward * self._dt + self._gamma ** self._dt * next_v).detach()
        bell_residual = (expected_v - v) / self._dt - adv + max_adv

        critic_loss = (bell_residual ** 2) + (max_adv ** 2)

        self._val_optimizer.zero_grad()
        self._adv_optimizer.zero_grad()
        critic_loss.mean().backward(retain_graph=True)
        self._val_optimizer.step()
        self._adv_optimizer.step()

        soft_update(self._adv_function, self._target_adv_function, self._tau)
        soft_update(self._val_function, self._target_val_function, self._tau)

        return critic_loss

    def critic(self, obs: Arrayable, action: Tensorable, target: bool = False) -> Tensor:
        func = self._adv_function if not target else self._target_adv_function
        if len(func.input_shape()) == 2:
            adv = func(obs, action).squeeze()
        else:
            adv_all = func(obs)
            action = check_tensor(action, self._device).long()
            adv = adv_all.gather(1, action.view(-1, 1)).squeeze()
        return adv

    def log(self):
        pass

    def critic_function(self, target: bool = False):
        func = self._adv_function if not target else self._target_adv_funciton
        return func

    def to(self, device):
        CompoundStateful.to(self, device)
        self._device = device
        return self

    @staticmethod
    def configure(**kwargs):
        observation_space = kwargs['observation_space']
        action_space = kwargs['action_space']
        assert isinstance(observation_space, Box)
        nb_state_feats = observation_space.shape[-1]
        net_dict = dict(nb_layers=kwargs['nb_layers'], hidden_size=kwargs['hidden_size'])
        val_function = MLP(nb_inputs=nb_state_feats, nb_outputs=1, **net_dict)
        if isinstance(action_space, Discrete):
            nb_actions = action_space.n
            adv_function = MLP(nb_inputs=nb_state_feats, nb_outputs=nb_actions,
                               **net_dict)
        elif isinstance(action_space, Box):
            nb_actions = action_space.shape[-1]
            adv_function = ContinuousAdvantageMLP(
                nb_outputs=1, nb_state_feats=nb_state_feats, nb_actions=nb_actions,
                **net_dict)
        if kwargs['normalize']:
            val_function = NormalizedMLP(val_function)
            adv_function = NormalizedMLP(adv_function)
        return AdvantageCritic(kwargs['dt'], kwargs['gamma'], kwargs['lr'], kwargs['tau'],
                               kwargs['optimizer'], val_function, adv_function)
