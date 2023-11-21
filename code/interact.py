"""Agent-env interactions."""
from typing import Tuple
from abstract import Arrayable
from gymnasium.vector import VectorEnv
from agents.agent import Agent
from numpy import ndarray as array
import numpy as np


def interact(
        env: VectorEnv,
        agent: Agent,
        start_obs: Arrayable) -> Tuple[array, array, array]:
    """One step interaction between env and agent.

    :args env: environment
    :args agent: agent
    :args start_obs: initial observation

    :return: (next observation, reward, terminal?)
    """
    action = agent.step(start_obs)
    next_obs, reward, terminated, truncated, information = env.step(action)
    done = np.logical_or(terminated, truncated)
    # time_limit = information['time_limit'] if 'time_limit' in information else None
    agent.observe(next_obs, reward, done, truncated)
    return next_obs, reward, done
