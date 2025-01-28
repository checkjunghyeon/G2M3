#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

import numpy as np
import cv2
import math
from typing import Optional, Type, Tuple

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="NavRLEnv")
class NavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE

        self._previous_measure = None
        self._previous_coverage = None
        self._previous_accuracy = None
        self._previous_action = None

        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[self._reward_measure_name]
        self._previous_coverage = 0
        return observations

    # def step 2nd (NavRLEnv)
    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)  # RLEnv's step()

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_seen_area(self, grid_map):
        seen_area = np.sum(grid_map > 0)
        return seen_area

    # CHOI: Functions to calculate actual rewards
    def get_reward(self, observations, **kwargs):
        # time-penalty reward
        reward = self._rl_config.SLACK_REWARD

        # progress reward
        current_measure = self._env.get_metrics()[self._reward_measure_name]  # distance_to_currgoal

        if self._episode_subsuccess():
            current_measure = self._env.task.foundDistance

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        # Calculate coverage reward
        grid_map = (self._env.currMap * self._env.expose)[:, :, 0]  # occupancy gt map
        current_coverage = self.get_seen_area(grid_map)

        reward += ((current_coverage - self._previous_coverage) * 0.002)
        # print(f"Coverage Reward: {current_coverage} - {self._previous_coverage} = {(current_coverage - self._previous_coverage) * 0.002}")
        self._previous_coverage = current_coverage

        if self._episode_subsuccess():
            self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        elif self._episode_subsuccess():
            reward += self._rl_config.SUBSUCCESS_REWARD
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
