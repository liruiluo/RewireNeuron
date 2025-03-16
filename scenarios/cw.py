#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# import gym
import gymnasium as gym
import random
from continualworld.envs import MT50, get_subtasks
from gymnasium.wrappers import TimeLimit
from continualworld.utils.wrappers import RandomizationWrapper
from core import Scenario, Task
import wandb

META_WORLD_TIME_HORIZON = 200


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self,e):
        super().__init__(e)
        # self.env=e

    def reset(self,seed=None,options=None):
        return {"env_obs":self.env.reset(seed=seed),"success":0.0}

    def step(self,a):
        observation, reward, terminated, truncated, info=self.env.step(a)
        observation={"env_obs":observation,"success":info["success"]} # ,"goalDist":info["goalDist"]
        return observation, reward, terminated, truncated, info


def make_cw_env(name,seed = 0, render = False):
    print("Building environment ", name)
    random.seed(seed)
    if render:
        env = MT50.train_classes[name](render_mode="rgb_array")
    else:
        env = MT50.train_classes[name]()
    if render:
        env.render_mode = "rgb_array"
        print('render')
    env = RandomizationWrapper(env, get_subtasks(name), kind = "random_init_all")
    if render:
        env = gym.wrappers.RecordVideo(env, f"{wandb.run.dir}/videos/{name}", step_trigger=lambda x: x % 200000 == 0)
    env = MetaWorldWrapper(env)
    env = TimeLimit(env,max_episode_steps = META_WORLD_TIME_HORIZON)
    return env


class CWScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,domain,tasks, repeat_scenario, **kwargs):
        super().__init__()
        tasks = list(tasks) * repeat_scenario
        print("Domain:",domain)
        print("Scenario:",tasks)
        print("Sequence:",tasks)
        for k,task in enumerate(tasks):
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_cw_env,
                "make_env_args":{"name":task,"render":True},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(Task(agent_cfg,k,n_steps))
            test_cfg = {
                "classname": "salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn": make_cw_env,
                "make_env_args": {"name": task},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(test_cfg,k,n_steps))
