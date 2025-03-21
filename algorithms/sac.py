#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy
import time
import typing as tp

import numpy as np
import salina
import torch
from salina import Workspace, get_arguments, get_class
from salina.agents import Agents, EpisodesDone, TemporalAgent
from salina.rl.replay_buffer import ReplayBuffer

from .tools import _state_dict, compute_time_unit, soft_update_params
from tqdm import tqdm


class SAC:
    """SAC implementation that could handle a regularization method and outputs logs, th eaction agent, the q function and additional data like the replay buffer"""
    def __init__(self,params: dict) -> None:
        self.cfg = params
    
    def run(self, action_agent: salina.Agent, q_agent: salina.Agent, env_agent: salina.Agent,logger, seed: int, n_max_interactions: int, info: dict = {}) -> tp.Tuple[dict, salina.Agent, salina.Agent, dict]:
        logger = logger.get_logger(type(self).__name__+str("/"))
        if n_max_interactions > 0:
            time_unit=None
            cfg=self.cfg
            if cfg.time_limit>0:
                time_unit=compute_time_unit(cfg.device)
                logger.message("Time unit is "+str(time_unit)+" seconds.")
            inner_epochs = int(cfg.inner_epochs * cfg.grad_updates_per_step)
            logger.message("Nb of updates per epoch: "+str(inner_epochs))
        
            action_agent.set_name("action")
            acq_agent = TemporalAgent(Agents(env_agent, copy.deepcopy(action_agent))).to(cfg.acquisition_device)
            acquisition_workspace=Workspace()
            acq_agent.seed(seed)

            control_agent = TemporalAgent(Agents(copy.deepcopy(env_agent), EpisodesDone(), copy.deepcopy(action_agent))).to(cfg.acquisition_device)
            control_agent.seed(seed)
            control_agent.eval()

            # == Setting up the training agents
            action_agent.to(cfg.learning_device)
            q_target_agent = copy.deepcopy(q_agent)
            q_target_agent.to(cfg.learning_device)
            q_agent.to(cfg.learning_device)

            replay_buffer = ReplayBuffer(cfg.buffer_size,device=cfg.buffer_device)
            acq_agent.train()
            action_agent.train()
            logger.message("Initializing replay buffer")
            acq_agent(acquisition_workspace, t=0, n_steps=cfg.n_timesteps)
            n_interactions = acquisition_workspace.time_size() * acquisition_workspace.batch_size()
            replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
            while replay_buffer.size() < cfg.initial_buffer_size:
                acquisition_workspace.copy_n_last_steps(1)
                with torch.no_grad():
                    acq_agent(acquisition_workspace, t=1, n_steps=cfg.n_timesteps - 1)
                replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
                n_interactions += (acquisition_workspace.time_size() - 1) * acquisition_workspace.batch_size()

            # == configuring SAC entropy
            optimizer_args = get_arguments(cfg.optimizer_entropy)
            action_card = np.prod(np.array( acquisition_workspace["action"].size()[2:]))
            target_entropy = - cfg.target_multiplier * action_card
            log_entropy = torch.tensor(np.log(cfg.init_temperature), requires_grad=True, device=cfg.learning_device)
            optimizer_entropy = get_class(cfg.optimizer_entropy)([log_entropy], **optimizer_args)

            optimizer_args = get_arguments(cfg.optimizer_q)
            optimizer_q = get_class(cfg.optimizer_q)(q_agent.parameters(), **optimizer_args)
        
            optimizer_args = get_arguments(cfg.optimizer_policy)
            optimizer_action = get_class(cfg.optimizer_policy)(action_agent.parameters(), **optimizer_args)
            iteration = 0
            epoch=0
            is_training=True
            _training_start_time=time.time()
            best_performance = - float("inf")
            logger.message("Start training")
            with tqdm(total=n_max_interactions, desc="Training Progress", unit="interactions") as pbar:
                while is_training:
                    # Compute average performance of multiple rollouts
                    if epoch%cfg.control_every_n_epochs==0:
                        for a in control_agent.get_by_name("action"):
                            a.load_state_dict(_state_dict(action_agent, cfg.acquisition_device))
                        control_agent.eval()
                        rewards=[]
                        for _ in range(cfg.n_control_rollouts):
                            w=Workspace()
                            control_agent(w, t=0, force_random = True, stop_variable="env/done", logger_render = logger)
                            length=w["env/done"].max(0)[1]
                            arange = torch.arange(length.size()[0], device=length.device)
                            creward = w["env/cumulated_reward"][length, arange]
                            rewards=rewards+creward.to("cpu").tolist()
                            if "env/success" in w.variables:
                                success_rate = w["env/success"][length, arange].mean().item()
                                logger.add_scalar("validation/success_rate", success_rate, epoch)
                            if "env/goalDist" in w.variables:
                                goalDist = w["env/goalDist"][length, arange].mean().item()
                                logger.add_scalar("validation/goalDist", goalDist, epoch)
                        mean_reward=np.mean(rewards)
                        logger.add_scalar("validation/reward", mean_reward, iteration)
                        # print("reward at ",epoch," = ",mean_reward," vs ",best_performance)
                        best_performance = max(mean_reward,best_performance)
                        logger.add_scalar("validation/best_reward", best_performance, iteration)

                    for a in acq_agent.get_by_name("action"):
                        a.load_state_dict(_state_dict(action_agent, cfg.acquisition_device))

                    acquisition_workspace.copy_n_last_steps(1)
                    with torch.no_grad():
                        acq_agent(acquisition_workspace,t=1,n_steps=cfg.n_timesteps - 1)
                    replay_buffer.put(acquisition_workspace, time_size=cfg.buffer_time_size)
                    done, creward = acquisition_workspace["env/done", "env/cumulated_reward"]
            
                    creward = creward[done]
                    if creward.size()[0] > 0:
                        logger.add_scalar("monitor/reward", creward.mean().item(), epoch)
                        if "env/success" in acquisition_workspace.variables:
                            success_rate = acquisition_workspace["env/success"][done].mean().item()
                            logger.add_scalar("monitor/success_rate", success_rate, epoch)
                        if "env/goalDist" in acquisition_workspace.variables:
                            goalDist = acquisition_workspace["env/goalDist"][done].mean().item()
                            logger.add_scalar("monitor/goalDist", goalDist, epoch)
                    logger.add_scalar("monitor/replay_buffer_size", replay_buffer.size(), epoch)
            
                    # 更新交互次数并刷新进度条
                    new_interactions = (acquisition_workspace.time_size() - 1) * acquisition_workspace.batch_size()
                    n_interactions += new_interactions
                    pbar.update(new_interactions)
                    logger.add_scalar("monitor/n_interactions", n_interactions + n_max_interactions * info["task_id"], epoch)
            
                    for inner_epoch in range(inner_epochs):
                        entropy = log_entropy.exp()
                        replay_workspace = replay_buffer.get(cfg.batch_size).to(cfg.learning_device)
                        done, reward = replay_workspace["env/done", "env/reward"]
                        not_done = 1.0 - done.float()
                        reward = reward * cfg.reward_scaling

                        # == q1 and q2 losses
                        q_agent(replay_workspace)
                        q_1 = replay_workspace["q1"]
                        q_2 = replay_workspace["q2"]
                        with torch.no_grad():
                            action_agent(replay_workspace, q_update = True)
                            q_target_agent(replay_workspace, q_update = True)
                            q_target_1 = replay_workspace["q1"]
                            q_target_2 = replay_workspace["q2"]
                            _logp = replay_workspace["action_logprobs"]
                            q_target = torch.min(q_target_1, q_target_2)
                            target = (reward[1:]+ cfg.discount_factor * not_done[1:] * (q_target[1:] - (entropy * _logp[1:]).detach()))
                        td_1 = ((q_1[:-1] - target) ** 2).mean()
                        td_2 = ((q_2[:-1] - target) ** 2).mean()
                        optimizer_q.zero_grad()
                        loss = td_1 + td_2
                        logger.add_scalar("loss/td_loss_1",td_1.item(),iteration)
                        logger.add_scalar("loss/td_loss_2",td_2.item(),iteration)
                        loss.backward()
                        if cfg.clip_grad > 0:
                            n = torch.nn.utils.clip_grad_norm_(q_agent.parameters(), cfg.clip_grad)
                            logger.add_scalar("monitor/grad_norm_q", n.item(), iteration)
                        optimizer_q.step()
                        
                        # == Actor and entropy losses
                        if iteration % cfg.policy_update_delay == 0:
                            action_agent(replay_workspace, policy_update = True)
                            q_agent(replay_workspace, policy_update = True)
                            logp = replay_workspace["action_logprobs"]
                            q1 = replay_workspace["q1"]
                            q2 = replay_workspace["q2"]
                            qloss = torch.min(q1,q2).mean()
                            entropy_loss = (entropy.detach() * logp).mean()
                            loss_regularizer = action_agent.add_regularizer(replay_workspace,n_interactions,inner_epoch)
                            optimizer_action.zero_grad()
                            loss = - qloss + entropy_loss + loss_regularizer
                            loss.backward()
                            if cfg.clip_grad > 0:
                                n = torch.nn.utils.clip_grad_norm_(action_agent.parameters(), cfg.clip_grad)
                                logger.add_scalar("monitor/grad_norm_action", n.item(), iteration)
                            logger.add_scalar("loss/q_loss", qloss.item(), iteration)
                            logger.add_scalar("loss/regularizer", loss_regularizer.item(), iteration)
                            optimizer_action.step()

                            optimizer_entropy.zero_grad()
                            entropy_loss = - (log_entropy.exp() * (logp + target_entropy).detach()).mean()
                            entropy_loss.backward()
                            if cfg.clip_grad > 0:
                                n = torch.nn.utils.clip_grad_norm_(log_entropy, cfg.clip_grad)
                                logger.add_scalar("monitor/grad_norm_entropy", n.item(), iteration)
                            optimizer_entropy.step()
                            logger.add_scalar("loss/entropy_loss", entropy_loss.item(), iteration)
                            logger.add_scalar("loss/entropy_value", entropy.item(), iteration)
            
                        # == Target network update
                        if iteration % cfg.target_update_delay == 0:
                            tau = cfg.update_target_tau
                            soft_update_params(q_agent[0], q_target_agent[0], tau)
                            soft_update_params(q_agent[1], q_target_agent[1], tau)


                        iteration += 1
                    epoch += 1
                    if n_interactions > n_max_interactions:
                        logger.message("== Maximum interactions reached")
                        is_training = False
                    else:
                        if cfg.time_limit>0:
                            is_training = time.time() - _training_start_time < cfg.time_limit*time_unit

            r = {"n_epochs":epoch, "training_time":time.time() - _training_start_time, "n_interactions":n_interactions}
            info["replay_buffer"] = replay_buffer
            return r, action_agent.to("cpu"), q_agent.to("cpu"), info

        else:
            logger.message("Out of budget. no training.")
            r = {"n_epochs":0, "training_time":0, "n_interactions":0}
            info = {}
            return r, action_agent.to("cpu"), q_agent.to("cpu"), info