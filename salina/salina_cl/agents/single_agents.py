#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import numpy as np
import torch
import torch.nn.functional as F
from salina_cl.core import CRLAgent
from torch import nn


class Action(CRLAgent):
    def __init__(self, input_dimension,output_dimension, hidden_size, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = False):
        super().__init__()
        self.start_steps = start_steps
        self.counter = 0
        self.iname = input_name
        self.task_id = 0
        self.output_dimension = output_dimension
        self.hs = hidden_size
        self.input_size = input_dimension
        self.activation = activation
        self.layer_norm = layer_norm
        self.model = nn.ModuleList([self.make_model()])

    def make_model(self):
        if self.layer_norm:
            return nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.output_dimension * 2),
        )       
        else:
            return nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.output_dimension * 2),
        )

    def forward(self, t = None, **kwargs):
        model_id = min(self.task_id,len(self.model) - 1)
        if not self.training:
            input = self.get((self.iname, t))
            mu, _ = self.model[model_id](input).chunk(2, dim=-1)
            action = torch.tanh(mu)
            self.set(("action", t), action)
        elif not (t is None):
            input = self.get((self.iname, t)).detach()
            if self.counter <= self.start_steps:
                action = torch.rand(input.shape[0],self.output_dimension).to(input.device) * 2 - 1
            else:
                mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
                log_std = torch.clip(log_std, min=-20., max=2.)
                std = log_std.exp()
                action = mu + torch.randn(*mu.shape).to(mu.device) * std
                action = torch.tanh(action)
            self.set(("action", t), action)
            self.counter += 1
        else:
            input = self.get(self.iname).detach()
            mu, log_std = self.model[model_id](input).chunk(2, dim=-1)
            log_std = torch.clip(log_std, min=-20., max=2.)
            std = log_std.exp()
            action = mu + torch.randn(*mu.shape).to(mu.device) * std
            log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1, keepdim=True)
            log_prob -= (2 * np.log(2) - action - F.softplus( - 2 * action)).sum(-1, keepdim=True)
            action = torch.tanh(action)
            self.set("action", action)
            self.set("action_logprobs", log_prob)

class MultiAction(Action):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(copy.deepcopy(self.model[-1]))
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id

class FromScratchAction(Action):
    def set_task(self,task_id = None):
        if task_id is None:
            self.model.append(self.make_model())
            self.task_id = len(self.model) - 1
        else:
            self.task_id = task_id


class EWCAction(Action):
    def __init__(self, input_dimension,output_dimension, hidden_size, fisher_coeff, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = False):
        super().__init__(input_dimension,output_dimension, hidden_size, start_steps, input_name, activation, layer_norm)
        self.fisher_coeff = fisher_coeff
        self.regularize = False

    def register_and_consolidate(self,fisher_diagonals):
        param_names = [n.replace('.', '_') for n, p in  self.model.named_parameters()]
        fisher_dict={n: f.detach() for n, f in zip(param_names, fisher_diagonals)}
        for name, param in self.model.named_parameters():
            name = name.replace('.', '_')
            self.model.register_buffer(f"{name}_mean", param.data.clone())
            if self.regularize:
                fisher = getattr(self.model, f"{name}_fisher") + fisher_dict[name].data.clone() ## add to the old fisher coeff
            else:
                fisher =  fisher_dict[name].data.clone()
            self.model.register_buffer(f"{name}_fisher", fisher)
        self.regularize = True
    
    def add_regularizer(self, *args):
        if self.regularize:
            losses = []
            for name, param in self.model.named_parameters():
                name = name.replace('.', '_')
                mean = getattr(self.model, f"{name}_mean")
                fisher = getattr(self.model,f"{name}_fisher")
                losses.append((fisher * (param - mean)**2).sum())
           
          
            return (self.fisher_coeff)*sum(losses).view(1).to(list(self.parameters())[0].device)
        else:
            return torch.Tensor([0.]).to(list(self.parameters())[0].device)

class L2Action(Action):
    def __init__(self, input_dimension,output_dimension, hidden_size, l2_coeff, start_steps = 0, input_name = "env/env_obs", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = False):
        super().__init__(input_dimension,output_dimension, hidden_size, start_steps, input_name, activation, layer_norm)
        self.l2_coeff = l2_coeff
        self.regularize = False

    def register_and_consolidate(self):
        for name, param in self.model.named_parameters():
            name = name.replace('.', '_')
            self.model.register_buffer(f"{name}_mean", param.data.clone())
        self.regularize = True
    
    def add_regularizer(self, *args):
        if self.regularize:
            losses = []
            for name, param in self.model.named_parameters():
                name = name.replace('.', '_')
                mean = getattr(self.model, f"{name}_mean")
                losses.append(((param - mean.detach())**2).sum())
            return (self.l2_coeff)*sum(losses).view(1).to(list(self.parameters())[0].device)
        else:
            return torch.Tensor([0.]).to(list(self.parameters())[0].device)

class Critic(CRLAgent):
    def __init__(self, obs_dimension, action_dimension, hidden_size, input_name = "env/env_obs", output_name = "q", activation = nn.LeakyReLU(negative_slope=0.2), layer_norm = False):
        super().__init__()

        self.iname = input_name 
        self.input_size = obs_dimension + action_dimension
        self.hs = hidden_size
        self.output_name = output_name
        self.layer_norm = layer_norm
        self.activation = activation
        if self.layer_norm:
            self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            nn.LayerNorm(self.hs),
            nn.Tanh(),
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,1),
        )       
        else:
            self.model =  nn.Sequential(
            nn.Linear(self.input_size,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,self.hs),
            self.activation,
            nn.Linear(self.hs,1),
        )

    def forward(self, **kwargs):
        input = self.get(self.iname).detach()
        action = self.get(("action"))
        input = torch.cat([input, action], dim=-1)
        critic = self.model(input).squeeze(-1)
        self.set(self.output_name, critic)