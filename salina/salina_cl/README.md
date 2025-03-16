# SaLinA Continual Learning framekwork (salina_cl)
SaLinA Continual Reinforcement Learning framework with multiple scenarios and methods already implemented. It is also the codebase of the paper [Building a subspace of Policies for scalable Continual Learning](https://arxiv.org/abs/2110.07910). 

![Alt Text](assets/forgetting.gif)
![Alt Text](assets/transfer.gif)
![Alt Text](assets/distraction.gif)
![Alt Text](assets/composability.gif)

## Requirements
Make sure you have [pytorch installed with cuda>11.0](https://pytorch.org/) and followed [instructions to install SaLinA](https://github.com/facebookresearch/salina#quick-start). In addition:
* additional packages: `pip install wandb ternary hydra`
* In order to install brax, we recommend to run the following installations:
  * jax+jaxlib: `pip install --upgrade "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
  * flax: `pip install flax==0.3.4`
  * brax: `pip install brax`
  * brax: `pip install brax==0.1` # pip install jax==0.4.20 jaxlib==0.4.20 flax==0.7.5 brax==0.9.3
  * /!\ we recommend you to use cuda/11.4 version (check it with `nvcc --version`).

## Get started
Simply run the file `run.py` with the desired config available in [configs](salina/salina_cl/configs/). You can select one of them with the flag `-cn=my_config`. Different scenarios are available in [configs/scenario](salina/salina_cl/configs/scenario/). Simply add `scenario=my_scenario` as an argument. For example if you want to run the CSP method on the forgetting scenario of halfcheetah:
 ```console
python run.py -cn=csp scenario=halfcheetah/forgetting
```

## Organization of salina_cl

The `core.py` file contains the building blocks of this framework. Each experiment consists in running a `Framework` over a `Scenario`, i.e. a sequence of train and test `Task`. The models are learning procedures that use salina agents to interact with the tasks and learn from them through one or multiple algorithms.

* [frameworks](frameworks/) contains generic learning procedures (e.g. using only one algorithm, or adding a regularization method in the end)
* [scenarios](scenarios/) contains CRL scenarios i.e sequence of train and test tasks
* [algorithms](algorithms/) contains different RL / CL algorithms (ppo, sac, td3, ewc regularization)
* [agents](agents/) contains salina agents (policy, critic, ...)
* [configs](configs/) contains the configs files of benchmarked methods/scenarios.


## Available methods

We implmented 8 different methods all built on top of soft-actor critic algorithm. To try them, just add the flag `-cn=my_method` on the command line. You can find the hps in [configs](configs):

* `csp`: Continual Subspace of Policies from [Building a subspace of Policies for scalable Continual Learning](https://arxiv.org/abs/2110.07910)

* `ft_1`: Fine-tune a single policy during the whole training
* `sac_n`: Fine-tune and save the policy at the end of the task. Start wit  a randomized policy when encountering a new task.
* `ft_n`: Fine-tune and save the policy at the end of the task. Clone the last policy when encountering a new task.
* `ft_l2`: Fine-tune a single policy during the whole training with a regularization cost (a simpler EWC method)
* `ewc`: see the paper [Overcoming catastrophic forgetting in neural](https://arxiv.org/pdf/1612.00796.pdf)
* `packnet`: see the paper [PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning](https://arxiv.org/pdf/1711.05769.pdf)
* `pnn`: see the paper [Progressive Neural Networks](https://arxiv.org/pdf/1606.04671.pdf)

## Available scenarios

We propose 9 scenarios over 3 different [brax](https://github.com/google/brax) domains. To try them, just add the flag `scenario=...` on the command line:

* Halfcheetah:
    * `halfcheetah/forgetting`: 8 tasks - 1M samples for each task
    * `halfcheetah/transfer`: 8 tasks - 1M samples for each task
    * `halfcheetah/distraction`: 8 tasks - 1M samples for each task
    * `halfcheetah/composability`: 8 tasks - 1M samples for each task

* Ant:
    * `ant/forgetting`: 8 tasks - 1M samples for each task
    * `ant/transfer`: 8 tasks - 1M samples for each task
    * `ant/distraction`: 8 tasks - 1M samples for each task
    * `ant/composability`: 8 tasks - 1M samples for each task

* Humanoid:
    * `humanoid/hard`: 4 tasks - 2M samples for each task

## Citing `salina_cl`

Please use this bibtex if you want to cite this repository in your publications:

```
@misc{https://doi.org/10.48550/arxiv.2211.10445,
  doi = {10.48550/ARXIV.2211.10445},
  url = {https://arxiv.org/abs/2211.10445},
  author = {Gaya, Jean-Baptiste and Doan, Thang and Caccia, Lucas and Soulier, Laure and Denoyer, Ludovic and Raileanu, Roberta},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Building a Subspace of Policies for Scalable Continual Learning},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```