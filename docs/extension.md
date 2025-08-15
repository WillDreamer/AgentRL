# Easy Extension to Your Agents



## Core Logics




## Examples

Let's take Search Agents for example:

- First, we will create a folder `recipe/search_r1` to include the setup of environments, scirpts to train.

- We also need to build the `recipe/search_r1/config` folder to include the hyper-parameters.

- Then, two main files are needed, `recipe/search_r1/main_xxx.py` and `recipe/search_r1/xxx_ray_trainer.py`. We will implememt the wrap up of VeRL framework here.