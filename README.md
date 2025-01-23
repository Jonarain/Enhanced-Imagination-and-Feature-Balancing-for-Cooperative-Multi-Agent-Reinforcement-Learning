# Enhanced Imagination and Feature Balancing for Cooperative Multi-Agent Reinforcement Learning
  
This repository contains the implementations of our proposed method **Enhanced Imagination and Feature Balancing (EIFB)** which is published in The International Conference on Neural Information Processing (ICONIP) 2024. 

The proposed method aims to improve centralized training of cooperative multi-agent reinforcement learning (MARL) in partially observable environments with: 
1. Enhanced imagination rollouts, which improve upon [Model-Based Value Decomposition (MBVD)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html), serve as one of the two sources of supplementary information, alongside the global state, during centralized training.
2. The feature balancing to dynamically calculate the weights of global state and imagination rollouts over training process.

## Related studies:

1. QMIX ([Monotonic Value Function Factorisation for Deep
 Multi-Agent Reinforcement Learning](https://www.jmlr.org/papers/v21/20-081.html))
2. QPLEX ([QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://openreview.net/forum?id=Rcmk0xxIQV))

    as the base MARL methods.

3. MBVD ([Mingling Foresight with Imagination: Model-Based
 Cooperative Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html))
---
## Implementation:
This codebase is built upon the [PyMARL](https://github.com/oxwhirl/pymarl) framework, along with open-source codebases from related studies, as listed below:
### QMIX-based comparison
1. QMIX ([released codebase](https://github.com/oxwhirl/pymarl))
2. MBVD+QMIX ([released codebase](https://proceedings.neurips.cc/paper_files/paper/2022/hash/49be51578b507f37cd8b5fad379af183-Abstract-Conference.html))
3. EIFB+QMIX

### QPLEX-based comparison
1. QPLEX ([released codebase](https://github.com/wjh720/QPLEX))
2. MBVD+QPLEX
3. EIFB+QPLEX

### Environments
1. Predator-Prey Game (the version from [Deep Coordination Graphs](https://github.com/wendelinboehmer/dcg/blob/master/src/envs/stag_hunt.py))
2. [The StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac), StarCraft 2: ver 2.4.6 (**Performance is not comparable between versions**)
   - 3s vs 5z (Hard)
   - 5m vs 6m (Hard)
   - 6h vs 8z (Super-Hard)
   - MMM2 (Super-Hard)
   - 3s5z vs 3s6z (Super-Hard)
   - Corridor (Super-Hard)

---
## Installation instructions
1. Build the Dockerfile    
```shell 
cd docker 
bash build.sh 
``` 
2. Set up StarCraft II:    
```shell 
bash install_sc2.sh
 ``` 
This will download SC2 into the 3rdparty folder.

## Run an experiment using the Docker container
 - Run QMIX in MMM2
```shell
bash run.sh 0 python3 src/main.py --config=qmix --env-config=sc2_MMM2_baseline with seed=421875111
```

 - Run MBVD+QMIX in MMM2
```shell
bash run.sh 0 python3 src/main.py --config=mbvd --env-config=sc2_MMM2_baseline with seed=421875111
```

 - Run EIFB+QMIX in MMM2 
```shell
bash run.sh 0 python3 src/main.py --config=oracle_mbvd --env-config=sc2_MMM2_baseline with seed=421875111 rollout_depth=6
```

  
## License  
  
Code licensed under the Apache License v2.0
