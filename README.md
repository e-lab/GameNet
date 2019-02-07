# GameNet

To start training run:

`python a2c.py`

Arguments:

`--icm` train with curiosity

`--scenario` specify the environment you want to train on

`--save_dir` specify where your model and testing results are going to be saved

For example to train a standard a2c on My Way Home scenario:

`python a2c.py`

To train a2c with curiosity on My Way Home Sparse scenario:

`python a2c.py --icm --scenario=./scenarios/my_way_home_sparse.cfg`


Introduction

This project aims at developing a neural network trained using reinforcement learning to explore across different environments and complete a specific task. The goal is for an agent to autonomously explore an environment to discover a target with efficient exploration. Testing and evaluation of this exploration algorithm are implemented on the ViZDoom environment. This project implements the Intrinsic Curiosity Model (ICM) network as described in the paper (https://arxiv.org/pdf/1705.05363.pdf) by Pathak et. al. This model gives the agent an intrinsic reward in addition to the extrinsic reward of reaching a given target. These intrinsic rewards are given when the agent visits locations and observes visual features it has not seen before, incentivizing the agent to explore new areas. 

Implementation

An image of the ICM network architecture is shown below in Figure 1.

![alt text](https://github.com/e-lab/GameNet/blob/master/images/ICM%20Network%20Architecture.PNG)
Figure 1. Neural Network Architecture for Intrinsic Curiosity

Testing Methodology and Results

There are 5 scenarios that the algorithm is assessed on. The scenarios are divided in two subgroups: Custom scenarios and “My Way Home” scenarios. Custom scenarios are designed by the team specifically for this task, while “My Way Home” scenarios are available from the ViZDoom framework. The Custom scenarios have the same wall textures across each room, whereas the “My Way Home” scenarios have different wall textures in each room. 

The Custom scenarios are as follow:

•	1-Room Scenario: Agent and target are spawned randomly in the same room in a single-room environment

•	2-Room Scenario: Agent and target are spawned randomly in two different rooms in a two-room environment connected with corridors

•	3-Room Scenario: Agent and target are spawned randomly in two different rooms in a three-room environment connected with corridors

The “My Way Home” scenarios are as follow:

•	Dense: Agent is spawned randomly in one of 17 spawn locations and must navigate the scenario to reach the target

•	Sparse: More complicated variant of My Way Home Dense, where the agent is spawned at one location far away from the objective

Shown below are videos of an agent, controlled by a neural network, exploring various mazes. In the first two videos, the agent is exploring the 2-Room Scenario. However, in the first case, the textures of the walls differ and present different features to provide a higher intrinsic reward to the agent. In the second, the textures of the walls are all the same and bare no differences visually or in terms of features to generate a significant intrinsic reward value.

Video Gameplay from 2 Room Scenario, Varied Textures:

[![Watch the video](https://img.youtube.com/vi/ZM_DA8amomU/hqdefault.jpg)](
https://youtu.be/ZM_DA8amomU)

Video Gameplay from 2 Room Scenario, Uniform Textures:

[![Watch the video](https://img.youtube.com/vi/gdGoK9cx1R4/hqdefault.jpg)](
https://youtu.be/gdGoK9cx1R4)

In the next two videos, the agent is attempting to solve the My Way Home Dense and My Way Home Sparse scenarios. In the My Way Home Sparse environment, the textures of the walls differ from room to room to better incentivize the agent to explore the environment. As demonstrated in the results plots, the agent fails to reach the external reward target in the My Way Home Sparse scenario with entirely uniform wall textures between the rooms. The second video is thus of the My Way Home Dense scenario with uniform wall textures throughout the entire environment. The agent is able to successfully navigate and find the external reward in this case. 

Video Gameplay from My Way Home Sparse, Varied Textures:

[![Watch the video](https://img.youtube.com/vi/b1hOzzZO2ag/hqdefault.jpg)](
https://youtu.be/b1hOzzZO2ag)

Video Gameplay from My Way Home Dense, Uniform Textures:

[![Watch the video](https://img.youtube.com/vi/FO8I7g8z_Jw/hqdefault.jpg)](
https://youtu.be/FO8I7g8z_Jw)


Conclusions
