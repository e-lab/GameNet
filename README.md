# GameNet

Introduction

This project aims at developing a neural network trained using reinforcement learning to explore across different environments and complete a specific task. The goal is for an agent to autonomously explore an environment to discover a target with efficient exploration. Testing and evaluation of this exploration algorithm are implemented on the ViZDoom environment. This project implements the Intrinsic Curiosity Model (ICM) network as described in the paper (https://arxiv.org/pdf/1705.05363.pdf) by Pathak et. al. This model gives the agent an intrinsic reward in addition to the extrinsic reward of reaching a given target. These intrinsic rewards are given when the agent visits locations and observes visual features it has not seen before, incentivizing the agent to explore new areas. 

Implementation

An image of the ICM network architecture is shown below in Figure 1.

![alt text](https://github.com/e-lab/GameNet/blob/master/images/ICM%20Network%20Architecture.PNG)


Testing Methodology and Results

There are 5 scenarios that the algorithm is assessed on. The scenarios are divided in two subgroups: Custom scenarios and “My Way Home” scenarios. Custom scenarios are designed by the team specifically for this task, while “My Way Home” scenarios are available from the ViZDoom framework. The Custom scenarios have the same wall textures across each room, whereas the “My Way Home” scenarios have different wall textures in each room. 

The Custom scenarios are as follow:

•	1-Room Scenario: Agent and target are spawned randomly in the same room in a single-room environment

•	2-Room Scenario: Agent and target are spawned randomly in two different rooms in a two-room environment connected with corridors

•	3-Room Scenario: Agent and target are spawned randomly in two different rooms in a three-room environment connected with corridors

The “My Way Home” scenarios are as follow:

•	Dense: Agent is spawned randomly in one of 17 spawn locations and must navigate the scenario to reach the target

•	Sparse: More complicated variant of My Way Home Dense, where the agent is spawned at one location far away from the objective



Conclusions
