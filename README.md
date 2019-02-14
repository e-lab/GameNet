# GameNet

## Overview:

### 1. Using the Program

### 2. Introduction

### 3. Implementation

### 4. Testing Methodology and Results

### 5. Conclusions

### 6. References

## 1. Using the Program:

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


## 2. Introduction:

This project aims at developing a neural network trained using reinforcement learning to explore across different environments and complete a specific task. The goal is for an agent to autonomously explore an environment to discover a target with efficient exploration. Of particular interest is developing a network capable of exploring with sparse and few external rewards, forcing the network to rely on alterior methods for exploring an environment. Testing and evaluation of this exploration algorithm are implemented on the ViZDoom environment. This project implements the Intrinsic Curiosity Model (ICM) network as described in the paper (https://arxiv.org/pdf/1705.05363.pdf) by Pathak et al. This model gives the agent an intrinsic reward in addition to the extrinsic reward of reaching a given target. These intrinsic rewards are given when the agent visits locations and observes visual features it has not seen before, incentivizing the agent to explore new areas. 

## 3. Implementation:

An image of the ICM network architecture is shown below in Figure 1. This predictive network incentivizes exploration of areas that result in high predictive error. As illustrated in the Figure below, the forward model takes the inputs of the current state features and the agent's action and then predicts the feature representation of the next state. The prediction of the feature representation is then compared to the ground truth, the actual feature representation. Where the prediction error of this comparison is high, the agent will receive a stronger intrinsic reward signal. This mechanism incentivizes exploration of areas with new and different features that have been previously seen. This signal, dubbed the curiosity based intrinsic reward signal, is utilized in junction with traditional external rewards to train an Advantage Actor Critic reinforcement learning network to solve a series of mazes, as described below. For further details on the methodology and implementation of this approach, please see the paper by Pathak et al. mentioned above. 

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%20Network%20Architecture.PNG" height="400" width="750">
Figure 1. Neural Network Architecture for Intrinsic Curiosity
"The agent in state s<sub>t</sub> interacts with the environment by executing an action at sampled from its current policy π and ends up in the state s<sub>t+1</sub>. The policy π is trained to optimize the sum of the extrinsic reward (r <sup>e</sup> <sub>t</sub>) provided by the environment E and the curiosity based intrinsic reward signal (r<sup>i</sup><sub>t</sub>) generated by our proposed Intrinsic Curiosity Module (ICM). ICM encodes the states s<sub>t</sub>, s<sub>t+1</sub> into the features φ(s<sub>t</sub>), φ(s<sub>t+1</sub>) that are trained to predict at (i.e. inverse dynamics model). The forward model takes as inputs φ(s<sub>t</sub>) and at and predicts the feature representation φˆ(s<sub>t+1</sub>) of s<sub>t+1</sub>. The prediction error in the feature space is used as the curiosity based intrinsic reward signal. As there is no incentive for φ(s<sub>t</sub>) to encode any environmental features that can not influence or are not influenced by the agent’s actions, the learned exploration strategy of our agent is robust to uncontrollable aspects of the environment."<sup>1</sup>

## 4. Testing Methodology and Results:

There are 5 scenarios that the algorithm is assessed on. The scenarios are divided in two subgroups: Custom scenarios and “My Way Home” scenarios. Custom scenarios are designed by the team specifically for this task, while “My Way Home” scenarios are available from the ViZDoom framework. The Custom scenarios have the same wall textures across each room, whereas the “My Way Home” scenarios have different wall textures in each room. 

The Custom scenarios are as follow:

•	1-Room Scenario: Agent and target are spawned randomly in the same room in a single-room environment

<img src="https://github.com/e-lab/GameNet/blob/master/images/1%20Room%20Map.jpg" height="200" width="200">

Figure 2. 1 Room Scenario Map

•	2-Room Scenario: Agent and target are spawned randomly in two different rooms in a two-room environment connected with corridors

<img src="https://github.com/e-lab/GameNet/blob/master/images/2%20Room%20Map.jpg" height="200" width="200">

Figure 3. 2 Room Scenario Map

•	3-Room Scenario: Agent and target are spawned randomly in two different rooms in a three-room environment connected with corridors

<img src="https://github.com/e-lab/GameNet/blob/master/images/3%20Room%20Map.png" height="200" width="200">

Figure 4. 3 Room Scenario Map

The “My Way Home” scenarios are as follow:

•	Dense: Agent is spawned randomly in one of 17 spawn locations and must navigate the scenario to reach the target

•	Sparse: More complicated variant of My Way Home Dense, where the agent is spawned at one location far away from the objective

Shown below in Figure 5 is the My Way Home map. For the Dense scenario, the agent can spawn at any of the blue circle indicators. However, in the sparse scenario, the agent will only spawn in the furthest left blue circle.

<img src="https://github.com/e-lab/GameNet/blob/master/images/My%20Way%20Home%20Map.png" height="250" width="250">

Figure 5. My Way Home Dense/Sparse Scenario Map

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

Shown below are plots comparing performance of the Intrinsic Curiosity Model (ICM) in various mazes built in VizDoom, particularly assessing performance in mazes with various textured walls in different rooms versus uniform textures amongst rooms in mazes. In each maze, there is a target for the model, which is controlling a character in the game, to find. A model being able to consistently and repeatedly “solve” the maze is signified if the respective plot line converges to the score of 1. Data plot lines are performance averaged over a minimum of 8 runs of a given network in a given scenario. The line plot indicates the average of the testing score across all of the runs, while the shaded area surrounding the line plot is the standard error associated with the average score. Figure 6 illustrates a comparison of a standard A2C network and the ICM model testing on the 1 room scenario. Both models display roughly the same behavior and training steps needed to reach convergence. 

<img src="https://github.com/e-lab/GameNet/blob/master/images/A2C%2C%20ICM%201%20Room.PNG" height="450" width="750">

Figure 6. Standard A2C vs ICM Model Performance in 1 Room Scenario

[Add ICM plots on 2/3 Rooms]

Figures 7 and 9 illustrate an A2C network testing performance on the 2 room and 3 room environments, both the uniformly textured and varied textured versions. The results indicate that in both of the environments, the network is able to train slightly faster towards convergence when placed in the varied texture version of the mazes over the uniform textured ones. This indicates that the A2C network intuitively trains to leverage variations in the environment that the ICM model attempts to exploit moreso. 

Figures 8 and 10 similarly demonstrate the ICM model's performance in the same respective environments. 

<img src="https://github.com/e-lab/GameNet/blob/master/images/A2C%202%20Rooms.PNG" height="450" width="750">

Figure 7. Standard A2C Model Performance in 2 Room Scenarios

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%202%20Rooms.PNG" height="450" width="750">

Figure 8. ICM Model Performance in 2 Room Scenarios

<img src="https://github.com/e-lab/GameNet/blob/master/images/A2C%203%20Rooms.PNG" height="450" width="750">

Figure 9. Standard A2C Model Performance in 3 Room Scenarios

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%203%20Rooms.PNG" height="450" width="750">

Figure 10. ICM Model Performance in 3 Room Scenarios

Shown below in Table 1 is a summary of the results displayed in Figures 6-10 for the 1, 2, and 3 Room Scenarios. The performance column values are the approximate number of training steps requiring for a given network to converge to the mean test score value of 1 for the given evaluation scenario (lower value is better). The percent change column characterizes the differences in performance of the A2C and ICM networks on the same scenario. The performances highlighted in green indicate the network that converged faster to solve a given scenario more efficiently. In the case of the 1 Room and 2 Room scenarios, the performance of A2C and ICM are rather comparable, while the 3 Room scenarios highlight ICM's better exploratory capacity in a larger maze. Between the varied texture and uniform texture environments, the performances of ICM show that the model excels moreso in environments with high variability. Even the standard A2C model tends to perform better in the varied texture environments relative to the uniform texture ones, indicating the ICM model manages to exploit the tendency of A2C to explore new and different states of environments.

| Scenario Type | Network Type | Performance (training steps) | 
| ------------- | ------------- | ------------- | 
| 1 Room Scenario  | A2C  | ```+2.0E+5```  |      
| 1 Room Scenario  | A2C + ICM  | 3.0E+5  | 
| 2 Room Scenario Uniform Textures | A2C  | ```+1.7E+06```  |    
| 2 Room Scenario Uniform Textures | A2C + ICM  | 1.9E+6  |    
| 2 Room Scenario Varied Textures | A2C  | ```+1.0E+06```  |    
| 2 Room Scenario Varied Textures | A2C + ICM  | 1.2E+6  |  
| 3 Room Scenario Uniform Textures | A2C  | 1.2E+7  |     
| 3 Room Scenario Uniform Textures | A2C + ICM  | ```+8.4E+6```  |     
| 3 Room Scenario Varied Textures | A2C  | 1.0E+7  |   
| 3 Room Scenario Varied Textures | A2C + ICM  | ```+6.6E+6```  |     

Table 1. Summary of A2C and ICM Performance Results

Figures 11 and 12 demonstrate the performance of the ICM model in the My Way Home Dense and Sparse scenarios with varied textures. Both of the scenarios have varied textures between rooms. Between these environments, the ICM model performs rather well, converging slightly faster in the case of the Sparse scenario. 
Figures 13 and 14 examine the ICM model's behavior in the My Way Home Dense and Sparse scenario again but now with uniform textures throughout the environment. As illustrated, the model is unable to repeat its successful convergence as done in the textured variants of the same environments. 

From testing in these My Way Home scenarios, the limitations of the ICM model were demonstrated when handling the case of the sparse scenario with uniform textured rooms. When the environment’s unique features are minimized, the model is unable to generate a substantial intrinsic reward to spur motivation across all rooms in the maze, and the model does not converge in that environment.

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%2C%20My%20Way%20Home%20Dense.PNG" height="450" width="750"> 

Figure 11. ICM Model Performance in My Way Home Dense with Varied Texture

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%2C%20My%20Way%20Home%20Sparse.PNG" height="450" width="750">

Figure 12. ICM Model Performance in My Way Home Sparse with Varied Texture

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%2C%20My%20Way%20Home%20Dense%2C%20Uniform.PNG" height="450" width="750">

Figure 13. ICM Model Performance in My Way Home Dense with Uniform Textures

<img src="https://github.com/e-lab/GameNet/blob/master/images/ICM%2C%20My%20Way%20Home%20Sparse%2C%20Uniform.PNG" height="450" width="750">

Figure 14. ICM Model Performance in My Way Home Sparse with Uniform Textures

## 5. Conclusions:
[Add commentary]

## 6. References:
1. Pathak et al. "Curiosity-driven Exploration by Self-supervised Prediction." 15 May 2017. https://arxiv.org/pdf/1705.05363.pdf
