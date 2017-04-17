# Prednet for self-driving
## Content

#### media

-Sample prednet image and video

#### m_1(Modified prednet sending R1 to sensor reading classifier)

#### m_3(Modified prednet sending R1,R2,R3 to sensor reading classifier)

#### o_1(Original prednet sending R1 to sensor reading classifier)

#### o_3(Original prednet sending R1,R2,R3 to sensor reading classifier)

python ___.py to run the corresponding code.(python 2.7)

## January

+ Set up environment for data collecting

## Feburary

+ Set up environment for reinforcement learning

+ Finish  tcp server communication script to get feedback from environment

+ Collect Dataset (60k images and sets of sensor readings)

+ Finish basic structure for reinforcement learning (waiting for embedding of prednet) 

## March

+ Finish LSTM module

+ Finish Prednet class

+ Finish class for image transformation and batch getting

+ Add head for sensor output to prednet

+ Modify Prednet (Feed prediction A_hat to subsequent layers instead of error)

## Apirl

+ Start training (length = 4, batch = 64)

+ Normalize sensor reading

+ Finish training for original and modified prednet with (20k batches)

* input
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/media/input.png)

* prediction
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/media/preview.png)

+ Problem identified: Sensor fitting is not ideal.



