# Prednet for self-driving
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
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/input.png)

* prediction
![](https://github.com/e-lab/GameNet/blob/master/train/prednet_pytorch/preview.png)

+ Problem identified: Sensor fitting is not ideal.



