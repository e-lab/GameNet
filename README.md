# GameNet

This project aims at developing neural network for end-to-end training, i.e. the network will get whole image as input and the output will be desired sensor values (acceleration, speed, direction, etc).
Since it is difficult to obtain dataset with these sensor values from real vehicle, and it is even more difficult to run multiple tests on actual road; we will be using games with high graphics quality to do analyze our network performace.

There are multiple goals of this project:

1. Train a network in surpvised manner and show that scene parsing is NOT the only solution for self-driving vehicles.
2. Use unsupervised technique to build a representation and then show that less data is sufficient enough for supervised training of network after this point.
3. Build our own metric to show that after certain point GFlops pose more significance than accuracy (IoU/iIoU).

List of games that we will be using in this project are:
+ [Grand Theft Auto V](http://www.rockstargames.com/V/)
+ games preferably with high graphics quality and driving assistance feature

Project members are: Abhishek Chaurasia, Eugenio Culurciello, Karan Dwivedi, and Xinrui Wang.

First use [utility](utils) folder to interface your system with any game and then prepare dataset compatible for GameNet.
Next, go to [train](train) folder to train your  neural network.

