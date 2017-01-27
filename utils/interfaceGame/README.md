# GTA-V Dataset & Environment Setup
## Dataset Mirrors(600K images, 42 hours of driving time)
[README](https://docs.google.com/document/d/1c3upen7NsvkAwgxTC9rhAZJr75Zkn3EP2FVf1Ailqww/edit)

[Archive.org](https://archive.org/details/deepdrive-baseline-uint8)

[dropbox](https://www.dropbox.com/s/b6ox0k1w42cn3ca/gtav-42-hours-uint8.tar.gz?dl=0)

[googledrive](https://drive.google.com/drive/folders/0B2UgaM91sqeAWGZVaDdmaGs2cmM)

File structure
```
HDF5 "train_0000.h5" {
GROUP "/" {
   DATASET "images" {
      DATATYPE  H5T_STD_U32LE
      DATASPACE  SIMPLE { ( 999, 3, 227, 227 ) / ( 999, 3, 227, 227 ) }
   }
   DATASET "targets" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 999, 6 ) / ( 999, 6 ) }
   }
   DATASET "vehicle_states" {
      DATATYPE  H5T_IEEE_F32LE
      DATASPACE  SIMPLE { ( 999, 6 ) / ( 999, 6 ) }
   }
}
}
```
Getting image of first frame:
```
def show_image_read_from_hdf5_file(img):
    from skimage.viewer import ImageViewer
    img = np.array(img, dtype='u1')
    img = img.transpose((1, 2, 0))
    img = img[:, :, ::-1]
    viewer = ImageViewer(img.reshape(SIZE, SIZE, 3))
    viewer.show()
```
![](https://github.com/wang3303/GTA-V/blob/master/sample.png)

Getting targets and states of first frame
```
#targets
array([-0.230168  , -1.      ,  0.57265699,  0.161715  , -0.052632  ,  
        0.40000001], dtype=float32)
#frames
array([-0.131395  , -1.        ,  0.38608   ,  0.024159  , -0.052632  ,
        0.46013099], dtype=float32)
#Metadata (non-pixel data) in the “targets” hdf5 dataset:
#Spin (yaw velocity)
#Direction (1 for left spin, -1 for right spin)
#Speed (velocity in the forward direction of the car, negative velocity represents going in reverse)
#Speed change (change in the above value)
#Steering (steering angle from 1 to -1, right to left, zero is straight ahead)
#Throttle (gas pedal position from 0 to 1)
```

## Preview dataset [code](https://github.com/wang3303/GTA-V/blob/master/preview_dataset.py)

Dataset link [here](https://drive.google.com/drive/folders/0B2UgaM91sqeAWGZVaDdmaGs2cmM)

The image interval is set to 0.5s.

![](https://github.com/wang3303/GTA-V/blob/master/preview.png)

## Enviroment Setup
### 1. Purchase [GTA-V](http://www.rockstargames.com/V/)
### 2. Game Setup
* Update GTAV to the last version 

* Copy-paste the contents of *bin/Release* under your GTAV installation directory 

* Replace your saved game data in *Documents/Rockstar Games/GTA V/Profiles/* with the contents contents of *bin/SaveGame*. 

* Download *[paths.xml](https://drive.google.com/open?id=0Bzh5djJlCOmMOTA1RVlOXzZ5dEk)* and store it also in the GTAV installation directory. 
### 3. Game Configuration
* Configure your setup in the *config.ini* file
* **mode:** 0 for dataset generation, 1 for reinforcement learning [integer]
* **imageWidth:** Width in pixels of the images generated by the plugin [integer]
* **imageHeight:** Height in pixels of the images generated by the plugin [integer]
* **car:** Vehicle to use. For now only blista (Toyota Prius) is supported (0) [integer]
* **weatherChangeDelay:** Time in seconds between weather changes [integer]
* **initialWeather:** Initial weather type [integer]
* **initialHour:** Initial in-game hour [integer]
* **initialMinute:** Initial in-game minute [integer]
* **initialPosX:** Initial x position of the vehicle (will be redefined to closest road) [integer]
* **initialPosY:** Initial y position of the vehicle (will be redefined to closest road) [integer]
* **maxDuration:** Duration in hours of the session [integer]

See *Scenario.cpp* for more details on the supported values, especially for car, drivingStyle and initialWeather.
#### 3.1. Dataset generation description
The in-game screenshots are stored as RGB PNG format, with the specified width and length of the *config.ini* file. These images are named in order of capture from 1 to undefined. Alongside the images, a file named *dataset.txt* contains a row for each image name, with the labels associated to it separated by spaces. The labels are the following, in the same order:

* Speed (m/s)
* Acceleration (m/s2)
* Brake pedal position (0 to 1)
* Steering angle (-1 to 1, left to right)
* Throttle pedal position (-1 to 1, negative is reverse)
* Yaw rate (deg/s)
* Direction (-1 to left, 1 to right)
* **captureFreq:** Frequency of image and labels capture in Hz (FPS) [integer]
* **datasetDir:** Absolute path with trailing slash to directory to save the dataset (must exist) [string]
* **setSpeed:** Set speed of the vehicle in m/s during dataset generation [float]
* **drivingStyle:** Driving style of the vehicle driver during dataset generation, -1 is manual mode [integer]

#### 3.2. Reinforcement learning description
* **reward:** Selection of reward function [integer]
* **desiredSpeed:** For rewards that require so, the set speed the vehicle should have (m/s) [float]
* **desiredAgressivity:** For rewards that require so, a float between 0 and 1 that regulates the behavior of the reward function [float]
* **host:** IP to connect to [string]
* **port:** Port to connect to [integer]

3 type of reward functions can be chosen, all of them penalize collisions:
* **Reach speed (0):** Reach and keep a desired speed.
* **Stay in lane (1):** The more centered the better, if the vehicle goes against traffic or gets out of the road the reward will be negative.
* **General (2):** It is a combination of the other two, the vehicle will try to stay in lane while maintaining a certain speed, the agressivity configuration option sets how the two partial rewards are averaged. The more agressivity the more reward will be given for keeping the speed and less to stay in lane (i.e. tendence to overtake other vehicles)

### 4. Server Setup
[Sever code](https://github.com/e-lab/GameNet/blob/wang3303-patch-1/utils/interfaceGame/drive.py)

This code is for runnning forward.

### 5. Change the vehicle
Lines 33 and 34 of Scenario.cpp tell the script what vehicle to load. http://grandtheftauto.net/gta5/vehicles