# E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator

## [Paper](resource/E2E_APA_IV24_final.pdf)  [Slides](resource/E2E_APA_IV24_Slides_final.pdf)

<img src="resource/front_video_final.gif">
<img src="resource/detail_video_2_final.gif">

This repository contains the code for the IEEE Intelligent Vehicles Symposium (IV 2024) paper 
[E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator](resource/E2E_APA_IV24_final.pdf).

This work presents an end-to-end neural network
to handle parking tasks. The inputs are the images captured by
surrounding cameras, while the outputs are control signals. 
The network learns
how to control the vehicle by imitating experienced drivers.

CARLA simulator is utilized for training data generation and closed-loop evaluation.

## Contents

1. [Setup](#setup)
2. [Dataset and Training](#dataset-and-training)
3. [Evaluation](#evaluation)
4. [Bibtex](#bibtex)


## Setup

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```Shell
git clone git@github.com:qintonguav/e2e-vehicle.git
cd e2e-vehicle/
conda env create -f environment.yml
conda activate E2EParking
chmod +x setup_carla.sh
./setup_carla.sh
```
CUDA 11.7 is used as default. We also validate the compatibility of CUDA 10.2 and 11.3.


Install CARLA python library to the conda env:
```Shell
cd carla/PythonAPI/carla/dist/
unzip carla-0.9.11-py3.7-linux-x86_64.egg -d carla-0.9.11-py3.7-linux-x86_64
cd carla-0.9.11-py3.7-linux-x86_64/
echo "install_carla"

cat>setup.py<<EOF
from distutils.core import setup
setup(name='carla', version='0.9.11', py_modules=['carla'],)
EOF

cd ..
pip install -e carla-0.9.11-py3.7-linux-x86_64
cd ../../../..
```


## Dataset and Training
Our dataset is generated in an open parking lot in map Town_04.
In total, we gathered 192 routes of parking data, counting to around 38k frames.
You can download the dataset (50GB) by running:

```Shell
chmod +x download_data.sh
./download_data.sh
```

The dataset is structured as follows:
```
- e2e_parking
    - Training_data
        - Routes
            - Tasks
                - rgb_front: front camera images
                - rgb_left: left camera images
                - rgb_right: right camera images
                - rgb_rear: rear camera images
                - depth_front: corresponding front depth images
                - depth_left: corresponding left depth images
                - depth_right: corresponding right depth images
                - depth_rear: corresponding rear depth images
                - measurements: ego-vehicle position, motion and control data
                - parking_goal: target parking goal positon
                - topdown: topdown segmentation maps
    - Evaluation_data
```

### Data generation
In addition to the dataset itself, we have provided the tools for manual parking data generation. 
The first step is to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl
```

In a separate terminal, use the script below for generating training data:
```Shell
python3 carla_data_gen.py
```

The main variables to set for this script:
```
--save_path         -> path to save sensor data (default: ./e2e_parking/)
--task_num          -> number of parking task (default: 16)
--shuffle_veh       -> shuffle static vehicles between tasks (default: True)
--shuffle_weather   -> shuffle weather between tasks (default: False)
--random_seed       -> random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)
```

Keyboard Control:
```
w/a/s/d:    throttle/left_steer/right_steer/hand_brake
space:      brake
q:          reverse gear
BackSpace:  reset current task
TAB:        switch camera view
```

Conditions for a successful parking:
```
position: vehicle center to slot center < 0.5 meter
orientation: rotation error < 0.5 degree
duration: satisfy above two conditions for 60 frames
```
Target parking slot is marked with a red 'T'. 

Automatically switch to next task when current one is completed.

Any collisions will reset the task.

### Training script

The code for training is provided in [pl_train.py](./pl_train.py) \
A minimal example of running the training script on a single GPU:
```Shell
python pl_train.py 
```
To configure the training parameters, please refer to [training.yaml](./config/training.yaml), including training data path, number of epoch and checkpoint path.

For parallel training, modify the settings in [pl_train.py](./pl_train.py).

For instance, 8 GPU parallel training:
```
line 14: os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
line 42: num_gpus = 8
```

## Evaluation
Similar to data generation, the first step is also to launch a CARLA server:

```Shell
./carla/CarlaUE4.sh -opengl
```

In a separate terminal, use the script below for trained model evaluation:
```Shell
python3 carla_parking_eva.py
```

The main variables to set for this script:
```
--model_path        -> path to model.ckpt
--eva_epochs        -> number of eva epochs (default: 4')
--eva_task_nums     -> number of evaluation task (default: 16')
--eva_parking_nums  -> number of parking nums for every slot (default: 6')
--eva_result_path   -> path to save evaluation result csv file
--shuffle_veh       -> shuffle static vehicles between tasks (default: True)
--shuffle_weather   -> shuffle weather between tasks (default: False)
--random_seed       -> random seed to initialize env (default: 0)
```
When evaluation is completed, metrics will be saved to csv files located at '--eva_result_path'.

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@inproceedings{E2EAPA,
	title={E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator},
	author={Yang, Yunfan and Chen, Denglong and Qin, Tong and Mu, Xiangru and Xu, Chunjing and Yang, Ming},
	booktitle={Conference on IEEE Intelligent Vehicles Symposium},
	year={2024}
}
```
