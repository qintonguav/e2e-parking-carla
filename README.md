# E2E Parking: Autonomous Parking by the End-to-end Neural Network on the CARLA Simulator

## [Paper](resource/E2E_APA_IV24_final.pdf)  [Slides](resource/E2E_APA_IV24_Slides_final.pdf)

<img src="resource/front_video_final.gif" width="600">
<img src="resource/detail_video_2_final.gif" width="600">

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
2. [Evaluation](#Evaluation (Inference with pre-trained model))
3. [Dataset and Training](#dataset-and-training)
4. [Bibtex](#bibtex)


## Setup

Clone the repo, setup CARLA 0.9.11, and build the conda environment:

```Shell
git clone https://github.com/qintonguav/e2e-parking-carla.git
cd /e2e-parking-carla/
conda env create -f environment.yml
conda activate E2EParking
chmod +x setup_carla.sh
./setup_carla.sh
```
CUDA 11.7 is used as default. We also validate the compatibility of CUDA 10.2 and 11.3.

## Evaluation (Inference with pre-trained model)
For inference, we prepare a [pre-trained model](https://drive.google.com/file/d/1XOlzBAb9W91R6WOB-srgdY8AZH3fXlML/view?usp=sharing). Due to enterprise restrictions, this pre-trained model has an overall success rate of around 75%, which is not our best-performing model. For training and evaluating the model, we recommend you generate the training dataset by yourself using our data generation pipeline.


The first step is to launch a CARLA server:

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
When the evaluation is completed, metrics will be saved to csv files located at '--eva_result_path'.

## Dataset and Training

### Training Data Generation
Since it is an imitation learning task, we have provided the tools for manual parking data generation, so that the vehicle can learn from your driving habit. 
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
BackSpace:  reset the current task
TAB:        switch camera view
```

Conditions for successful parking:
```
position: vehicle center to slot center < 0.5 meter
orientation: rotation error < 0.5 degree
duration: satisfy the above two conditions for 60 frames
```
The target parking slot is marked with a red 'T'. 

Automatically switch to the next task when the current one is completed.

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
