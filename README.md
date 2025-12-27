# Autopilot Neural Network

## 🤖 About

This package implements the data collection, training, and inference pipeline for an end-to-end autonomous driving system in ROS 2. It is responsible for gathering data, organizing and storing datasets for training, and providing the neural network training scripts along with a runtime inference node for autonomous control.

## 💻 Instalation


Clone this repository into your ```workspace/src``` folder. If you don't have a workspace set up, you can learn more about creating one in the [ROS 2 workspace tutorial](https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html).

```bash
cd <path_to_your_workspace>/src
git clone git@github.com:lucasmazzetto/autopilot_neural_network.git
```

### 🐧 Linux Setup

This project is designed to run on Linux Ubuntu 24.04 and may also work on other Linux versions or distributions, although additional adjustments might be required. 

#### 📚 Requirements

To use this package, you'll need the following:

- [Linux Ubuntu 24.04](https://ubuntu.com/blog/tag/ubuntu-24-04-lts)
- [ROS 2 Jazzy Jalisco](https://docs.ros.org/en/rolling/Releases/Release-Jazzy-Jalisco.html)
- [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/getstarted/)


This package is designed to work alongside the following optional packages. If you choose to use them, make sure the dependencies listed below are installed.

- [gazebo_racing_tracks](https://github.com/lucasmazzetto/gazebo_racing_tracks): provides race track simulation environments  
- [gazebo_ackermann_steering_vehicle](https://github.com/lucasmazzetto/gazebo_ackermann_steering_vehicle): provides the Ackermann steering vehicle, controller, camera, and joystick interface  

It is also required to install the Python dependencies using a virtual environment configured with access to system-wide Python packages. This configuration ensures compatibility with the ROS 2 Python ecosystem, as some dependencies are installed at the system level and must be available at runtime alongside the Python packages installed in the virtual environment. Create the virtual environment with system package access using the following command:

```bash
python3 -m venv <path_to_your_venv> --system-site-packages
```

After creating the virtual environment, activate it and install the required Python dependencies using pip:

```bash
source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace>/src/autopilot_neural_network

pip install -r requirements.txt
```

#### 🛠️ Build

Source the ROS 2 environment and the Python virtual environment, then build the package:

```bash
source /opt/ros/jazzy/setup.bash

source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace>

colcon build
```

After a successful build, the package is ready to be used.

## 🚀 Usage

This section explains how to use the system to collect data, train the neural network, and run autonomous inference in the simulation environment.

### 🗂️ Data Collection

Before collecting data, configure where the dataset will be stored. Update the `dataset_path` parameter in the `autopilot_neural_network/config/parameters.yaml` file to point to your desired dataset destination.

Then, to launch the simulation for data collection, set up the environment and start the vehicle simulation with the desired track. Available tracks include `dirt_track`, `snow_track`, `grass_track`,`sand_track`, and `grass_track`:

```bash
source /opt/ros/jazzy/setup.bash 

source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace>

source install/setup.bash 

ros2 launch gazebo_ackermann_steering_vehicle vehicle.launch.py \
  world:=$(ros2 pkg prefix gazebo_racing_tracks)/share/gazebo_racing_tracks/worlds/dirt_track.sdf
```

To enable manual vehicle control, connect an Xbox One controller and in a separate terminal launch the joystick interface:

```bash
source /opt/ros/jazzy/setup.bash

source <path_to_your_venv>/bin/activate

cd ~/workspace

source install/setup.bash

ros2 launch gazebo_ackermann_steering_vehicle joystick.launch.py
```

In a third terminal, set up the environment and launch the `data_collector` node to start recording vehicle data:

```bash
source /opt/ros/jazzy/setup.bash 

source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace>

source install/setup.bash 

ros2 launch autopilot_neural_network data_collector.launch.py
```


### 📉 Neural Network Training


To start the training process, activate the Python virtual environment and run the training script with the dataset and model paths specified:

```bash
source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace>/src/autopilot_neural_network/scripts/

python3 train.py --dataset_path <path_to_your_dataset> \
                 --model_path <path_to_your_model>
```

The following arguments can be used with the `train.py` script to configure the training process:

- --`dataset_path`: Path to the directory containing the training dataset.

- --`model_path`: Path where the best trained model checkpoint will be saved.

- --`epochs`: Number of training epochs to run (default: 100).

- --`batch_size`: Batch size used for the training data loader (default: 1024).

- --`val_batch_size`: Batch size used for the validation data loader (default: 256).

- --`val_fraction`: Fraction of the dataset reserved for validation (e.g., 0.2 uses 20% of the data for validation).

- --`learning_rate`: Initial learning rate used by the optimizer  (default: 1e-3).

- --`lr_patience`: Number of epochs without validation loss improvement before reducing the learning rate.

- --`lr_factor`: Factor by which the learning rate is reduced when a plateau is detected.

- --`alpha`: Weight applied to the velocity component of the loss function.

- --`beta`: Weight applied to the steering angle component of the loss function.

- --`num_workers`: Number of worker processes used for loading data (default: number of CPU cores minus one).

- --`height`: Target height for resizing input images (default: 96 pixels).

- --`width`: Target width for resizing input images (default: 128 pixels).

- --`sampler_low_fraction`: Fraction of low-steering samples retained during dataset balancing.

- --`sampler_threshold_ratio`: Steering ratio (relative to maximum steering) used to define the low-steering region.

### 🚗 Inference

Before starting the `autopilot` node, make sure the correct model is configured. Update the `model_path` parameter in the `autopilot_neural_network/config/parameters.yaml` file to point to the location of your trained model. This ensures that the autopilot loads and uses the intended neural network during execution.

Start the simulation environment before running the inference node. Launch the vehicle simulation using `vehicle.launch.py`. This initializes the Gazebo world and the vehicle:

```bash
source /opt/ros/jazzy/setup.bash

source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace> 

source install/setup.bash

ros2 launch gazebo_ackermann_steering_vehicle vehicle.launch.py \
  world:=$(ros2 pkg prefix gazebo_racing_tracks)/share/gazebo_racing_tracks/worlds/grass_track.sdf
```

Once the simulation is running, the autopilot node need to be started in a separate terminal, following the commands shown next:

```bash
source /opt/ros/jazzy/setup.bash 

source <path_to_your_venv>/bin/activate

cd <path_to_your_workspace> 

source install/setup.bash 

ros2 launch autopilot_neural_network autopilot.launch.py
```

After launching the simulation, the system is ready to operate. If the model was trained, the inference node will begin controlling the vehicle automatically.

## ⚙️ Parameters

The parameters for the data colection and inference nodes can be configured in the `autopilot_neural_network/config/parameters.yaml` file. This file includes the following settings with their default values:

```bash
# Vehicle node and topics
vehicle_node: "vehicle_controller" # Name of the node from which to retrieve vehicle parameters
velocity_topic: "/velocity" # Topic for publishing/subscribing to velocity commands
steering_angle_topic: "/steering_angle" # Topic for publishing/subscribing to steering commands
image_topic: "/camera/image_raw" # Topic for subscribing to raw camera images

# Data Collector Node Parameters
collector_min_velocity_factor: 0.25 # Fraction of max velocity below which data is not recorded
collector_timeout_time: 1000000000 # Timeout for data freshness in nanoseconds [ns]
collector_update_period: 0.5 # Period for the node's update loop in seconds [s]

# Autopilot (Inference) Node Parameters
image_height: 96 # Height of the image for model input [px] 
image_width: 128 # Width of the image for model input [px] 

# File Paths
dataset_path: "/tmp/autopilot_neural_network/dataset" # Path where dataset is saved and loaded
model_path: "/tmp/autopilot_neural_network/model.pt" # Path to the trained model for inference
```