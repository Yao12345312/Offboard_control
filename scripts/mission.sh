#!/bin/bash
source /opt/ros/jazzy/setup.bash

source ~/mavros2_ws/install/setup.bash
#pi5 UART0 connect
sudo chmod 777 /dev/ttyAMA1
sudo chmod 777 /dev/ttyAMA0
sudo chmod 777 /dev/ttyAMA4

export PATH="$PATH:/home/pi5/.local/bin"

export ROS_DOMAIN_ID=1
export ROS_MASTER_URI=http://192.168.50.1:11311

ros2 launch tracked2vision tracked2vision.launch.py 

ros2 bag record -a


exit 0