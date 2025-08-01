#!/bin/bash
# 脚本路径：/home/pi5/start_vision_with_bag.sh

# 初始化ROS环境
source /opt/ros/jazzy/setup.bash
source ~/mavros2_ws/install/setup.bash

# 设置UART权限
sudo chmod 777 /dev/ttyAMA1
sudo chmod 777 /dev/ttyAMA0
sudo chmod 777 /dev/ttyAMA4

# 设置环境变量
export PATH="$PATH:/home/pi5/.local/bin"
export ROS_DOMAIN_ID=1
export ROS_MASTER_URI=http://192.168.50.1:11311

# 创建专用工作目录
BAG_DIR="/home/pi5/ros2_bags"
mkdir -p $BAG_DIR

# 启动ros2 launch和ros2 bag（后台运行）
ros2 launch tracked2vision tracked2vision.launch.py &
LAUNCH_PID=$!

# 等待节点启动
sleep 5

# 启动ros2 bag记录（带日期时间戳）
ros2 bag record -a -o $BAG_DIR/vision_$(date +%Y%m%d_%H%M%S) &
BAG_PID=$!

# 等待服务终止
wait $LAUNCH_PID

# 优雅终止bag记录
kill -INT $BAG_PID
wait $BAG_PID

exit 0