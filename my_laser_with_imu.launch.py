"""
  Copyright 2018 The Cartographer Authors
  Copyright 2022 Wyca Robotics (for the ros2 conversion)

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetRemap
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import RegisterEventHandler, ExecuteProcess, EmitEvent
from launch.event_handlers import OnProcessIO
from launch.events import Shutdown
import os
import re
# const int GLOG_INFO =0, GLOG_WARNING =1, GLOG_ERROR = 2, GLOG_FATAL = 3,NUM_SEVERITIES =4;
def generate_launch_description():

    ## ***** Launch arguments *****
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    ## ***** File paths ******
    pkg_share = FindPackageShare('cartographer_ros')
    
    # 新增URDF路径定义
    #urdf_model_path = PathJoinSubstitution([
    #    pkg_share,
    #    'urdf/my_backpack_2d.urdf'
    #])

    # 添加关节状态发布节点
#    joint_state_publisher_node = Node(
#        package='joint_state_publisher',
#        executable='joint_state_publisher',
#        name='joint_state_publisher',
#        arguments=[urdf_model_path],  # 添加逗号
#        output='screen'  # 建议添加输出配置
#    )

    # URDF加载节点（保持原样）
#    robot_state_publisher_node = Node(
#        package='robot_state_publisher',
#        executable='robot_state_publisher',
#        name='robot_state_publisher',
#        output='screen',
#        parameters=[{
#            'robot_description': urdf_model_path  # 使用统一定义的路径
#        }]
 #   )

    # 修正cartographer节点（缩进4空格+参数对齐）
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            # 新增消息队列控制参数
            # {'num_subdivisions_per_laser_scan': 1},
            # {'collate_landmarks': False},
            # {'collate_fixed_frame': False},
            # {'imu_gravity_time_constant': 10.0}
        ],
        
        arguments=[
            '-configuration_directory',
            PathJoinSubstitution([
                FindPackageShare('cartographer_ros'),
                'configuration_files'
            ]),
            '-configuration_basename', 'my_laser_with_imu.lua','-minloglevel','2'
        ],
        remappings=[
            ('scan', 'scan'),
            ('imu', '/mavros/imu/data')
        ],
        output='screen'
    )

    # 修正occupancy grid节点（补全参数列表）
    cartographer_occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'resolution': 0.05}  # 补全末尾逗号
        ]
    )

    # 修正rviz节点（参数垂直对齐）
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{
        'use_sim_time': LaunchConfiguration('use_sim_time'),
        'qos_overrides./tf_static.publisher.reliability': 'reliable',  # 提升TF可靠性
        'queue_size': 1000  # 增大消息队列容量
    }],
        output='screen'
    )
        # 定义终止轨迹0的服务调用
    finish_trajectory_service = ExecuteProcess(
        cmd=[
            'ros2', 'service', 'call',
            '/cartographer_node/finish_trajectory',
            'cartographer_ros_msgs/srv/FinishTrajectory',
            '{"trajectory_id": 0}'
        ],
        output='screen'
    )
    
    # 事件处理器：监听Cartographer节点的日志输出
    def on_cartographer_output(event):
        text = event.text.decode('utf-8').strip()
        # 正则匹配目标错误日志
        if re.search(r"Can't run final optimization.*trajectory with ID 0", text):
            print("\n[检测到地图完成信号] 正在终止轨迹0...*************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************")
            return [finish_trajectory_service]  # 可选：终止launch
        return []

    # 注册事件监听器
    carto_log_handler = RegisterEventHandler(
        event_handler=OnProcessIO(
            target_action=cartographer_node,
            on_stdout=on_cartographer_output,
            on_stderr=on_cartographer_output  # 同时监控stderr
        )
    )
    static_tf_pub = Node(
	package='tf2_ros',
	executable='static_transform_publisher',
	name='base_link_to_laser_static_tf',
	# x y z qx qy qz qw parent child
	arguments=['0', '0', '0.05', '0', '0', '0', '1',
	'base_link', 'laser'],
	output='screen'
	)
    return LaunchDescription([
        use_sim_time_arg,
#        robot_state_publisher_node,
#        joint_state_publisher_node,
#        rviz_node,
        cartographer_node,
        cartographer_occupancy_grid_node,
        #carto_log_handler,
        static_tf_pub
      

    ])
