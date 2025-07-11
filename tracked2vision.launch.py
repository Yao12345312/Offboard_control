import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource

def generate_launch_description():
    # 获取包的路径
    lslidar_driver_launch_path = PathJoinSubstitution([FindPackageShare("lslidar_driver"), "launch", "lsm10_uart_launch.py"])
    cartographer_ros_launch_path = PathJoinSubstitution([FindPackageShare("cartographer_ros"), "launch", "my_laser_with_imu.launch.py"])
    mavros_launch_path = PathJoinSubstitution([FindPackageShare("mavros"), "launch", "px4.launch"])



    return LaunchDescription([
            # 启动 cartographer_laser_transfer 节点
        Node(
            package='tracked2vision',
            executable='cartographer_laser_transfer',
            name='cartographer_laser_transfer',
            output='screen'
        ),
         # 启动 servo_py 节点
        #Node(
        #    package='offboard_control',
        #    executable='servo_node.py',
        #    name='servo_node',
        #    output='screen'
        #),
        
        #启动云台节点
        Node(
            package='offboard_control',  
            executable='step_motor.py', 
            name='step_motor_node',
            output='screen'
        ),
	#启动激光笔控制节点
        Node(
            package='offboard_control',  
            executable='laser_pointer_control.py', 
            name='laser_pointer_control_node',
            output='screen'
        ),        
         # 启动 cv_py 节点
        Node(
            package='offboard_control',
            executable='cv.py',
            name='cv_node',
            output='screen'
        ),
    	# 启动 lslidar_driver lsn10_launch.launch.py 文件 (Python)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lslidar_driver_launch_path),
            launch_arguments={}.items()
        ),

         #启动 mavros px4.launch 文件 (XML)
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(mavros_launch_path),
            launch_arguments={}.items()
        ),
        
        # servo_node 
 	# 延时启动 Cartographer，假设 mavros 已经连接
        TimerAction(
            period=11.0,  # 定时器的持续时间为 11 秒
            actions=[
            IncludeLaunchDescription(
            	PythonLaunchDescriptionSource(cartographer_ros_launch_path),
            	launch_arguments={}.items()
            )]
        ),
        
	#TimerAction(
        #period=15.0,
        #actions=[
        #    Node(
        #        package='offboard_control',
        #        executable='offboard_node',
        #        name='offboard_node',
        #        output='screen'
        #    	)
        #	]
    	#),
       
       
        #tf2:map->odom->base_link
        #Node(
        #    package='tracked2vision',
         #   executable='tf_broadcaster_node',
        #    name='tf_broadcaster_node',
        #    output='screen'
        #),
        

    ])
