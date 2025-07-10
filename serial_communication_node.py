#!/usr/bin/env python3  
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time

class SerialCommunicationNode(Node):
    def __init__(self):
        super().__init__('serial_communication_node')

        # 串口初始化
        try:
            self.serial_port = serial.Serial('/dev/ttyAMA3', baudrate=115200, timeout=5)
        except serial.SerialException as e:
            self.get_logger().error(f"Unable to open serial port: {e}")
            rclpy.shutdown()

        # 创建发布器发布 /task_topic
        self.task_publisher = self.create_publisher(String, '/task_topic', 10)

        # 创建订阅器接收 /serial_screen_command
        self.create_subscription(String, '/serial_screen_command', self.screen_command_callback, 10)

        # 启动串口数据读取定时器
        self.serial_timer = self.create_timer(0.4, self.read_serial_data)

        # 初始化上一个读取的数据为空
        self.last_received_data = None

    def read_serial_data(self):
        """从串口读取数据并发布到 /task_topic"""
        if self.serial_port.in_waiting > 0:
            data=self.serial_port.readline().decode('gb2312',errors='ignore').strip()

            if data and data != self.last_received_data:
                self.last_received_data = data
                self.get_logger().info(f"Received data from serial: {data}")
                msg = String()
                msg.data = data
                self.task_publisher.publish(msg)

    def screen_command_callback(self, msg):
        """接收 /serial_screen_command 话题数据并发送到串口"""
        command = msg.data
        self.get_logger().info(f"Sending command to serial: {command}")
        try:
            # 使用 GB2312 编码发送数据到串口
            self.serial_port.write(command.encode('gb2312') + b'\xFF\xFF\xFF')
	    
        except serial.SerialException as e:
            self.get_logger().error(f"Error while sending command to serial: {e}")

    def __del__(self):
        """关闭串口连接"""
        if self.serial_port.is_open:
            self.serial_port.close()

def main(args=None):
    rclpy.init(args=args)

    node = SerialCommunicationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
