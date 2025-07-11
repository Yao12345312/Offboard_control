#!/usr/bin/env python3  
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
from gpiozero import LED
from threading import Timer

class SerialCommunicationNode(Node):
    def __init__(self):
        super().__init__('serial_communication_node')

        # 初始化 gpiozero 控制的 LED，GPIO17（物理引脚11）
        self.led = LED(17)
        self.led.off()  # 默认关闭
        self.current_data = None       # 当前要定时发布的数据
        self.publish_timer = None      # 定时发布的计时器对象


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
        self.last_command = None  # 上一个命令记录

    def read_serial_data(self):
        """从串口读取数据，如果与上次不同则定时发布"""
        if self.serial_port.in_waiting > 0:
            data = self.serial_port.readline().decode('gb2312', errors='ignore').strip()

            if data and data != self.current_data:
                self.get_logger().info(f"New serial data detected: {data}")
                self.current_data = data  # 更新当前数据内容

                # 若已有定时器，先取消
                if self.publish_timer is not None:
                    self.publish_timer.cancel()

                # 启动新的定时发布
                self.publish_timer = self.create_timer(1.0, self.publish_current_data)

    def publish_current_data(self):
        """定时发布当前串口数据"""
        if self.current_data:
            msg = String()
            msg.data = self.current_data
            self.task_publisher.publish(msg)
            self.get_logger().info(f"Published: {self.current_data}")

    def screen_command_callback(self, msg):
        """接收 /serial_screen_command 话题数据并发送到串口"""
        command = msg.data
        self.get_logger().info(f"Sending command to serial: {command}")
        
        try:
            self.serial_port.write(command.encode('gb2312') + b'\xFF\xFF\xFF')
        except serial.SerialException as e:
            self.get_logger().error(f"Error while sending command to serial: {e}")

        # 如果 command 与上次不同，则触发 LED
        if command != self.last_command:
            self.last_command = command
            self.trigger_led()

    def trigger_led(self):
        """点亮LED 1秒后自动熄灭"""
        self.get_logger().info("Command changed, LED ON for 1 second")
        self.led.on()
        Timer(1.0, self.led.off).start()  # 1秒后关闭 LED

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
