#!/usr/bin/env python3    
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import time
from gpiozero import LED
from threading import Thread

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

        # 创建发布器发布 /target_goods_number
        self.target_goods_number_publisher = self.create_publisher(String, '/target_goods_number', 10)

        # 创建订阅器接收 /serial_screen_command
        self.create_subscription(String, '/serial_screen_command', self.screen_command_callback, 10)

        # 启动串口数据读取定时器
        self.serial_timer = self.create_timer(0.4, self.read_serial_data)

        # 初始化上一个读取的数据为空
        self.last_received_data = None
        self.last_command = None  # 上一个命令记录

        # 启动串口读取线程
        self.read_thread = Thread(target=self.read_serial_data_thread)
        self.read_thread.daemon = True  # 设置为守护线程，随主线程退出
        self.read_thread.start()

    def read_serial_data(self):
        """串口数据读取接口，将数据发布到话题"""
        if self.serial_port.in_waiting > 0:
            data = self.serial_port.readline().decode('gb2312', errors='ignore').strip()

            # 如果数据存在并且与上次的数据不同
            if data and data != self.current_data:
                self.get_logger().info(f"New serial data detected: {data}")
                self.current_data = data  # 更新当前数据内容

                # 发布前12位（包括第12位）到 /task_topic
                task_data = data[:12]  # 获取前12个字符
                msg_task = String()
                msg_task.data = task_data
                self.task_publisher.publish(msg_task)
                self.get_logger().info(f"Published to /task_topic: {task_data}")

                # 判断第13位是否为 A、B、C、D，若是，则将其余部分发布到 /target_goods_number
                if len(data) > 12 and data[12] in 'ABCD':
                    target_goods_number = data[12:]  # 从第13位开始（包括第13位）
                    msg_target = String()
                    msg_target.data = target_goods_number
                    self.target_goods_number_publisher.publish(msg_target)
                    self.get_logger().info(f"Published to /target_goods_number: {target_goods_number}")

    def screen_command_callback(self, msg):
        """接收 /serial_screen_command 话题数据并发送到串口"""
        command = msg.data
        self.get_logger().info(f"Sending command to serial: {command}")
        
        # 启动串口写入线程
        write_thread = Thread(target=self.write_serial_data, args=(command,))
        write_thread.daemon = True  # 设置为守护线程，随主线程退出
        write_thread.start()

    def write_serial_data(self, command):
        """串口写数据"""
        try:
            self.serial_port.write(command.encode('gb2312') + b'\xFF\xFF\xFF')
            self.get_logger().info(f"Sent command to serial: {command}")
        except serial.SerialException as e:
            self.get_logger().error(f"Error while sending command to serial: {e}")

        # 如果 command 与上次不同，则触发 LED
        if command != self.last_command:
            self.last_command = command
            self.trigger_led()

    def trigger_led(self):
        """点亮 LED 1 秒后自动熄灭（线程方式更稳）"""
        def led_flash():
            self.get_logger().info("Command changed, LED ON for 1 second")
            self.led.on()
            time.sleep(1.0)
            self.led.off()
            self.get_logger().info("LED OFF")

        Thread(target=led_flash, daemon=True).start()

    def read_serial_data_thread(self):
        """独立线程循环读取串口数据"""
        while rclpy.ok():
            self.read_serial_data()
            time.sleep(0.1)  # 控制读取频率

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
