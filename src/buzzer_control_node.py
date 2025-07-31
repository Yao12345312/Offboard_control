#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from gpiozero import OutputDevice
import time


class BuzzerController(Node):
    def __init__(self):
        super().__init__('buzzer_controller')

        # 初始化 GPIO17 控制蜂鸣器（BCM 引脚号）
        self.buzzer = OutputDevice(17)

        # 创建订阅者
        self.subscription = self.create_subscription(
            Bool,
            'control_buzzer',
            self.listener_callback,
            10
        )

        self.get_logger().info('Buzzer Controller Node started and waiting for control messages.')

        # 标志是否正在响铃，防止重复触发
        self.buzzer_active = False

    def listener_callback(self, msg):
        if msg.data and not self.buzzer_active:
            self.get_logger().info('Received True: Turning buzzer ON for 0.5 seconds.')
            self.buzzer_active = True
            self.buzzer.on()

            # 启动一个定时器，在0.5秒后关闭蜂鸣器
            self.create_timer(0.5, self.turn_off_buzzer_once)

        elif not msg.data:
            self.get_logger().info('Received False: Ensuring buzzer is OFF.')
            self.buzzer.off()
            self.buzzer_active = False

    def turn_off_buzzer_once(self):
        if self.buzzer_active:
            self.buzzer.off()
            self.buzzer_active = False
            self.get_logger().info('Buzzer automatically turned OFF after 0.5 seconds.')


def main(args=None):
    rclpy.init(args=args)
    node = BuzzerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.buzzer.off()
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
