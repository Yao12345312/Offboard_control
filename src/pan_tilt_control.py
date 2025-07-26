#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial
import threading
import time

class OffsetSerialSenderNode(Node):
    def __init__(self):
        super().__init__('offset_serial_sender')

        # 初始化串口（根据你实际的串口口名调整）
        try:
            self.serial_port = serial.Serial('/dev/ttyAMA3', baudrate=115200, timeout=1)
            self.get_logger().info("Serial port opened successfully.")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            rclpy.shutdown()

        # 初始化 offset 数据
        self.latest_offset = None
        self.lock = threading.Lock()

        # 订阅 /ring/offset
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/ring/offset',
            self.offset_callback,
            10
        )

        # 启动串口发送线程
        self.sender_thread = threading.Thread(target=self.serial_send_loop)
        self.sender_thread.daemon = True
        self.sender_thread.start()

    def offset_callback(self, msg):
        """收到视觉节点发送的 offset_x/y 数据（单位：米）"""
        if len(msg.data) >= 2:
            offset_x = msg.data[0]
            offset_y = msg.data[1]
            with self.lock:
                self.latest_offset = (offset_x, offset_y)

    def serial_send_loop(self):
        """串口持续发送线程，每隔 50ms 发送一次最新偏差"""
        while rclpy.ok():
            with self.lock:
                if self.latest_offset:
                    offset_x, offset_y = self.latest_offset
                    try:
                        # 构造字符串发送，例如："0.025,-0.010\n"
                        output = f"{offset_x:.3f},{offset_y:.3f}\n"
                        self.serial_port.write(output.encode('utf-8'))
                        self.get_logger().info(f"Sent offset to serial: {output.strip()}")
                    except serial.SerialException as e:
                        self.get_logger().error(f"Serial write error: {e}")
            time.sleep(0.05)  # 20Hz发送频率

def main(args=None):
    rclpy.init(args=args)
    node = OffsetSerialSenderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
