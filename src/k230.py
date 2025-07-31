#!/usr/bin/env python3    
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
import serial
import re
import json
from threading import Thread

class K230SerialNode(Node):
    def __init__(self):
        super().__init__('k230_node')
        
        # 类别映射表 (class_id : class_name)
        self.class_map = {
            0: "elephant",
            1: "peacock", 
            2: "monkey",
            3: "tiger",
            4: "wolf"
        }
        
        # 串口初始化
        self.serial_port = self.init_serial('/dev/ttyAMA1', 115200)
        
        # 创建ROS发布器
        self.class_pub = self.create_publisher(String, '/k230/class_info', 10)  # 发布(id,name)对
        self.point_pub = self.create_publisher(Point, '/k230/position', 10)    # 发布坐标
        
        # 启动串口读取线程
        Thread(target=self.read_serial, daemon=True).start()

    def init_serial(self, port, baudrate):
        """初始化串口连接"""
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                bytesize=8,
                parity='N',
                stopbits=1
            )
            self.get_logger().info(f"成功连接K230串口 {port}")
            return ser
        except Exception as e:
            self.get_logger().error(f"串口连接失败: {e}")
            raise

    def read_serial(self):
        """持续读取串口数据"""
        while rclpy.ok():
            if self.serial_port.in_waiting > 0:
                try:
                    raw = self.serial_port.readline().decode('ascii').strip()
                    self.parse_data(raw)
                except Exception as e:
                    self.get_logger().warn(f"数据读取异常: {e}")

    def parse_data(self, data):
        """解析K230数据格式: C{class_id}X{x}Y{y}"""
        if not data.startswith('C'):
            return
            
        match = re.match(r'C(\d+)X(\d+)Y(\d+)', data)
        if match:
            class_id, x, y = match.groups()
            self.publish_messages(int(class_id), int(x), int(y))
        else:
            self.get_logger().warn(f"无效数据格式: {data}")

    def publish_messages(self, class_id, x, y):
        """发布组合消息"""
        # 处理类别信息
        class_name = self.class_map.get(class_id, f"unknown_{class_id}")
        class_data = {
            "class_id": class_id,
            "class_name": class_name
        }
        
        # 发布类别组合信息(JSON格式)
        class_msg = String()
        class_msg.data = json.dumps(class_data)  # 序列化为JSON字符串
        self.class_pub.publish(class_msg)
        
        # 发布坐标信息
        point_msg = Point()
        point_msg.x = float(x)
        point_msg.y = float(y)
        point_msg.z = 0.0  # 2D坐标
        self.point_pub.publish(point_msg)
        
        # 打印调试信息
        self.get_logger().info(
            f"发布: 类别=({class_id},{class_name}) 位置=({x},{y})",
            throttle_duration_sec=1
        )

def main(args=None):
    rclpy.init(args=args)
    node = K230SerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
