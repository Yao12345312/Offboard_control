#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point  # 发布圆环中心位置
from std_msgs.msg import Float32MultiArray  # 发布坐标偏差
import cv2
import numpy as np
import time

class KalmanFilter:
    """
    卡尔曼滤波器类，用于跟踪圆环的位置和半径
    """
    def __init__(self, dt=1.0):
        # 状态向量: [x, y, r, vx, vy, vr] (位置、半径、速度)
        self.kalman = cv2.KalmanFilter(6, 3)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0]], np.float32)
        
        # 状态转移矩阵
        self.kalman.transitionMatrix = np.array([[1, 0, 0, dt, 0, 0],
                                                [0, 1, 0, 0, dt, 0],
                                                [0, 0, 1, 0, 0, dt],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]], np.float32)
        
        self.Q = 0.1
        self.R = 10
        self.P = 1000

        # 过程噪声协方差矩阵
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * self.Q
        
        # 测量噪声协方差矩阵
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.R
        
        # 后验误差协方差矩阵
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * self.P
        
        self.is_initialized = False
        self.last_seen = 0
        self.max_lost_frames = 10
    
    def initialize(self, x, y, r):
        """初始化卡尔曼滤波器"""
        self.kalman.statePre = np.array([x, y, r, 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, r, 0, 0, 0], dtype=np.float32)
        self.is_initialized = True
        self.last_seen = 0
    
    def predict(self):
        """预测下一帧的状态"""
        if not self.is_initialized:
            return None
        
        prediction = self.kalman.predict()
        return prediction[:3]  # 返回 [x, y, r]
    
    def update(self, x, y, r):
        """更新测量值"""
        if not self.is_initialized:
            self.initialize(x, y, r)
            return np.array([x, y, r])
        
        measurement = np.array([x, y, r], dtype=np.float32)
        self.kalman.correct(measurement)
        self.last_seen = 0
        
        # 返回修正后的状态
        return self.kalman.statePost[:3]
    
    def is_lost(self):
        """检查目标是否丢失"""
        return self.last_seen > self.max_lost_frames
    
    def increment_lost_frames(self):
        """增加丢失帧数"""
        self.last_seen += 1

class SingleRingTracker:
    """
    单圆环跟踪器类，管理一个圆环的卡尔曼滤波器
    """
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.distance_threshold = 50  # 关联距离阈值
    
    def update(self, detections):
        """更新跟踪器"""
        if not detections:
            # 没有检测到圆环
            if self.kalman_filter.is_initialized:
                self.kalman_filter.increment_lost_frames()
                
                if not self.kalman_filter.is_lost():
                    # 使用预测状态
                    prediction = self.kalman_filter.predict()
                    if prediction is not None:
                        return prediction, "predicted"
                else:
                    # 重置跟踪器
                    self.kalman_filter = KalmanFilter()
                    return None, "lost"
            return None, "no_detection"
        
        # 只处理第一个检测到的圆环
        det_x, det_y, det_r, inner_r = detections[0]
        
        if self.kalman_filter.is_initialized:
            # 检查是否与现有跟踪器匹配
            prediction = self.kalman_filter.predict()
            if prediction is not None:
                pred_x, pred_y, pred_r = prediction
                distance = np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)
                
                if distance < self.distance_threshold:
                    # 匹配成功，更新跟踪器
                    corrected_state = self.kalman_filter.update(det_x, det_y, det_r)
                    return corrected_state, "tracking"
                else:
                    # 距离太远，重新初始化
                    self.kalman_filter = KalmanFilter()
                    self.kalman_filter.initialize(det_x, det_y, det_r)
                    corrected_state = self.kalman_filter.update(det_x, det_y, det_r)
                    return corrected_state, "reinitialized"
            else:
                # 预测失败，重新初始化
                self.kalman_filter = KalmanFilter()
                self.kalman_filter.initialize(det_x, det_y, det_r)
                corrected_state = self.kalman_filter.update(det_x, det_y, det_r)
                return corrected_state, "reinitialized"
        else:
            # 首次检测，初始化跟踪器
            self.kalman_filter.initialize(det_x, det_y, det_r)
            corrected_state = self.kalman_filter.update(det_x, det_y, det_r)
            return corrected_state, "initialized"

def detect_rings(frame):
    """
    识别圆环的函数
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 使用HoughCircles检测圆形
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,  # 圆心之间的最小距离
        param1=50,   # Canny边缘检测的高阈值
        param2=40,   # 圆心检测的阈值
        minRadius=100,  # 最小半径
        maxRadius=150  # 最大半径
    )
    
    rings = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # 检查是否为圆环（中心有空洞）
        for (x, y, r) in circles:
            # 提取圆形区域
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # 创建内圆mask检测中心空洞
            inner_mask = np.zeros(gray.shape, dtype=np.uint8)
            inner_radius = int(r * 0.71)  # 内圆半径为外圆的75%
            cv2.circle(inner_mask, (x, y), inner_radius, 255, -1)
            
            # 计算内圆区域的平均亮度
            inner_region = cv2.bitwise_and(gray, inner_mask)
            inner_mean = cv2.mean(inner_region, mask=inner_mask)[0]
            
            # 计算圆环区域的平均亮度
            ring_mask = cv2.subtract(mask, inner_mask)
            ring_region = cv2.bitwise_and(gray, ring_mask)
            ring_mean = cv2.mean(ring_region, mask=ring_mask)[0]
            
            # 如果内圆明显比圆环区域亮（或暗），则可能是圆环
            brightness_diff = abs(inner_mean - ring_mean)
            if brightness_diff > 20:  # 阈值可调整
                rings.append((x, y, r, inner_radius))
    
    return rings

class RingDetectorNode(Node):
    def __init__(self):
        super().__init__('ring_detector_node')
        
        # 初始化摄像头（保持原有设置）
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
            raise RuntimeError("无法打开摄像头")
        
        # 初始化跟踪器（保持原有逻辑）
        self.tracker = SingleRingTracker()
        self.camera_center_x = 160
        self.camera_center_y = 120
        
        # 添加ROS2发布者
        self.position_pub = self.create_publisher(Point, '/ring/position', 10)
        self.offset_pub = self.create_publisher(Float32MultiArray, '/ring/offset', 10)
        
        # 保持原有的可视化窗口
        self.window_name = 'Ring Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 使用定时器控制处理频率（约30Hz）
        self.timer = self.create_timer(0.033, self.process_frame)
        
        self.get_logger().info("圆环检测节点已启动 (带可视化)")
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("无法读取摄像头帧")
            return
        
        # 更新相机中心（保持原有逻辑）
        height, width = frame.shape[:2]
        self.camera_center_x = width // 2
        self.camera_center_y = height // 2
        
        # 检测圆环（保持原有逻辑）
        raw_rings = detect_rings(frame)
        
        # 跟踪圆环（保持原有逻辑）
        tracked_state, tracking_status = self.tracker.update(raw_rings)
        
        # 可视化部分（保持原有逻辑）
        for (x, y, outer_r, inner_r) in raw_rings:
            cv2.circle(frame, (x, y), outer_r, (100, 100, 100), 1)
        
        if tracked_state is not None:
            x, y, r = int(tracked_state[0]), int(tracked_state[1]), int(tracked_state[2])
            inner_r = int(r * 0.75)
            
            # 计算偏差（保持原有逻辑）
            offset_x = (x - self.camera_center_x) * 0.0025
            offset_y = (y - self.camera_center_y) * 0.0025
            
            # 发布ROS2消息（新增部分）
            self.publish_detection(x, y, offset_x, offset_y)
            
            # 可视化跟踪结果（保持原有逻辑）
            color = (0, 255, 0) if tracking_status == "tracking" else (0, 255, 255)
            cv2.circle(frame, (x, y), r, color, 2)
            cv2.circle(frame, (x, y), inner_r, (255, 0, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            cv2.line(frame, (self.camera_center_x, self.camera_center_y), 
                    (x, y), (0, 255, 255), 2)
        
        # 显示相机中心和帧率（保持原有逻辑）
        cv2.circle(frame, (self.camera_center_x, self.camera_center_y), 5, (0, 255, 255), -1)
        cv2.imshow(self.window_name, frame)
        
        # 按'q'键退出（保持原有逻辑）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cleanup()
            rclpy.shutdown()
    
    def publish_detection(self, x, y, offset_x, offset_y):
        """发布检测结果到ROS2主题"""
        # 发布位置消息
        pos_msg = Point()
        pos_msg.x = float(x)
        pos_msg.y = float(y)
        pos_msg.z = 0.0  # 2D情况下z设为0
        #self.position_pub.publish(pos_msg)
        
        # 发布偏差消息
        offset_msg = Float32MultiArray()
        offset_msg.data = [float(offset_x), float(offset_y)]
        self.offset_pub.publish(offset_msg)
        
        # 可选：打印日志
        self.get_logger().info(
            f"偏差: ({offset_x:.3f}, {offset_y:.3f})"
            # throttle_duration_sec=1.0  # 限流，每秒最多打印一次
        )
    
    def cleanup(self):
        """清理资源"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("资源已释放")

def main(args=None):
    rclpy.init(args=args)
    node = RingDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
