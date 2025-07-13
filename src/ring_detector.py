import cv2
import numpy as np

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
        
        # 过程噪声协方差矩阵
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        
        # 测量噪声协方差矩阵
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 10
        
        # 后验误差协方差矩阵
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1000
        
        self.is_initialized = False
        self.last_seen = 0
        self.max_lost_frames = 5
    
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

class RingTracker:
    """
    圆环跟踪器类，管理多个圆环的卡尔曼滤波器
    """
    def __init__(self):
        self.trackers = []
        self.next_id = 0
        self.distance_threshold = 50  # 关联距离阈值
    
    def update(self, detections):
        """更新跟踪器"""
        # 预测所有跟踪器的状态
        predictions = []
        for tracker in self.trackers:
            pred = tracker.predict()
            if pred is not None:
                predictions.append(pred)
            else:
                predictions.append(None)
        
        # 数据关联 - 匹配检测结果与跟踪器
        matched_trackers = []
        unmatched_detections = list(detections)
        
        for i, tracker in enumerate(self.trackers):
            if predictions[i] is None:
                continue
            
            pred_x, pred_y, pred_r = predictions[i]
            best_match = None
            best_distance = float('inf')
            
            for j, (det_x, det_y, det_r, inner_r) in enumerate(unmatched_detections):
                # 计算预测位置与检测位置的距离
                distance = np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)
                
                if distance < self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is not None:
                # 找到匹配，更新跟踪器
                det_x, det_y, det_r, inner_r = unmatched_detections[best_match]
                corrected_state = tracker.update(det_x, det_y, det_r)
                matched_trackers.append((tracker, corrected_state, inner_r, i))
                unmatched_detections.pop(best_match)
            else:
                # 没有匹配，增加丢失帧数
                tracker.increment_lost_frames()
        
        # 创建新的跟踪器用于未匹配的检测
        for det_x, det_y, det_r, inner_r in unmatched_detections:
            new_tracker = KalmanFilter()
            new_tracker.initialize(det_x, det_y, det_r)
            corrected_state = new_tracker.update(det_x, det_y, det_r)
            matched_trackers.append((new_tracker, corrected_state, inner_r, self.next_id))
            self.trackers.append(new_tracker)
            self.next_id += 1
        
        # 移除丢失的跟踪器
        self.trackers = [t for t in self.trackers if not t.is_lost()]
        
        return matched_trackers

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
        param2=50,   # 圆心检测的阈值
        minRadius=50,  # 最小半径
        maxRadius=70  # 最大半径
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
            inner_radius = int(r * 0.72)  # 内圆半径为外圆的30%
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

def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 初始化圆环跟踪器
    tracker = RingTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 识别圆环
        raw_rings = detect_rings(frame)
        
        # 使用卡尔曼滤波器跟踪圆环
        tracked_rings = tracker.update(raw_rings)
        
        # 绘制原始检测结果（浅色）
        for (x, y, outer_r, inner_r) in raw_rings:
            cv2.circle(frame, (x, y), outer_r, (100, 100, 100), 1)  # 灰色，原始检测
        
        # 绘制跟踪结果（亮色）
        for tracker_obj, (x, y, r), inner_r, track_id in tracked_rings:
            x, y, r = int(x), int(y), int(r)
            
            # 绘制外圆
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # 绿色，跟踪结果
            # 绘制内圆
            cv2.circle(frame, (x, y), inner_r, (255, 0, 0), 2)  # 蓝色
            # 绘制圆心
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # 红色
            
            # 添加跟踪ID标签
            cv2.putText(frame, f'Ring {track_id} ({x},{y})', 
                       (x-50, y-r-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示统计信息
        info_text = f'Raw: {len(raw_rings)}, Tracked: {len(tracked_rings)}'
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示图例
        cv2.putText(frame, 'Gray=Raw Detection, Green=Kalman Filtered', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow('Ring Detection with Kalman Filter', frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
