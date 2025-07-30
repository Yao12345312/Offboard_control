import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

from model import MobileNetV2


class AnimalVideoClassifier:
    def __init__(self, model_path="./mobilenetV2.pth", json_path="./class_indices.json"):
        """初始化视频流分类器"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 数据预处理管道
        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载类别标签
        self.load_class_labels(json_path)
        
        # 加载模型
        self.load_model(model_path)
        
        # 预测相关变量
        self.prediction_interval = 10  # 每10帧预测一次，提高性能
        self.frame_count = 0
        self.last_prediction = "正在识别..."
        self.last_confidence = 0.0
        
        # 单目标检测相关变量
        self.best_detection = None  # 存储最佳检测结果
        self.detection_threshold = 0.3  # 检测阈值
        
        # 初始化背景减除器用于运动检测
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def load_class_labels(self, json_path):
        """加载类别标签"""
        assert os.path.exists(json_path), f"类别文件不存在: {json_path}"
        
        with open(json_path, "r", encoding='utf-8') as f:
            self.class_indict = json.load(f)
        
        print(f"加载类别标签: {self.class_indict}")
    
    def load_model(self, model_path):
        """加载模型"""
        assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
        
        # 创建模型
        self.model = MobileNetV2(num_classes=len(self.class_indict)).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print("模型加载成功")
    
    def detect_moving_objects(self, frame):
        """检测画面中的运动目标"""
        # 应用背景减除
        fg_mask = self.background_subtractor.apply(frame)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选有效的运动区域
        moving_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 过滤太小的区域
            if area > 1000:  # 可调整
                x, y, w, h = cv2.boundingRect(contour)
                # 过滤太小或太大的框
                if w > 50 and h > 50 and w < frame.shape[1]*0.8 and h < frame.shape[0]*0.8:
                    moving_boxes.append((x, y, x+w, y+h))
        
        return moving_boxes
    
    def sliding_window_detection(self, frame):
        """使用滑动窗口检测动物"""
        height, width = frame.shape[:2]
        detected_boxes = []
        
        # 定义不同尺寸的滑动窗口
        window_sizes = [(150, 150), (200, 200), (250, 250)]
        step_size = 80  # 增大步长减少检测框数量
        
        for window_w, window_h in window_sizes:
            for y in range(0, height - window_h, step_size):
                for x in range(0, width - window_w, step_size):
                    # 提取窗口区域
                    window = frame[y:y+window_h, x:x+window_w]
                    
                    # 预测窗口内容
                    class_name, confidence = self.predict_region(window)
                    
                    # 如果置信度高于阈值，记录检测结果
                    if confidence > self.detection_threshold and class_name != "错误":
                        detected_boxes.append({
                            'bbox': (x, y, x+window_w, y+window_h),
                            'class': class_name,
                            'confidence': confidence
                        })
        
        return detected_boxes
    
    def predict_region(self, region):
        """预测图像区域的类别"""
        try:
            # 转换BGR到RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            pil_image = Image.fromarray(region_rgb)
            
            # 应用预处理
            img_tensor = self.data_transform(pil_image)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            
            # 预测
            with torch.no_grad():
                output = torch.squeeze(self.model(img_tensor.to(self.device))).cpu()
                predict_probs = torch.softmax(output, dim=0)
                predict_class = torch.argmax(predict_probs).numpy()
            
            # 获取结果
            class_name = self.class_indict[str(predict_class)]
            confidence = predict_probs[predict_class].numpy()
            
            return class_name, confidence
            
        except Exception as e:
            return "错误", 0.0
    
    def find_best_detection(self, detections):
        """从所有检测结果中选择最佳的一个"""
        if not detections:
            return None
        
        # 按置信度排序，选择最高的
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        return best_detection
    
    def detect_and_classify(self, frame):
        """检测并分类画面中的动物 - 单目标版本"""
        all_detections = []
        
        # 方法1: 使用运动检测
        moving_boxes = self.detect_moving_objects(frame)
        
        # 对运动区域进行分类
        for box in moving_boxes:
            x1, y1, x2, y2 = box
            region = frame[y1:y2, x1:x2]
            
            if region.size > 0:
                class_name, confidence = self.predict_region(region)
                
                if confidence > self.detection_threshold:
                    all_detections.append({
                        'bbox': box,
                        'class': class_name,
                        'confidence': confidence
                    })
        
        # 方法2: 如果运动检测没有找到目标，使用滑动窗口（降低频率）
        if not all_detections and self.frame_count % (self.prediction_interval * 4) == 0:
            sliding_detections = self.sliding_window_detection(frame)
            all_detections.extend(sliding_detections)
        
        # 选择最佳检测结果
        best_detection = self.find_best_detection(all_detections)
        
        return best_detection
    
    def draw_single_detection(self, frame, detection):
        """绘制单个检测结果"""
        if detection is None:
            return frame
        
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # 根据置信度设置颜色
        if confidence > 0.7:
            color = (0, 255, 0)  # 绿色
        elif confidence > 0.4:
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 0, 255)  # 红色
        
        # 绘制边界框 - 加粗显示
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # 绘制框角加强显示
        corner_length = 30
        corner_thickness = 5
        
        # 左上角
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # 右上角
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # 左下角
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # 右下角
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # 绘制类别和置信度标签
        label = f"{class_name}: {confidence:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # 绘制标签背景
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0] + 10, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 绘制置信度条
        bar_width = x2 - x1
        bar_height = 8
        fill_width = int(bar_width * confidence)
        
        # 置信度条背景
        cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), (50, 50, 50), -1)
        # 置信度条填充
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + fill_width, y2 + 5 + bar_height), color, -1)
        # 置信度条边框
        cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), (255, 255, 255), 1)
        
        return frame
    
    def draw_info(self, frame, detection):
        """绘制信息面板"""
        height, width = frame.shape[:2]
        
        # 绘制信息背景
        info_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        if detection:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # 显示检测结果
            cv2.putText(frame, f"检测结果: {class_name}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, f"置信度: {confidence:.3f} ({confidence*100:.1f}%)", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示检测框坐标
            x1, y1, x2, y2 = detection['bbox']
            cv2.putText(frame, f"位置: ({x1},{y1}) - ({x2},{y2})", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "未检测到目标", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
            cv2.putText(frame, f"检测阈值: {self.detection_threshold:.2f}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run_camera(self, camera_id=0):
        """运行摄像头实时识别"""
        print(f"启动摄像头 {camera_id}...")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"无法打开摄像头 {camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("摄像头启动成功！")
        print("使用说明:")
        print("- 系统会自动检测并框选置信度最高的动物")
        print("- 绿色框=高置信度，黄色框=中等置信度，红色框=低置信度")
        print("- 按 'q' 退出程序")
        print("- 按 't' 调整检测阈值")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                self.frame_count += 1
                
                # 检测并分类 - 只保留最佳结果
                if self.frame_count % self.prediction_interval == 0:
                    best_detection = self.detect_and_classify(frame)
                    self.best_detection = best_detection
                    
                    # 打印检测结果
                    if best_detection:
                        print(f"帧 {self.frame_count}: 检测到 {best_detection['class']} (置信度: {best_detection['confidence']:.3f})")
                    else:
                        print(f"帧 {self.frame_count}: 未检测到目标")
                
                # 绘制单个最佳检测结果
                frame_with_detection = self.draw_single_detection(frame, self.best_detection)
                frame_with_info = self.draw_info(frame_with_detection, self.best_detection)
                
                # 显示画面
                cv2.imshow('动物单目标检测与识别', frame_with_info)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    # 调整检测阈值
                    new_threshold = input(f"当前阈值: {self.detection_threshold:.2f}, 输入新阈值 (0.1-0.9): ")
                    try:
                        self.detection_threshold = max(0.1, min(0.9, float(new_threshold)))
                        print(f"阈值已调整为: {self.detection_threshold:.2f}")
                    except:
                        print("无效输入，保持原阈值")
        
        except KeyboardInterrupt:
            print("\n用户中断")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("摄像头已关闭")


def main():
    """主函数"""
    # 检查必要文件
    model_path = "./CV/mobilenetV2.pth"
    json_path = "./CV/class_indices.json"
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    if not os.path.exists(json_path):
        print(f"类别文件不存在: {json_path}")
        print("请先运行训练脚本生成类别索引文件")
        return
    
    # 创建分类器
    print("正在初始化动物单目标检测与识别系统...")
    classifier = AnimalVideoClassifier(model_path, json_path)
    
    # 获取摄像头ID
    camera_id = input("输入摄像头ID (默认0): ").strip()
    camera_id = int(camera_id) if camera_id.isdigit() else 0
    
    print(f"\n启动单目标实时检测与识别...")
    
    # 运行实时识别
    classifier.run_camera(camera_id)


if __name__ == '__main__':
    main()