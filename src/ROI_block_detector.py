import cv2
import numpy as np

class RedROIDetector:
    def __init__(self):
        # 红色HSV阈值范围
        self.red_lower1 = np.array([0, 50, 50])      # 红色范围1 (0-10)
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])    # 红色范围2 (170-179)
        self.red_upper2 = np.array([179, 255, 255])
        
        # ROI相关变量
        self.roi_mask = None
        self.roi_contour = None
        self.roi_bbox = None
        
    def detect_red_objects(self, frame):
        """检测红色物块"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码（红色在HSV中分为两个范围）
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        
        # 合并两个掩码
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        return red_mask
    
    def find_largest_red_contour(self, mask):
        """找到最大的红色轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到面积最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 过滤太小的轮廓
        if cv2.contourArea(largest_contour) < 500:
            return None
        
        return largest_contour
    
    def create_roi_from_red_object(self, frame, expand_factor=1.5):
        """根据红色物块创建ROI区域"""
        # 检测红色物块
        red_mask = self.detect_red_objects(frame)
        
        # 找到最大的红色轮廓
        red_contour = self.find_largest_red_contour(red_mask)
        
        if red_contour is None:
            print("未检测到足够大的红色物块")
            return False
        
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(red_contour)
        
        # 计算扩展后的ROI区域
        center_x = x + w // 2
        center_y = y + h // 2
        
        new_w = int(w * expand_factor)
        new_h = int(h * expand_factor)
        
        # 确保ROI在图像范围内
        height, width = frame.shape[:2]
        roi_x = max(0, center_x - new_w // 2)
        roi_y = max(0, center_y - new_h // 2)
        roi_w = min(new_w, width - roi_x)
        roi_h = min(new_h, height - roi_y)
        
        # 保存ROI信息
        self.roi_bbox = (roi_x, roi_y, roi_w, roi_h)
        self.roi_contour = red_contour
        
        # 创建ROI掩码
        self.roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(self.roi_mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 255, -1)
        
        print(f"ROI区域创建成功: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
        return True
    
    def apply_roi_to_frame(self, frame):
        """将ROI应用到帧上"""
        if self.roi_mask is None:
            return frame
        
        # 应用ROI掩码
        roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        return roi_frame
    
    def get_roi_region(self, frame):
        """获取ROI区域的图像"""
        if self.roi_bbox is None:
            return None
        
        x, y, w, h = self.roi_bbox
        roi_region = frame[y:y+h, x:x+w]
        return roi_region
    
    def draw_roi_visualization(self, frame):
        """绘制ROI可视化"""
        result = frame.copy()
        
        # 绘制红色物块轮廓
        if self.roi_contour is not None:
            cv2.drawContours(result, [self.roi_contour], -1, (0, 0, 255), 2)
            
            # 计算并显示红色物块的中心点
            M = cv2.moments(self.roi_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(result, "Red Object", (cx-50, cy-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制ROI矩形
        if self.roi_bbox is not None:
            x, y, w, h = self.roi_bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result, "ROI Region", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示ROI信息
            roi_info = f"ROI: {w}x{h} at ({x},{y})"
            cv2.putText(result, roi_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 添加半透明ROI覆盖
        if self.roi_mask is not None:
            overlay = result.copy()
            overlay[self.roi_mask > 0] = [0, 255, 0]  # 绿色覆盖
            cv2.addWeighted(overlay, 0.2, result, 0.8, 0, result)
        
        return result
    
    def run_red_roi_detection(self, camera_id=0):
        """运行红色物块ROI检测"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=== 红色物块ROI检测 ===")
        print("操作说明:")
        print("'SPACE' - 基于当前红色物块创建ROI")
        print("'r' - 重置ROI")
        print("'s' - 保存当前ROI区域图像")
        print("'q' - 退出")
        print("\n请将红色物块放在摄像头前，按SPACE键创建ROI区域")
        
        roi_created = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 实时检测红色物块
            red_mask = self.detect_red_objects(frame)
            red_contour = self.find_largest_red_contour(red_mask)
            
            if not roi_created:
                # 显示红色检测结果
                result = frame.copy()
                
                # 显示红色掩码区域
                red_colored = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
                result = cv2.addWeighted(result, 0.7, red_colored, 0.3, 0)
                
                # 绘制检测到的红色轮廓
                if red_contour is not None:
                    cv2.drawContours(result, [red_contour], -1, (0, 255, 255), 2)
                    area = cv2.contourArea(red_contour)
                    
                    # 显示轮廓信息
                    M = cv2.moments(red_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(result, f"Red Object (Area: {int(area)})", 
                                   (cx-80, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        if area >= 500:
                            cv2.putText(result, "Press SPACE to create ROI", 
                                       (cx-100, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.putText(result, "Detecting Red Objects - Press SPACE when ready", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Red Object Detection', result)
            else:
                # ROI已创建，显示ROI应用效果
                # 创建多窗口显示
                height, width = frame.shape[:2]
                display_width = 300
                display_height = int(height * display_width / width)
                
                # 原图
                frame_small = cv2.resize(frame, (display_width, display_height))
                
                # ROI可视化
                roi_vis = self.draw_roi_visualization(frame)
                roi_vis_small = cv2.resize(roi_vis, (display_width, display_height))
                
                # ROI应用结果
                roi_applied = self.apply_roi_to_frame(frame)
                roi_applied_small = cv2.resize(roi_applied, (display_width, display_height))
                
                # ROI区域单独显示
                roi_region = self.get_roi_region(frame)
                if roi_region is not None and roi_region.size > 0:
                    roi_region_resized = cv2.resize(roi_region, (display_width, display_height))
                else:
                    roi_region_resized = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                
                # 组合显示
                gap = 10
                gap_img = np.zeros((display_height, gap, 3), dtype=np.uint8)
                
                top_row = np.hstack([frame_small, gap_img, roi_vis_small])
                bottom_row = np.hstack([roi_applied_small, gap_img, roi_region_resized])
                
                h_gap = np.zeros((gap, top_row.shape[1], 3), dtype=np.uint8)
                combined = np.vstack([top_row, h_gap, bottom_row])
                
                # 添加标题
                font_scale = 0.5
                cv2.putText(combined, "Original", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                cv2.putText(combined, "ROI Visualization", (display_width + gap + 10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                cv2.putText(combined, "ROI Applied", (10, display_height + gap + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                cv2.putText(combined, "ROI Region", (display_width + gap + 10, display_height + gap + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                
                cv2.imshow('Red ROI Detection Result', combined)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键创建ROI
                if red_contour is not None and cv2.contourArea(red_contour) >= 500:
                    if self.create_roi_from_red_object(frame, expand_factor=1.5):
                        roi_created = True
                        cv2.destroyAllWindows()
                        print("ROI创建成功！现在显示ROI应用效果")
                else:
                    print("未检测到足够大的红色物块，请调整位置后重试")
            elif key == ord('r'):  # 重置ROI
                self.roi_mask = None
                self.roi_contour = None
                self.roi_bbox = None
                roi_created = False
                cv2.destroyAllWindows()
                print("ROI已重置，重新检测红色物块")
            elif key == ord('s') and roi_created:  # 保存ROI区域
                roi_region = self.get_roi_region(frame)
                if roi_region is not None:
                    filename = f"roi_region_{cv2.getTickCount()}.jpg"
                    cv2.imwrite(filename, roi_region)
                    print(f"ROI区域已保存为: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数示例"""
    detector = RedROIDetector()
    
    try:
        # 运行红色物块ROI检测
        detector.run_red_roi_detection(camera_id=0)
        
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

if __name__ == "__main__":
    main()