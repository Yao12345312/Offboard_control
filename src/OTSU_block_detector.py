import cv2
import numpy as np

class ColorBlockDetector:
    def __init__(self):
        self.image = None
        self.binary_image = None
        
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def otsu_threshold(self, gray_image):
        """使用大津法进行阈值分割"""
        # 计算大津阈值
        threshold_value, binary_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        self.binary_image = binary_image
        return threshold_value, binary_image
    
    def find_color_blocks(self, binary_image):
        """查找色块轮廓"""
        # 形态学操作，去除噪声
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def filter_blocks(self, contours, min_area=100):
        """过滤色块，去除过小的区域"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def detect_blocks_from_frame(self, frame, min_area=100):
        """从单帧图像检测色块"""
        # 预处理
        gray = self.preprocess_image(frame)
        
        # 大津法阈值分割
        threshold_value, binary = self.otsu_threshold(gray)
        
        # 查找色块
        contours = self.find_color_blocks(binary)
        
        # 过滤色块
        filtered_contours = self.filter_blocks(contours, min_area)
        
        return filtered_contours, threshold_value, binary
    
    def draw_results(self, image, contours, threshold_value=None):
        """绘制检测结果"""
        result_image = image.copy()
        
        # 绘制轮廓
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        
        # 添加序号和面积信息
        for i, contour in enumerate(contours):
            # 计算中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 计算面积
                area = cv2.contourArea(contour)
                
                # 添加文字标注
                cv2.putText(result_image, f"{i+1}", (cx-10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(result_image, f"Area:{int(area)}", (cx-30, cy+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 添加统计信息
        info_text = f"Blocks: {len(contours)}"
        if threshold_value is not None:
            info_text += f" | Threshold: {threshold_value:.1f}"
        
        cv2.putText(result_image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result_image
    
    def run_camera_detection(self, camera_id=0, min_area=500, show_processing=False):
        """实时摄像头色块检测
        
        Args:
            camera_id: 摄像头ID (通常为0)
            min_area: 最小色块面积
            show_processing: 是否显示处理过程
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        print("开始实时色块检测，按键控制:")qq
        print("'q' - 退出")
        print("'s' - 切换显示模式")
        print("'+'/'-' - 调整最小面积阈值")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 检测色块
            contours, threshold_value, binary = self.detect_blocks_from_frame(frame, min_area)
            
            # 绘制结果
            result = self.draw_results(frame, contours, threshold_value)
            
            if show_processing:
                # 显示处理过程
                gray = self.preprocess_image(frame)
                
                # 创建组合图像显示
                height, width = frame.shape[:2]
                
                # 调整图像大小以便显示
                display_width = width // 2
                display_height = height // 2
                
                frame_small = cv2.resize(frame, (display_width, display_height))
                gray_small = cv2.resize(gray, (display_width, display_height))
                binary_small = cv2.resize(binary, (display_width, display_height))
                result_small = cv2.resize(result, (display_width, display_height))
                
                # 转换二值图为3通道
                binary_colored = cv2.cvtColor(binary_small, cv2.COLOR_GRAY2BGR)
                gray_colored = cv2.cvtColor(gray_small, cv2.COLOR_GRAY2BGR)
                
                # 组合图像
                top_row = np.hstack([frame_small, gray_colored])
                bottom_row = np.hstack([binary_colored, result_small])
                combined = np.vstack([top_row, bottom_row])
                
                # 添加标题
                cv2.putText(combined, "Original", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Gray", (display_width + 10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Binary", (10, display_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined, "Result", (display_width + 10, display_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Color Block Detection - Processing View', combined)
            else:
                # 只显示结果
                cv2.imshow('Color Block Detection', result)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_processing = not show_processing
                cv2.destroyAllWindows()
            elif key == ord('+') or key == ord('='):
                min_area += 50
                print(f"最小面积阈值: {min_area}")
            elif key == ord('-'):
                min_area = max(50, min_area - 50)
                print(f"最小面积阈值: {min_area}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    detector = ColorBlockDetector()
    
    try:
        # 启动实时摄像头检测
        detector.run_camera_detection(camera_id=0, min_area=500, show_processing=True)
        
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

if __name__ == "__main__":
    main()