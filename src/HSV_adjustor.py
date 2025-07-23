import cv2
import numpy as np

class HSVThresholdAdjuster:
    def __init__(self):
        # HSV阈值参数
        self.lower_hsv = np.array([0, 50, 50])
        self.upper_hsv = np.array([10, 255, 255])
        
        # 窗口名称
        self.control_window = "HSV Threshold Control"
        self.preview_window = "Real-time Preview"
        
        # 当前帧
        self.current_frame = None
        
        # 预设颜色
        self.color_presets = {
            "Red1": ([0, 50, 50], [10, 255, 255]),
            "Red2": ([170, 50, 50], [179, 255, 255]), 
            "Green": ([40, 50, 50], [80, 255, 255]),
            "Blue": ([100, 50, 50], [130, 255, 255]),
            "Yellow": ([20, 50, 50], [40, 255, 255]),
            "Orange": ([10, 50, 50], [25, 255, 255]),
            "Purple": ([130, 50, 50], [170, 255, 255]),
            "Cyan": ([80, 50, 50], [100, 255, 255]),
            "Pink": ([140, 50, 50], [170, 255, 255]),
            "White": ([0, 0, 200], [179, 30, 255]),
            "Black": ([0, 0, 0], [179, 255, 50])
        }
        
    def create_control_panel(self):
        """创建HSV控制面板"""
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 500, 400)
        
        # 创建滑块
        cv2.createTrackbar('H_min', self.control_window, self.lower_hsv[0], 179, self.on_trackbar_change)
        cv2.createTrackbar('S_min', self.control_window, self.lower_hsv[1], 255, self.on_trackbar_change)
        cv2.createTrackbar('V_min', self.control_window, self.lower_hsv[2], 255, self.on_trackbar_change)
        cv2.createTrackbar('H_max', self.control_window, self.upper_hsv[0], 179, self.on_trackbar_change)
        cv2.createTrackbar('S_max', self.control_window, self.upper_hsv[1], 255, self.on_trackbar_change)
        cv2.createTrackbar('V_max', self.control_window, self.upper_hsv[2], 255, self.on_trackbar_change)
        
        # 预设颜色选择
        preset_names = list(self.color_presets.keys())
        cv2.createTrackbar('Preset', self.control_window, 0, len(preset_names), self.on_preset_change)
        
        # 形态学操作参数
        cv2.createTrackbar('Morphology', self.control_window, 1, 10, self.on_trackbar_change)
        cv2.createTrackbar('Blur', self.control_window, 0, 20, self.on_trackbar_change)
        
    def on_trackbar_change(self, val):
        """滑块值改变时的回调"""
        # 获取当前滑块值
        h_min = cv2.getTrackbarPos('H_min', self.control_window)
        s_min = cv2.getTrackbarPos('S_min', self.control_window)
        v_min = cv2.getTrackbarPos('V_min', self.control_window)
        h_max = cv2.getTrackbarPos('H_max', self.control_window)
        s_max = cv2.getTrackbarPos('S_max', self.control_window)
        v_max = cv2.getTrackbarPos('V_max', self.control_window)
        
        # 更新HSV阈值
        self.lower_hsv = np.array([h_min, s_min, v_min])
        self.upper_hsv = np.array([h_max, s_max, v_max])
        
        # 更新预览
        self.update_preview()
        
    def on_preset_change(self, val):
        """预设颜色改变时的回调"""
        preset_names = list(self.color_presets.keys())
        if 0 <= val < len(preset_names):
            preset_name = preset_names[val]
            lower, upper = self.color_presets[preset_name]
            
            # 更新滑块位置
            cv2.setTrackbarPos('H_min', self.control_window, lower[0])
            cv2.setTrackbarPos('S_min', self.control_window, lower[1])
            cv2.setTrackbarPos('V_min', self.control_window, lower[2])
            cv2.setTrackbarPos('H_max', self.control_window, upper[0])
            cv2.setTrackbarPos('S_max', self.control_window, upper[1])
            cv2.setTrackbarPos('V_max', self.control_window, upper[2])
            
            # 更新HSV值
            self.lower_hsv = np.array(lower)
            self.upper_hsv = np.array(upper)
            
            print(f"应用预设: {preset_name}")
            print(f"HSV范围: {self.lower_hsv} - {self.upper_hsv}")
    
    def apply_hsv_filter(self, frame):
        """应用HSV过滤"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 应用HSV阈值
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # 获取形态学操作参数
        morph_size = cv2.getTrackbarPos('Morphology', self.control_window)
        blur_size = cv2.getTrackbarPos('Blur', self.control_window)
        
        # 应用模糊
        if blur_size > 0:
            blur_kernel = blur_size * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        
        # 形态学操作
        if morph_size > 0:
            kernel = np.ones((morph_size, morph_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 应用掩码
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        return mask, result
    
    def create_control_info_display(self):
        """创建控制信息显示"""
        info_img = np.zeros((400, 500, 3), dtype=np.uint8)
        
        # 标题
        cv2.putText(info_img, "HSV Threshold Adjuster", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 当前HSV值
        cv2.putText(info_img, f"Lower HSV: {self.lower_hsv}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"Upper HSV: {self.upper_hsv}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 控制说明
        y_pos = 140
        instructions = [
            "Controls:",
            "- Adjust sliders for HSV values",
            "- Preset: Select common colors",
            "- Morphology: Noise reduction size",
            "- Blur: Gaussian blur kernel",
            "",
            "Keyboard Commands:",
            "'p' - Print current HSV values",
            "'s' - Save current settings",
            "'l' - Load settings from file",
            "'r' - Reset to default",
            "'q' - Quit application",
            "",
            "Tips:",
            "- H: Hue (color type)",
            "- S: Saturation (color intensity)",
            "- V: Value (brightness)"
        ]
        
        for i, text in enumerate(instructions):
            color = (255, 255, 0) if text.endswith(":") else (255, 255, 255)
            font_scale = 0.5 if text.endswith(":") else 0.4
            thickness = 2 if text.endswith(":") else 1
            
            cv2.putText(info_img, text, (10, y_pos + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return info_img
    
    def update_preview(self):
        """更新预览显示"""
        if self.current_frame is None:
            return
        
        # 应用HSV过滤
        mask, filtered = self.apply_hsv_filter(self.current_frame)
        
        # 创建显示组合 - 增大显示尺寸
        height, width = self.current_frame.shape[:2]
        display_height = 480  # 增大显示高度从300到480
        display_width = int(width * display_height / height)
        
        # 调整图像大小
        original = cv2.resize(self.current_frame, (display_width, display_height))
        mask_colored = cv2.cvtColor(cv2.resize(mask, (display_width, display_height)), cv2.COLOR_GRAY2BGR)
        filtered_resized = cv2.resize(filtered, (display_width, display_height))
        
        # 创建HSV图像用于参考
        hsv_img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        hsv_resized = cv2.resize(hsv_img, (display_width, display_height))
        
        # 组合显示 - 增大间隔
        gap = 10  # 增大间隔从5到10
        gap_img = np.zeros((display_height, gap, 3), dtype=np.uint8)
        
        top_row = np.hstack([original, gap_img, hsv_resized])
        bottom_row = np.hstack([mask_colored, gap_img, filtered_resized])
        
        h_gap = np.zeros((gap, top_row.shape[1], 3), dtype=np.uint8)
        combined = np.vstack([top_row, h_gap, bottom_row])
        
        # 添加标题 - 增大字体
        font_scale = 0.8  # 增大字体从0.6到0.8
        cv2.putText(combined, "Original", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(combined, "HSV", (display_width + gap + 15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(combined, "Mask", (15, display_height + gap + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        cv2.putText(combined, "Filtered", (display_width + gap + 15, display_height + gap + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # 显示当前HSV范围 - 增大字体
        hsv_text = f"HSV: [{self.lower_hsv[0]}-{self.upper_hsv[0]}, {self.lower_hsv[1]}-{self.upper_hsv[1]}, {self.lower_hsv[2]}-{self.upper_hsv[2]}]"
        cv2.putText(combined, hsv_text, (15, combined.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # 增大字体
        
        cv2.imshow(self.preview_window, combined)
    
    def save_settings(self, filename="hsv_settings.txt"):
        """保存HSV设置到文件"""
        try:
            with open(filename, 'w') as f:
                f.write(f"# HSV Threshold Settings\n")
                f.write(f"H_min: {self.lower_hsv[0]}\n")
                f.write(f"S_min: {self.lower_hsv[1]}\n")
                f.write(f"V_min: {self.lower_hsv[2]}\n")
                f.write(f"H_max: {self.upper_hsv[0]}\n")
                f.write(f"S_max: {self.upper_hsv[1]}\n")
                f.write(f"V_max: {self.upper_hsv[2]}\n")
                f.write(f"# Lower HSV: {self.lower_hsv}\n")
                f.write(f"# Upper HSV: {self.upper_hsv}\n")
            print(f"设置已保存到: {filename}")
        except Exception as e:
            print(f"保存失败: {e}")
    
    def load_settings(self, filename="hsv_settings.txt"):
        """从文件加载HSV设置"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            values = {}
            for line in lines:
                if line.startswith('#') or ':' not in line:
                    continue
                key, value = line.strip().split(': ')
                values[key] = int(value)
            
            # 更新滑块
            cv2.setTrackbarPos('H_min', self.control_window, values['H_min'])
            cv2.setTrackbarPos('S_min', self.control_window, values['S_min'])
            cv2.setTrackbarPos('V_min', self.control_window, values['V_min'])
            cv2.setTrackbarPos('H_max', self.control_window, values['H_max'])
            cv2.setTrackbarPos('S_max', self.control_window, values['S_max'])
            cv2.setTrackbarPos('V_max', self.control_window, values['V_max'])
            
            print(f"设置已从 {filename} 加载")
        except Exception as e:
            print(f"加载失败: {e}")
    
    def print_current_values(self):
        """打印当前HSV值"""
        print("\n=== 当前HSV阈值 ===")
        print(f"Lower HSV: {self.lower_hsv}")
        print(f"Upper HSV: {self.upper_hsv}")
        print(f"OpenCV代码:")
        print(f"lower_hsv = np.array({list(self.lower_hsv)})")
        print(f"upper_hsv = np.array({list(self.upper_hsv)})")
        print("==================\n")
    
    def run_camera_adjuster(self, camera_id=0):
        """运行摄像头HSV调整器"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        # 设置摄像头分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 创建窗口
        self.create_control_panel()
        cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
        
        print("=== HSV阈值调整器 ===")
        print("使用滑块调整HSV阈值，实时查看效果")
        print("按键说明:")
        print("'p' - 打印当前HSV值")
        print("'s' - 保存设置")
        print("'l' - 加载设置")
        print("'r' - 重置为默认值")
        print("'q' - 退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            self.current_frame = frame
            
            # 显示控制面板信息
            control_info = self.create_control_info_display()
            cv2.imshow(self.control_window, control_info)
            
            # 更新预览
            self.update_preview()
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.print_current_values()
            elif key == ord('s'):
                self.save_settings()
            elif key == ord('l'):
                self.load_settings()
            elif key == ord('r'):
                # 重置为默认值
                cv2.setTrackbarPos('H_min', self.control_window, 0)
                cv2.setTrackbarPos('S_min', self.control_window, 50)
                cv2.setTrackbarPos('V_min', self.control_window, 50)
                cv2.setTrackbarPos('H_max', self.control_window, 179)
                cv2.setTrackbarPos('S_max', self.control_window, 255)
                cv2.setTrackbarPos('V_max', self.control_window, 255)
                print("已重置为默认值")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    adjuster = HSVThresholdAdjuster()
    
    try:
        # 直接启动摄像头实时调整
        adjuster.run_camera_adjuster(camera_id=0)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()