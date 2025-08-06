from libs.PipeLine import PipeLine, ScopedTiming
from libs.YOLO import YOLOv5
import os,sys,gc
import ulab.numpy as np
import image
from machine import UART
from machine import FPIOA

if __name__=="__main__":
    # 串口设置
    fpioa = FPIOA()
    fpioa.set_function(5, FPIOA.UART2_TXD)  # 设置UART2的TX引脚
    fpioa.set_function(6, FPIOA.UART2_RXD)
    # 串口设置
    u2 = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
    # UART write
    u2.write("UART2 Init successful\n")
    # 显示模式，默认"hdmi",可以选择"hdmi"和"lcd"
    display_mode="lcd"
    rgb888p_size=[1280,720]
    if display_mode=="hdmi":
        display_size=[1920,1080]
    else:
        display_size=[800,480]
    kmodel_path="/data/green_best.kmodel"
    labels = ["elephant","peacock","monkey","tiger","wolf"]
    confidence_threshold = 0.85
    nms_threshold=0.30
    model_input_size=[640,640]
    # 初始化PipeLine
    pl=PipeLine(rgb888p_size=rgb888p_size,display_size=display_size,display_mode=display_mode)
    pl.create(hmirror=True,vflip=True)
    # 初始化YOLOv5实例
    yolo=YOLOv5(task_type="detect",mode="video",kmodel_path=kmodel_path,labels=labels,rgb888p_size=rgb888p_size,model_input_size=model_input_size,display_size=display_size,conf_thresh=confidence_threshold,nms_thresh=nms_threshold,max_boxes_num=3,debug_mode=0)
    yolo.config_preprocess()

    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total",1):
                # 逐帧推理
                img=pl.get_frame()
                res=yolo.run(img)
                #yolo.draw_result(res,pl.osd_img)
                pl.show_image()

                # 初始化检测标志
                has_detection = False

                if res and len(res) == 3:  # 确保res有3个元素
                    bboxes = res[0]  # 边界框数组列表
                    class_ids = res[1]  # 类别ID列表
                    confidences = res[2]  # 置信度列表

                    # 统计每个类别出现的次数
                    class_counts = {i:0 for i in range(len(labels))}
                    for class_id in class_ids:
                        class_counts[class_id] += 1

                    # 遍历所有检测到的对象(最多3个)
                    for i in range(min(len(bboxes), 3)):
                        bbox = bboxes[i]  # 获取边界框
                        class_id = class_ids[i]  # 获取类别ID
                        confidence = confidences[i]  # 获取置信度

                        # 提取坐标 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                        # 计算中心点坐标
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # 设置检测标志
                        has_detection = True

                        # 构造输出字符串
                        output_str = f"C{class_id}X{int(center_x)}Y{int(center_y)}N{class_counts[class_id]}\n"
                        u2.write(output_str)

                # 如果没有检测到任何对象，持续发送C5X0Y0N0
                if not has_detection:
                    u2.write("C5X0Y0N0\n")

                gc.collect()

    except Exception as e:
        sys.print_exception(e)
    finally:
        u2.deinit()
        yolo.deinit()
        pl.destroy()
