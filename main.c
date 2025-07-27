#include "stm32f10x.h"
#include "DMA_UART.h"
#include "SERVO_CONTROL.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stm32f10x_it.h>
// PID 控制器参数
PID_Controller pid_x = { .kp = 80.0f, .kd = 40.0f, .prev_error = 0.0f, .filtered_error = 0.0f };
PID_Controller pid_y = { .kp = 80.0f, .kd = 40.0f, .prev_error = 0.0f, .filtered_error = 0.0f };

float delta_angle_x=0;
float delta_angle_y=0;
// 节流控制
uint32_t last_control_time = 0;
#define CONTROL_INTERVAL_MS 50  // 控制间隔：50ms

int main(void)
{
    // 初始化串口和PWM
    USART1_Config(115200);
    PWM_TIM2_Init();
	init_millis_timer();

    while (1)
    {
        // 当前时间
        uint32_t now = millis();

        if (com1_recv_end_flag && (now - last_control_time) >= CONTROL_INTERVAL_MS)
        {
            com1_recv_end_flag = 0;
            com1_rx_buffer[com1_rx_len] = '\0';

            float offset_x = 0.0f, offset_y = 0.0f;
            sscanf((char *)com1_rx_buffer, "%f,%f", &offset_x, &offset_y);

            // ---------- PID 控制 ----------
            delta_angle_x = pid_control(&pid_x, offset_x);
            delta_angle_y = pid_control(&pid_y, offset_y);

            current_angle_x += delta_angle_x;
            current_angle_y += delta_angle_y;

            // 限幅
            if (current_angle_x < ANGLE_MIN) current_angle_x = ANGLE_MIN;
            if (current_angle_x > ANGLE_MAX) current_angle_x = ANGLE_MAX;
            if (current_angle_y < ANGLE_MIN) current_angle_y = ANGLE_MIN;
            if (current_angle_y > ANGLE_MAX) current_angle_y = ANGLE_MAX;

            // 驱动舵机
            Servo_SetAngle(TIM2, 1, current_angle_x);
            Servo_SetAngle(TIM2, 2, current_angle_y);

            // 更新时间戳 & 清空缓冲
            last_control_time = now;
            memset(com1_rx_buffer, 0, sizeof(com1_rx_buffer));
            com1_rx_len = 0;
        }
    }
}
