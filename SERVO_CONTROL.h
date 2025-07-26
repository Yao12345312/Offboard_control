#ifndef __SERVO_CONTROL_H__
#define __SERVO_CONTROL_H__
#include "stm32f10x.h"                  // Device header

#define FILTER_FACTOR    0.15f
#define ANGLE_MIN        0.0f
#define ANGLE_MAX        180.0f
#define CENTER_ANGLE_X  90.0f   // 舵机中位角度
#define CENTER_ANGLE_Y  90.0f

typedef struct {
    float kp;
    float kd;
    float prev_error;
    float filtered_error;
    float output;
} PID_Controller;

extern float current_angle_x;
extern float current_angle_y;

void PWM_TIM2_Init(void);
float pid_control(PID_Controller *pid, float error);
#endif

