#include "stm32f10x.h"                  // Device header
#include "SERVO_CONTROL.h"   


float current_angle_x = 90.0f;
float current_angle_y = 90.0f;

float delta=0;

void PWM_TIM2_Init(void)
{
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2, ENABLE);  // 开启TIM2时钟
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE); // PA口时钟

    GPIO_InitTypeDef GPIO_InitStructure;
    TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
    TIM_OCInitTypeDef TIM_OCInitStructure;

    // 1. 配置PA0, PA1为复用推挽输出（TIM2_CH1/CH2）
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0 | GPIO_Pin_1;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // 2. TIM2 定时器设置（PWM周期为20ms = 50Hz）
    // 假设使用72MHz主频
    TIM_TimeBaseStructure.TIM_Period = 20000 - 1;        // 自动重装载值（ARR）= 20ms
    TIM_TimeBaseStructure.TIM_Prescaler = 72 - 1;        // 预分频72，即1MHz计数频率
    TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
    TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
    TIM_TimeBaseInit(TIM2, &TIM_TimeBaseStructure);

    // 3. 输出比较PWM模式配置
    TIM_OCInitStructure.TIM_OCMode = TIM_OCMode_PWM1;     // PWM模式1
    TIM_OCInitStructure.TIM_OutputState = TIM_OutputState_Enable;
    TIM_OCInitStructure.TIM_Pulse = 1500;                 // 初始1.5ms=中位,针对不同舵机需要实测
    TIM_OCInitStructure.TIM_OCPolarity = TIM_OCPolarity_High;

    TIM_OC1Init(TIM2, &TIM_OCInitStructure);  // CH1 = PA0
    TIM_OC2Init(TIM2, &TIM_OCInitStructure);  // CH2 = PA1

    // 4. 使能输出通道和定时器
    TIM_OC1PreloadConfig(TIM2, TIM_OCPreload_Enable);
    TIM_OC2PreloadConfig(TIM2, TIM_OCPreload_Enable);
    TIM_ARRPreloadConfig(TIM2, ENABLE);
    TIM_Cmd(TIM2, ENABLE);
}

float pid_control(PID_Controller *pid, float error)
{
    // 一阶低通滤波：平滑输入误差
    pid->filtered_error = FILTER_FACTOR * error + (1.0f - FILTER_FACTOR) * pid->filtered_error;

    delta = pid->kp * pid->filtered_error + pid->kd * (pid->filtered_error - pid->prev_error);
    pid->prev_error = pid->filtered_error;

    return delta;
}


void Servo_SetAngle(TIM_TypeDef* TIMx, uint8_t channel, float angle)
{
    uint16_t pulse = 500 + (uint16_t)((angle / 180.0f) * 2000);  // SG90 0°~180° → 500us~2500us

    switch(channel) {
        case 1: TIM_SetCompare1(TIMx, pulse); break;
        case 2: TIM_SetCompare2(TIMx, pulse); break;
        case 3: TIM_SetCompare3(TIMx, pulse); break;
        case 4: TIM_SetCompare4(TIMx, pulse); break;
    }
}
