#include "stm32f10x.h"                  // Device header

void Laser_delay_ms(uint32_t ms)
{
    for (uint32_t i = 0; i < ms * 8000; i++)
        __NOP();
}
//GPIO推挽输出
void Laser_GPIO_Configuration(void)
{	//配置时钟
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	
    GPIO_InitTypeDef GPIO_InitStructure;

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
}


void Laser_On(void)
{
    GPIO_SetBits(GPIOA, GPIO_Pin_0);
}

void Laser_Off(void)
{
    GPIO_ResetBits(GPIOA, GPIO_Pin_0);
}
//传参：[1,0.5] 代表点亮激光笔保持0.5s
void Laser_Control(float cmd[2])
{
    if (cmd[0] == 1.0f)
    {
        Laser_On();
    }
    else
    {
        Laser_Off();
    }

    // 延迟控制时间（单位秒 -> 毫秒）
    uint32_t delay_time_ms = (uint32_t)(cmd[1] * 1000.0f);
    Laser_delay_ms(delay_time_ms);

    // 自动熄灭激光（也可以选择不自动熄灭）
    Laser_Off();
}

