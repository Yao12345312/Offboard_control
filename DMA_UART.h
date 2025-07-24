#ifndef __DMA_UART_H__
#define __DMA_UART_H__

#include "stm32f10x.h"

#define USART_MAX_LEN 400

extern volatile uint16_t com1_rx_len;
extern volatile uint8_t com1_recv_end_flag;
extern uint8_t com1_rx_buffer[USART_MAX_LEN];


void USART1_Config(u32 bound);//同时配置接收和发送

void USART1_IRQHandler(void);  //串口1中断服务程序

void DMA1_Channel4_IRQHandler(void);

void DMA_USART1_Send(uint8_t *data,u16 size);//串口1DMA发送函数

#endif

