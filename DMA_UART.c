#include "string.h" 
#include "DMA_UART.h"

volatile uint16_t com1_rx_len = 0;  //接收帧数据的长度
volatile uint8_t com1_recv_end_flag = 0; //帧数据接收完成标
uint8_t com1_rx_buffer[USART_MAX_LEN]={0};//接收数据缓存
uint8_t DMA_USART1_TX_BUF[400]; //发送数据缓存

void USART1_Config(u32 bound)//同时配置接收和发送,传入串口波特率
{
    GPIO_InitTypeDef  GPIO_InitStructure;
    USART_InitTypeDef USART_InitStructure;
    NVIC_InitTypeDef  NVIC_InitStructure;
    //1时钟
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1|RCC_APB2Periph_GPIOA, ENABLE);
    //2GPIO USART1_TX   GPIOA.9
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9; //PA.9
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;	//复用推挽输出
    GPIO_Init(GPIOA, &GPIO_InitStructure);//初始化GPIOA.9
    //USART1_RX	  GPIOA.10初始化
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;//PA10
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;//浮空输入
    GPIO_Init(GPIOA, &GPIO_InitStructure);//初始化GPIOA.10
    USART_DeInit(USART1);
    //3中断  NVIC 配置
    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority=2 ;//抢占优先级2
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;		//子优先级1
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;			//IRQ通道使能
    NVIC_Init(&NVIC_InitStructure);	//根据指定的参数初始化VIC寄存器

    NVIC_InitStructure.NVIC_IRQChannel = DMA1_Channel4_IRQn; //嵌套通道为DMA1_Channel4_IRQn
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2; //抢占优先级为 2
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 4+3; //响应优先级为 7
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE; //通道中断使能
    NVIC_Init(&NVIC_InitStructure);

    NVIC_InitStructure.NVIC_IRQChannel = DMA1_Channel5_IRQn ;//串口1发送中断通道
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;     //抢占优先级
    NVIC_InitStructure.NVIC_IRQChannelSubPriority        = 5+3;	  //子优先级
    NVIC_InitStructure.NVIC_IRQChannelCmd                = ENABLE;	//IRQ通道使能
    NVIC_Init(&NVIC_InitStructure);
    //4配置 USART设置
    USART_InitStructure.USART_BaudRate = bound;//串口波特率
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;//字长为8位数据格式
    USART_InitStructure.USART_StopBits = USART_StopBits_1;//一个停止位
    USART_InitStructure.USART_Parity = USART_Parity_No;//无奇偶校验位
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;//无硬件数据流控制
    USART_InitStructure.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;	//收发模式

    USART_Init(USART1, &USART_InitStructure); //初始化串口1

    USART_ITConfig(USART1, USART_IT_IDLE, ENABLE);
    USART_DMACmd(USART1,USART_DMAReq_Rx,ENABLE);
    USART_DMACmd(USART1,USART_DMAReq_Tx,ENABLE);
    USART_Cmd(USART1,ENABLE);

    DMA_InitTypeDef    DMA_Initstructure;
    /*开启DMA时钟*/
    RCC_AHBPeriphClockCmd(RCC_AHBPeriph_DMA1,ENABLE);
    DMA_DeInit(DMA1_Channel5); 
    /*DMA配置*/
    DMA_Initstructure.DMA_PeripheralBaseAddr =  (u32)(&USART1->DR);;
    DMA_Initstructure.DMA_MemoryBaseAddr     = (u32)com1_rx_buffer;
    DMA_Initstructure.DMA_DIR = DMA_DIR_PeripheralSRC;
    DMA_Initstructure.DMA_BufferSize = USART_MAX_LEN;
    DMA_Initstructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_Initstructure.DMA_MemoryInc =DMA_MemoryInc_Enable;
    DMA_Initstructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
    DMA_Initstructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
    DMA_Initstructure.DMA_Mode = DMA_Mode_Normal;
    DMA_Initstructure.DMA_Priority = DMA_Priority_High;
    DMA_Initstructure.DMA_M2M = DMA_M2M_Disable;
    DMA_Init(DMA1_Channel5,&DMA_Initstructure);
    DMA_InitTypeDef DMA_InitStructure;
	DMA_DeInit(DMA1_Channel4);    //初始化DMA1
	/* 配置DMA1 USART1发送 */
	DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)(&USART1->DR);
	DMA_InitStructure.DMA_MemoryBaseAddr = (uint32_t)DMA_USART1_TX_BUF;
	DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralDST;
	DMA_InitStructure.DMA_BufferSize = 0;
	DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
	DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;
	DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
	DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
	DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
	DMA_InitStructure.DMA_Priority = DMA_Priority_VeryHigh;
	DMA_InitStructure.DMA_M2M = DMA_M2M_Disable;
	DMA_Init(DMA1_Channel4, &DMA_InitStructure);//初始化
    //启动DMA
    DMA_Cmd(DMA1_Channel4,ENABLE); //TX
    //开启DMA发送发成中断
    DMA_ITConfig(DMA1_Channel4,DMA_IT_TC,ENABLE);

    DMA_ITConfig(DMA1_Channel5,  DMA_IT_TC, ENABLE);
    USART_DMACmd(USART1,USART_DMAReq_Rx,ENABLE);
    DMA_Cmd(DMA1_Channel5,ENABLE); //RX
}

void USART1_IRQHandler(void)  //串口1中断服务程序
{
    /* 使用串口DMA空闲接收 */
    if(USART_GetITStatus(USART1,USART_IT_IDLE)!=RESET) 	//空闲中断触发
    {
    	com1_recv_end_flag=1;	//接收完成标志
        DMA_Cmd(DMA1_Channel5, DISABLE); /* 暂时关闭dma，数据尚未处理 */
        com1_rx_len = USART_MAX_LEN - DMA_GetCurrDataCounter(DMA1_Channel5);/* 获取接收到的数据长度 单位为字节*/
        USART_ClearITPendingBit(USART1,USART_IT_IDLE);
        DMA_SetCurrDataCounter(DMA1_Channel5,USART_MAX_LEN);/* 重新赋值计数值，必须大于等于最大可能接收到的数据帧数目 */
        DMA_Cmd(DMA1_Channel5, ENABLE);   /*打开DMA*/
    	USART_ReceiveData(USART1);//清除空闲中断标志位（接收函数有清标志位的作用）
    }
    /* 检查DMA发送完成，关闭TC标志位 */
  	if(USART_GetFlagStatus(USART1,USART_IT_TXE)==RESET)	//串口发送完成
  	{
    	USART_ITConfig(USART1,USART_IT_TC,DISABLE);
 	}
}

void DMA1_Channel4_IRQHandler(void)
{
	if(DMA_GetITStatus(DMA1_IT_TC4))
	{
		DMA_ClearITPendingBit(DMA1_IT_TC4);  // 清除传输完成中断标志位
   		DMA_Cmd(DMA1_Channel4,DISABLE);
   		DMA1_Channel4->CNDTR=0;         // 清除数据长度
    	USART_ITConfig(USART1,USART_IT_TC,ENABLE); //打开串口发送完成中断
	}
}
void DMA_USART1_Send(uint8_t *data,u16 size)//串口1DMA发送函数
{
	DMA_Cmd(DMA1_Channel4, DISABLE);
	memcpy(DMA_USART1_TX_BUF, data, size);
	while (DMA_GetCurrDataCounter(DMA1_Channel4));  // 检查DMA发送通道内是否还有数据
	DMA_SetCurrDataCounter(DMA1_Channel4, size);   // 重新写入要传输的数据数量
	DMA_Cmd(DMA1_Channel4, ENABLE);     // 启动DMA发送
}
