#ifndef __laser_pointer_control_H__
#define	__laser_pointer_control_H__

void Laser_GPIO_Configuration(void);
void Laser_Control(float cmd[2]);
void Laser_On(void);
void Laser_Off(void);

#endif

