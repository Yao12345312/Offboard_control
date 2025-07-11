- 1.检查所有螺丝，顺序：四个电机的螺丝，电源是否固定好，机架上板16颗螺丝 ，防止震动问题
- 2.检查树莓派是否有报错，如果有供电类的警告，检查电源
- 3.机载电脑输入ros2 topic echo /mavros/estimator_status，检查融合情况
- 4.检查qgc地面站里的 pos horiz参数，检查外部定位数据是否被融合
地面站mavlink终端依次输入，commander status，listener estimator_status
，ekf2 status  检查融合情况以及数据丢失情况，检查各状态是否正常
- 5.使用外部视觉定位时，记得关掉光流定位！！
- 6.飞定高定点offboard时 记得确认距离传感器有没有被线或其他东西挡住
