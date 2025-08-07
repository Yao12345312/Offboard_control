### 材料选型
|  配件   | 选型  |
|  ----  | ----  |
| 飞控  | 微空H743-PX4固件 |
| 电调  | 微空蓝鸟四合一电调60A |
| 电机  | 朗宇三代X2216 |
| 机架  | HSKRC Q4 |
| 脚架  | 6061铝管 |
| 桨叶  | 8045GF乾丰碳纤尼龙桨 |
| 电池  | 格式5300mAH 45C |
| 接收机+遥控  | 富斯ia6B+i6x |
| 激光雷达  | 镭神N10/N10P |
| 高度传感器  | 微空MT-01P |
| 机载电脑  | 树莓派5-8G |
| 机载电脑供电  | 5V5A供电稳压模块P23C |
| 视觉模块  | K230/USB摄像头+opencv |

系统环境：PX4 1.14.3+Ubuntu24.04+ROS2 jazzy

### 开机自启动服务设置

开启开机自启动tracked2vision
```
sudo systemctl enable my_auto_boot.service
```
关闭开机自启动脚本
```
sudo systemctl disable my_auto_boot.service
```
修改了.service文件后，重载服务文件
```
sudo systemctl daemon-reload
```

手动开启或关闭服务
```
sudo systemctl start my_auto_boot.service

```
查看某个服务运行日志，检查状态
```
sudo journalctl -u my_auto_boot.service -f
```
**设置自启动后，可以取消程序中自动切换offboard模式的注释，PX4修改对应参数，无需遥控器起飞**


**注意：开机自启程序使用串口时，会产生权限问题，影响其他节点导致初始化失败，办法如下：**

1.运行以下命令：
```
sudo visudo
```
2.在末尾添加一行（注意替换用户名和命令路径，路径改成自己自启动.sh路径）：
```
yourusername ALL=(ALL) NOPASSWD: /path/to/your_script.sh

```
或者也可以：

```
yourusername ALL=(ALL) NOPASSWD: ALL

```
给所有命令执行权限（有一定风险）

### 串口相关设置
树莓派5的/boot/firmware/config.txt要这样改
```
dtoverlay=uart1-pi5
dtoverlay=uart2-pi5
dtoverlay=uart3-pi5
dtoverlay=uart4-pi5
```

这时， 用ls /dev/ttyAMA* 之后，得到的结果是
```
/dev/ttyAMA1  /dev/ttyAMA10  /dev/ttyAMA2  /dev/ttyAMA3  /dev/ttyAMA4
```
多了个/dev/ttyAMA10， 这是树莓派5 debug用的串口。

3. 树莓派4B->树莓派5的串口的对应关系（图示为4B，树莓派5要根据对应关系修改）
```
uart0  ->  uart10,     即  /dev/ttyAMA0 -> /dev/ttyAMA10
uart2  ->  uart1-pi5,  即  /dev/ttyAMA2 -> /dev/ttyAMA1
uart3  ->  uart2-pi5,  即  /dev/ttyAMA3 -> /dev/ttyAMA2
uart4  ->  uart3-pi5,  即  /dev/ttyAMA4 -> /dev/ttyAMA3
uart5  ->  uart4-pi5,  即  /dev/ttyAMA5 -> /dev/ttyAMA4
```
<img width="887" height="791" alt="3569f2a5b7e9786ca6e9d7693fdce94" src="https://github.com/user-attachments/assets/c6251606-8598-465a-8a94-9a912aa1e73c" />
<img width="814" height="267" alt="4f901b40965be35f5d96909f9de6946" src="https://github.com/user-attachments/assets/ae255c6f-bdb9-41d7-8312-2cb4f13ebfa1" />

### 常见注意事项
1，飞前检查机架各个螺丝，脚架有没有松动，检查电池电压

2，固定好飞机各个部件确定桨叶不会打到

3，飞前检查定位数据是否正常，特别注意确认高度传感器有没有被挡住，否则在offboard模式下飞机达不到目标高度会冲天花板

4，遥控器模式切换通道预留一个位置设置land模式，快炸机时直接切到land模式即可，会尝试降落回原点，切手动稳定降落需要练习，不好控制
