|  配件   | 选型  |
|  ----  | ----  |
| 飞控  | 微空H743 |
| 电调  | 微空蓝鸟四合一电调60A |
| 电机  | 朗宇三代X2216 |
| 机架  | HSKRC Q4 |
| 脚架  | 6061铝管 |
| 桨叶  | 8045GF乾丰碳纤尼龙桨 |
| 接收机+遥控  | 富斯ia6B+i6x |
| 激光雷达  | 镭神N10/N10P |
| 机载电脑  | 树莓派5 |
| 机载电脑供电  | 5V5A供电稳压模块P23C |
| 视觉模块  | K230/USB摄像头+opencv |

系统环境：PX4 1.14.3+Ubuntu24.04+ROS2jazzy


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

3. 树莓派4B和树莓派5的串口的对应关系
```
uart0  ->  uart10,     即  /dev/ttyAMA0 -> /dev/ttyAMA10
uart2  ->  uart1-pi5,  即  /dev/ttyAMA2 -> /dev/ttyAMA1
uart3  ->  uart2-pi5,  即  /dev/ttyAMA3 -> /dev/ttyAMA2
uart4  ->  uart3-pi5,  即  /dev/ttyAMA4 -> /dev/ttyAMA3
uart5  ->  uart4-pi5,  即  /dev/ttyAMA5 -> /dev/ttyAMA4
```
<img width="887" height="791" alt="3569f2a5b7e9786ca6e9d7693fdce94" src="https://github.com/user-attachments/assets/c6251606-8598-465a-8a94-9a912aa1e73c" />
<img width="814" height="267" alt="4f901b40965be35f5d96909f9de6946" src="https://github.com/user-attachments/assets/ae255c6f-bdb9-41d7-8312-2cb4f13ebfa1" />
