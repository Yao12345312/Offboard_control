# 24 ground_station with CM4 IO board,or rasbarry PI5
## 此地面站方案优势

使用ROS2话题通信，通信量远大于无线串口，配置简单，数据格式多样，不同搭载ROS2的主机配置主从IP后，在同一局域网内话题自动共享

## 存在问题
通信的前提是从机连上主机的热点，但是现场不能确定什么时候连上，可以在外部添加连接成功的标志或者定时发送通信数据

## 近几年题目均涉及到地面站图形化设计，我们使用陶晶驰串口屏，需要注意以下几点

1，使用绘图控件绘制的图会在跳转其他page时消失

2，涉及算法或者其他处理的结果需要以图片形式展示时，可以使用外部图片控件，但是需要注意X2系列解析外部图片会有问题（基本用不了），最好选用X5


## 使用设备
树莓派5（配置为客户端模式，连接热点），树莓派4b（配置为热点模式）
## 树莓派配置为热点模式
1) 安装必要的软件
```
  sudo apt update
  sudo apt install hostapd dnsmasq
```
2）编辑/etc/hostapd/hostapd.conf，设置WiFi热点的参数
```
sudo nano /etc/hostapd/hostapd.conf

```
3)在文件中添加以下内容,根据需求修改
```
interface=wlan0
driver=nl80211
ssid=Pi_Hotspot        # WiFi的名称
hw_mode=g
channel=7
auth_algs=1
wpa=2
wpa_passphrase=raspberry # 热点密码
wpa_key_mgmt=WPA2-PSK
rsn_pairwise=CCMP

```
4)配置hostapd服务,编辑/etc/default/hostapd，指定配置文件路径
```
sudo nano /etc/default/hostapd

添加或修改 :DAEMON_CONF="/etc/hostapd/hostapd.conf"
```
5)配置dnsmasq（DHCP服务）,编辑/etc/dnsmasq.conf，配置DHCP服务器：
```
sudo nano /etc/dnsmasq.conf
添加以下内容：
interface=wlan0      # 使用的网卡
dhcp-range=192.168.50.10,192.168.50.50,255.255.255.0,24h

```
6)启用IP转发,编辑/etc/sysctl.conf，启用IP转发
```
sudo nano /etc/sysctl.conf
取消以下行的注释：
net.ipv4.ip_forward=1
加载配置:
sudo sysctl -p

```

7)配置NAT转发,使得通过热点连接的设备可以访问互联网。在/etc/rc.local中添加以下行（（不联网可以忽略)
```
sudo nano /etc/rc.local

在exit 0之前添加:
iptables --t nat -A POSTROUTING -o eth0 -j MASQUERADE

如果使用的是WiFi连接外部网络，eth0改为wlan0，或者根据需要设置不同的接口
```
8)配置DHCP服务器转发
```
sudo apt install dhcpcd5
sudo nano /etc/dhcpcd.conf
```
在末尾添加：
```
interface wlan0
static ip_address=192.168.50.1/24  # 设置静态 IP 地址
static routers=192.168.50.1        # 设置网关
static domain_name_servers=8.8.8.8  # 设置 DNS 服务器

```

)启动hostapd和dnsmasq和dhcpcd
```
sudo systemctl start hostapd
sudo systemctl start dnsmasq
sudo systemctl start dhcpcd

```

)让hostapd和dnsmasq开机启动：
```
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq
sudo systemctl enable dhcpcd
```

)查看主从机ip地址是否在一个子网
```
sudo apt install iw
iw dev wlan0 info
```

##另一个树莓派连接热点
1)使用nmcli命令连接到WiFi热点:
```
nmcli device wifi connect Pi_Hotspot password raspberry
```
2）使用ping测试网络连接,确认在同一个子网内
```
ping 192.168.50.1  #
```

## 树莓派从热点模式回退客户端模式方法：
1)已经配置了hostapd和dnsmasq来创建WiFi热点，首先需要停止这些服务
```
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq
```
2)确保下次启动时树莓派不会自动进入热点模式，你需要禁用hostapd和dnsmasq的自启动
```
sudo systemctl disable hostapd
sudo systemctl disable dnsmasq
```
3)重启wifi服务或直接重启树莓派
```
sudo wpa_cli reconfigure

reboot
```

## 相关冲突项设置
```
sudo systemctl stop NetworkManager
sudo systemctl disable NetworkManager
sudo systemctl stop wpa_supplicant
sudo systemctl disable wpa_supplicant
```
若要用回客户端模式，diaable改为enable即可开机自启

若要启用ros2在同一局域网内话题互通功能，需要关闭防火墙
```
sudo ufw disable
```
当一个树莓派开启热点时，设置ros2主从机，写进客户端机载电脑的bashrc
```
sudo gedit ~/.bashrc
export ROS_DOMAIN_ID=30   # 选择一个域 ID（可以是任意值，主从机设置成一样即可）
export ROS_MASTER_URI=http://192.168.43.1:11311
 # 设置为 主机 的 IP 地址（通过前面的iw dev wlan0 info查询)
```

## 25H_groundstation.py使用绘图控件显示轨迹，25H_groundstation2.py使用外部图片控件显示轨迹
