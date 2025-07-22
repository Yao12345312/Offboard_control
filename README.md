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
