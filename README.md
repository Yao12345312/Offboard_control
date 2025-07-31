## 训练
python train.py --weights yolov5s.pt  --cfg animals/yolov5n.yaml  --data animals/my_voc.yaml --epoch 100 --batch-size 8 --img 640   --device 0

## 导出onnx 
python export.py --weights animals/best.pt --img 640 --batch 1 --include onnx

## onnx转kmodel
# 转换kmodel,onnx模型路径请自定义，生成的kmodel在onnx模型同级目录下
python to_kmodel.py --target k230 --model ./best.onnx --dataset ../test --input_width 640 --input_height 640 --ptq_option 0

# 环境准备
python==3.8.20
torch                        2.0.1+cu118
torchaudio                   2.0.2+cu118
torchvision                  0.15.2+cu118

onnx                         1.12.0
onnxruntime                  1.12.1
onnxsim                      0.4.36

模型转换需要在训练环境安装如下库：

## linux平台：nncase和nncase-kpu可以在线安装，nncase-2.x 需要安装 dotnet-7
sudo apt-get install -y dotnet-sdk-7.0
pip install --upgrade pip
pip install nncase==2.9.0
pip install nncase-kpu==2.9.0

## windows平台：请自行安装dotnet-7并添加环境变量,支持使用pip在线安装nncase，但是nncase-kpu库需要离线安装，在https://github.com/kendryte/nncase/releases下载nncase_kpu-2.*-py2.py3-none-win_amd64.whl
## 进入对应的python环境，在nncase_kpu-2.*-py2.py3-none-win_amd64.whl下载目录下使用pip安装
pip install nncase_kpu-2.*-py2.py3-none-win_amd64.whl

## 除nncase和nncase-kpu外，脚本还用到的其他库包括：
pip install onnx
pip install onnxruntime
pip install onnxsim
下载脚本工具，将模型转换脚本工具 test_yolov5.zip 解压到 yolov5 目录下；

wget https://kendryte-download.canaan-creative.com/developer/k230/yolo_files/test_yolov5.zip
unzip test_yolov5.zip
