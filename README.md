# 聋哑人辅助交流眼镜

## :star:简介

该项目主要是让聋哑人士"听见"外界的声音，同时让普通人群"读懂"手语表达，实现真正意义上的无障碍沟通。

关键技术点如下：
* 开发**快速收集手语训练数据**代码，实现近**10余种手语**数据的收集，每种手语700张图片左右，均保存为标准的yolo格式的数据形式。
* 对手语数据分训练集和测试集，对**yolov5s**模型进行100轮迭代训练。
* 对yolov5模型转为**rk3566上专用的rknn模型**，在orangepi 3B平台上部署，实现**资源的最大化利用**。
* 利用百度**流式实时语音识别**SDK，实现语音识别功能（本地部署资源不够了）。
* 将手势识别和语音识别整合到一起，实现项目的全部应用逻辑。

## :bookmark_tabs:文件说明
* GestureDetect：yolov5项目
  * 1check_gpu.py  # 检查GPU和CUDA是否可用
  * 2check_find_hand.py  # 测试mediapipe库中定位手部位置的功能
  * 3collect_yolo_hand_data.py  # 使用mediapipe库，在摄像头中快速收集手语数据
  * 4pic_gesture_recognition.py  # 测试yolov5s模型，对图片进行检测
  * 5live_gesture_recognition.py  # 测试yolov5s模型，对流式摄像头数据进行检测
  * dataset\：以yolo要求的数据格式存储的训练数据和测试数据
  * runs\train\exp*\weights\：保存训练完成的模型文件
  * runs\train\exp*\weights\last.pt：上一轮训练完毕后产生的权重文件
  * runs\train\exp*\weights\best.pt：最优的权重文件
  * runs\train\exp*\weights\best.onnx：pt模型转为onnx模型
  * 其他文件和文件夹：均为yolov5项目自带
  ---
* rknn-toolkit2-1.5.0：rk公司提供的模型转换工具
  * packages\：对应python版本的库文件
  * examples\onnx\yolov5\：针对yolov5的模型转换工具
  * examples\onnx\yolov5\test.py：官方自带的转化脚本
  * examples\onnx\yolov5\test_own.py：修改过后处理逻辑的转化脚本
  * examples\onnx\yolov5\best.onnx：yolov5项目导出的onnx模型
  * examples\onnx\yolov5\best.rknn：onnx模型经过转换后的rknn模型
  ---
* AssistiveGlasses：部署于orangepi 3B，实现全部应用层逻辑功能
  * main.py # 核心功能脚本
  * best.rknn # 手语识别模型
  * test_*.py # 测试相关功能的文件
  * *.wav/mp3 # 手语对应的语音文件
  * rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl # 需在orangepi上安装的rknn python库



## :eyeglasses:辅助眼镜快速使用说明
注意，使用的镜像为：
`Orangepi3b_1.0.8_ubuntu_focal_desktop_xfce_linux5.10.160.7z`
其他版本可能不支持rknn推理，具体请查阅用户手册

安装conda环境miniconda
请参考：https://blog.csdn.net/air__Heaven/article/details/134808794

`conda create -n glasses python=3.8`
`conda init`
`conda activate glasses`

将整个AssistiveGlasses上传到orangepi 3B中
`cd AssistiveGlasses`
`pip install rknn_toolkit_lite2-1.5.0-cp38-cp38-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple`
`pip install python-opencv tkinter pygame websocket`
`python main.py`
缺啥包就安装啥，直到脚本跑起来就行


## :cyclone:训练与模型转化教程
注意：环境版本非常重要，统一python版本为3.8，兼容性最好
推荐：将该部分代码部署在一台GPU性能比较好的电脑上，可以极大提高训练的速度

### 本地环境中需要做的事情
#### 收集数据
`conda create -n glasses python=3.8`
`conda init`
`conda activate glasses`
`cd GestureDetect`
`python 1check_gpu.py  # 检测GPU和CUDA，报错就安装对应的包`
`python 2check_find_hand.py  # 测试定位手部位置的库`
`python 3collect_yolo_hand_data.py  # 收集yolo数据（记得修改代码中的手势类别）`
确认dataset中的数据符合yolo格式要求，别忘了修改./dataset/dataset.yaml类别


#### 训练模型 best.pt
`python train.py --img 640  --batch 32 --epochs 100  --data ./dataset/dataset.yaml --weights yolov5s.pt --device 0`
其中：
* img选择640最好，因为rknn默认内部运算采用的640*640，运算性能最好
* batch根据实际显存选择，性能高就设置大一些，性能低就设置小一些
* epochs设置为100绝对够了

训练完成的结果，会保存到：
`runs\train\exp*\weights\best.pt`


#### 测试模型性能
`python 5live_gesture_recognition.py`


#### 将pt模型转为onnx模型
`python export.py --weights ./runs/train/exp/weights/best.pt --include onnx --opset 12 --imgsz 640 640`


### docker环境中需要做的事情
rk提供的模型转换工具最好在docker环境中执行，保证本地环境的清洁
首先安装docker
`cd rknn-toolkit2-1.5.0`
创建环境：
`docker run -it --name rknn_env -v ${PWD}:/workspace ubuntu:20.04`
启动环境：
`docker start rknn_env; docker exec -it rknn_env bash`
`cd /workspace/examples/onnx/yolov5`
进入docker环境后，确认安装python3.8
`python -V`
`cd packages`
`python install rknn_toolkit2-1.5.0+1fa95b5c-cp38-cp38-linux_x86_64.whl`
`python install python-opencv`
将待转换的onnx文件放到docker目录中的rknn-toolkit2-1.5.0\examples\onnx\yolov5\
`cd examples\onnx\yolov5\`
执行转化：
`python test_own.py`
执行的结果就是本地生成一个best.rknn

### 接下来
将best.rknn拷贝到orangepi 3B中的项目文件夹下，就可以用起来啦，太棒啦:heart_eyes:


### 可能使用的其他指令：
`查看是否有npu驱动：ls -l /dev/rknpu`
`查看是否有npu驱动：dmesg | grep -i rknpu`
`查看录音设备情况：arecord -l`
`设置GPIO可操作：sudo chmod 777 /sys/class/gpio/export`
`设置GPIO可操作：sudo chmod 777 /sys/class/gpio/unexport`
`设置GPIO可操作：sudo chmod 777  /sys/class/gpio/gpio1/direction`
`设置音量（静音）：alsamixer`
`查看cpu情况：htop`
`查看NPU温度：cat /sys/class/thermal/thermal_zone0/temp`
`查看NPU情况：watch -n 1 sudo cat /sys/kernel/debug/rknpu/load`


