# Hippocampus 入门开发文档

## 概览

### _目录_

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [hippocampus 入门开发文档](#hippocampus-入门开发文档)
  - [1. 开发环境及工具](#1-开发环境及工具)
  - [2. hippocampus 调试](#2-hippocampus-调试)
  - [3. 其它参考](#3-其它参考)

<!-- code_chunk_output -->

## 1. 开发环境及工具

### 1.1 Prerequisite

- CPU: 2 Core
- Memory: 4G
- GPU: Nvidia T4
- Disk: 40G
- OS: Ubuntu 22.04

### 1.2 安装 GPU 驱动

参考 <https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-ubuntu-22-04-lts/>

```bash
apt-get update -y
apt-get upgrade -y

ubuntu-drivers devices
ubuntu-drivers autoinstall

reboot
```

```console
root@t4-lab:~# nvidia-smi
Wed Oct 12 17:55:51 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:07.0 Off |                    0 |
| N/A   60C    P0    30W /  70W |      2MiB / 15360MiB |      5%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 1.3 安装 GPU 容器引擎

参考
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian>

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -y
# 本质是在改 /etc/docker/daemon.json
apt-get install -y nvidia-docker2
systemctl restart docker

# 测试 docker 可以使用 GPU
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

测试 `nvidia-docker` 可以 run cuda sample

```console
root@t4-lab:~# docker run nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
Failed to allocate device vector A (error code CUDA driver version is insufficient for CUDA runtime version)!
[Vector addition of 50000 elements]

root@t4-lab:~# nvidia-docker run nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

## 2. Hippocampus 调试

### 2.1 Hippocampus 开发环境准备

```bash
# 安装 ffmpeg 工具安装，用于产生连续的 RTSP 流，以及推送 RMTP 流到直播音视频网络传输服务（lal 或者 livego）
apt install ffmpeg -y

# 直播音视频网络传输服务 https://pengrl.com/lal/，接收 RMTP 流，输出 HTTP FLV 流
docker run -d -p 1935:1935 -p 8088:8080 --name=lalserver q191201771/lal /lal/bin/lalserver -c /lal/conf/lalserver.conf.json

# 产生连续的 RTSP 流
# 方案 1
# docker run -v ~/Videos/test.mp4:/test.mp4 --rm -p 8554:8554 -d --name=gst-rtsp-launch steabert/gst-rtsp-launch "filesrc location=/test.mp4 ! decodebin ! x264enc ! rtph264pay name=pay0 pt=96"
# 方案 2
# docker run --rm -it -d -e RTSP_PROTOCOLS=tcp -p 8554:8554 -p 8888:8888 aler9/rtsp-simple-server
# docker run --restart always -d -v ~/videos/test.flv:/tmp/workdir/test.flv --name app_ffmpeg jrottenberg/ffmpeg -re -stream_loop -1 -i /tmp/workdir/test.flv -r 25 -c:v libx264 -s 1920x1080 -rtsp_transport tcp -f rtsp rtsp://172.17.0.1:8554/mystream
# ffmpeg -re -stream_loop -1 -i ~/Videos/test.flv -r 25 -c:v libx264 -s 1920x1080 -rtsp_transport tcp -f rtsp rtsp://localhost:8554/mystream
# 这里用了 test.flv，如果你的视频时 mp4 格式，可以直接用 mp4，但性能略差，连续播放也可能失败。可以用 https://www.aconvert.com/cn/video/mp4-to-flv/ 将 mp4 转换成 flv
# 方案三：
docker run -d -e SOURCE_URL=/tmp/workdir/test.mp4 -v ~/Videos/test.mp4:/tmp/workdir/test.mp4 --restart=always --name=rtsp_simulator --net=host registry.cn-shanghai.aliyuncs.com/openv2x/rtsp_simulator:latest
# test.flv和test.mp4 可以在这里下载：链接: https://pan.baidu.com/s/1suMwFqcewUGGEclSj5FZiw 提取码: m2m4
```

至此，RTSP 流应该是可用的，可以用 VLC 测试一下，默认路径是：`rtsp://localhost:8554/mystream` 和
`rtsp://<external-ip>:8554/mystream`。

### 2.2 安装 Python 3.8

当前的 Hippocampus 用 `pytorch < 1.10` 版本，python 必须 `< 3.10`，选用
3.8，安装方式参考：<https://blog.51cto.com/ganzy/5636414>，

```bash
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.8 python3.8-distutils

git clone https://github.com/open-v2x/hippocampus.git
cd hippocampus

# 创建 python 虚拟环境
python3 -m pip install virtualenv
python3 -m virtualenv -p python3.8 .venv

# 或者，可选
# apt install python3.8-venv -y
# apt install python3.8-distutils -y
# apt install python3.8-dev -y
# python3.8 -m pip install virtualenv
# python3.8 -m virtualenv -p python3.8 .venv
```

### 2.3 运行 Hippocampus

```bash
# 进入 python 虚拟环境
. .venv/bin/activate
# 安装依赖
pip install -r requirements.txt
# 启动服务
export rtsp=rtsp://172.17.0.1:8554/mystream
export camera_id=cam_1
python main.py
```

如果 main.py 报错，先检查下 rtsp 流是否正常。

1. 可以通过 VLC 检查 rtsp 流是否正常可以访问
2. 可以看 `rtsp-simple-server` 容器的日志 `docker logs -f rtsp-simple-server`

   正常流收入：

   ```log
   2022/11/05 11:48:41 INF [RTSP] [session 641310733] opened by 172.17.0.1:40388
   2022/11/05 11:48:41 INF [RTSP] [session 641310733] is publishing to path 'mystream', 2 tracks with TCP
   2022/11/05 11:48:51 INF [RTSP] [conn 172.17.0.1:57138] opened
   2022/11/05 11:48:51 INF [RTSP] [session 740411152] opened by 172.17.0.1:57138
   2022/11/05 11:48:51 INF [RTSP] [session 740411152] is reading from path 'mystream', 2 tracks with TCP
   ```

   ffmpeg 推 rtsp 流停止

   ```log
   2022/11/05 12:26:02 INF [RTSP] [conn 172.17.0.1:40388] closed (EOF)
   2022/11/05 12:26:02 INF [RTSP] [session 641310733] closed (not in use)
   2022/11/05 12:26:02 INF [RTSP] [conn 172.17.0.1:57138] closed (terminated)
   2022/11/05 12:26:02 INF [RTSP] [session 740411152] closed (terminated)
   ```

   有 rtsp 流请求，但没有 ffmpeg 推流进来

   ```log
   2022/11/05 12:26:13 INF [RTSP] [conn 172.17.0.1:46222] opened
   2022/11/05 12:26:13 INF [RTSP] [conn 172.17.0.1:46222] closed (no one is publishing to path 'mystream')
   2022/11/05 12:26:20 INF [RTSP] [conn 172.17.0.1:33468] opened
   2022/11/05 12:26:20 INF [RTSP] [conn 172.17.0.1:33468] closed (no one is publishing to path 'mystream')
   2022/11/05 12:26:28 INF [RTSP] [conn 172.17.0.1:57958] opened
   ```

3. 看 hippocampus 和 aaffmpeg 容器（或进程）运行情况，应该保持 running，异常 crash 就要看一下日志

### 2.4 查看视频流

VLC 访问 `rtmp://<external-ip>:1935/live/cam_1.flv`，以及 `http://<external-ip>:8088/live/cam_1.flv`

## 3. 其它参考

### 3.1 ffmpeg 直接通过文件推 RTMP 流到直播服务

参考 <https://zhc3o5gmf9.feishu.cn/docx/doxcnNrfygF4WnWVYhCygILsuqd>

```bash
ffmpeg -re -stream_loop -1 -i /root/test.mp4 -r 25 -c:v libx264 -s 480x270 -f flv rtmp://localhost:1935/live/cam_3
```

vlc 可以访问 <rtmp://localhost:1935/live/result>
