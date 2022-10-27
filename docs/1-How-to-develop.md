# hippocampus 入门开发文档

## 概览

### _目录_

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [hippocampus入门开发文档](#hippocampus-入门开发文档)
  - [1、开发环境及工具](#开发环境及工具)
  - [2、hippocampus 部署](#hippocampus-部署)
  - [3、本地调试](#本地调试)

<!-- code_chunk_output -->

## 1、开发环境及工具

1. 安装 GPU 驱动

1.1 参考 <https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-ubuntu-22-04-lts/>

```bash
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
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

1.2. 参考 <https://zhc3o5gmf9.feishu.cn/docx/doxcndHJNw5SGOeAh4SxOiwcRSf>

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/libnvidia-container.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

apt install docker.io -y
```

1.3. 修改 docker 配置文件 /etc/docker/daemon.json

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

1.4. 启动 docker

```bash
sudo systemctl restart docker
```

1.5. 测试 docker 可以使用 GPU

```console
root@t4-lab:~# docker run nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
Unable to find image 'nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2' locally
vectoradd-cuda10.2: Pulling from nvidia/k8s/cuda-sample
171857c49d0f: Pull complete 
419640447d26: Pull complete 
61e52f862619: Pull complete 
c118dad7e37a: Pull complete 
29c091e4be16: Pull complete 
d85c81a4428d: Pull complete 
13463320fb92: Pull complete 
a0a71ed83844: Pull complete 
Digest: sha256:4593078cdb8e786d35566faa2b84da1123acea42f0d4099e84e2af0448724af1
Status: Downloaded newer image for nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda10.2
[Vector addition of 50000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 196 blocks of 256 threads
Copy output data from the CUDA device to the host memory
Test PASSED
Done
```

## 2、 hippocampus 部署

1. hippocampus 开发环境准备

```bash
# ffmpeg 推流工具安装
apt install ffmpeg -y
# docker 启动livego
docker run -it -p 1935:1935 -p 8088:8080 --name=lalserver q191201771/lal /lal/bin/lalserver -c /lal/conf/lalserver.conf.json
```

2. 参考 <https://blog.51cto.com/ganzy/5636414>，现在 hippocampus 用 pytorch < 1.10 版本，python 必须 < 3.10

```bash
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.8 -y

git clone https://github.com/open-v2x/hippocampus.git
cd hippocampus

# 创建 python 虚拟环境
python3 -m virtualenv -p python3.8 .venv

# 或者，可选
# apt install python3.8-venv -y
# apt install python3.8-distutils -y
# apt install python3.8-dev -y
# python3.8 -m pip install virtualenv
# python3.8 -m virtualenv -p python3.8 .venv
```

3. hipppcampus 运行

```bash
# 进入 python 虚拟环境
. .venv/bin/activate
# 安装依赖
pip install -r requirements.txt
# 启动服务
python main.py
```

## 3、本地调试

1. ffmpeg 推流调试

参考 <https://zhc3o5gmf9.feishu.cn/docx/doxcnNrfygF4WnWVYhCygILsuqd>

```bash
ffmpeg -re -stream_loop -1 -i /root/test.mp4 -r 25 -c:v libx264 -s 480x270 -f flv rtmp://localhost:1935/live/cam_3
```

vlc 可以访问 <rtmp://localhost:1935/live/result>

2. hippocampus 算法处理 test.mp4 文件并进行推流调试

hippocampus 运行起来，vlc或者网页 访问 <http://localhost:7001/live/cam_1.flv>
