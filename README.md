<h1>数字人对话demo</h1>
基于开源ASR、LLM、TTS、THG的数字人对话demo，首包延迟3-5s，持续更新中。<br><br>

在线demo：https://www.modelscope.cn/studios/AI-ModelScope/video_chat

**中文简体** | [**English**](./docs/README_en.md)

</div>

## 技术选型
* ASR (Automatic Speech Recognition): [FunASR](https://github.com/modelscope/FunASR)
* LLM (Large Language Model): [Qwen](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api)
* TTS (Text to speech): [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* THG (Talking Head Generation): [MuseTalk](https://github.com/TMElyralab/MuseTalk/tree/main)

## 本地部署
### 环境配置
使用镜像版本：ubuntu22.04-py310-torch2.1.2-tf2.14.0-modelscope1.14.0

```bash
$ git lfs install
$ git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git
$ conda create -n metahuman python=3.10
$ conda activate metahuman
$ cd video_chat
$ pip install -r requirement.txt
```

### 权重下载
#### 1. 创空间下载（推荐）
创空间仓库已设置`git lfs`追踪权重文件，如果是通过`git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git`克隆，则无需额外配置

#### 2. 手动下载
2.1 MuseTalk

参考：https://github.com/TMElyralab/MuseTalk/blob/main/README.md#download-weights

目录如下：
``` plaintext
./weights/
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── musetalk
│   ├── musetalk.json
│   └── pytorch_model.bin
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```
2.2 GPT-SoVITS

参考：https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B



### 启动服务
```bash
$ python app.py
```
### 使用自定义的数字人形象
1. 在`/data/video/`中添加录制好的数字人形象视频
2. 修改`/src/thg.py`中`Muse_Talk`类的`avatar_list`，加入`(形象名, bbox_shfit)`，关于bbox_shift的说明参考[这个链接](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#use-of-bbox_shift-to-have-adjustable-results)
3. 在`/app.py`中Gradio的avatar_name中加入数字人形象名后重新启动服务，等待完成初始化即可。

## TODO
1. 音色克隆
2. 链路优化（端到端语音API）