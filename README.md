# 数字人对话demo
基于开源ASR、LLM、TTS、THG的数字人对话demo，首包延迟3-5s。

在线demo：https://www.modelscope.cn/studios/AI-ModelScope/video_chat

详细的技术介绍请看[这篇文章](https://mp.weixin.qq.com/s/jpoB8O2IyjhXeAWNWnAj7A)

**中文简体** | [**English**](./docs/README_en.md)

## TODO
- [x] TTS模块添加音色克隆功能
- [ ] TTS模块添加edge-tts
- [ ] LLM模块添加qwen本地推理
- [ ] 链路优化：端到端语音

## 技术选型
* ASR (Automatic Speech Recognition): [FunASR](https://github.com/modelscope/FunASR)
* LLM (Large Language Model): [Qwen](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api)
* TTS (Text to speech): [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice), [edge-tts](https://github.com/rany2/edge-tts)
* THG (Talking Head Generation): [MuseTalk](https://github.com/TMElyralab/MuseTalk/tree/main)
<!-- * 端到端语音LLM: [Mini-Omni](https://github.com/gpt-omni/mini-omni) -->

## 本地部署
### 1. 环境配置

* ubuntu 22.04
* python 3.10
* torch 2.1.2

```bash
$ git lfs install
$ git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git
$ conda create -n metahuman python=3.10
$ conda activate metahuman
$ cd video_chat
$ pip install -r requirement.txt
$ pip install --upgrade gradio # 安装Gradio 5
```

### 2. 权重下载
#### 2.1 创空间下载（推荐）
创空间仓库已设置`git lfs`追踪权重文件，如果是通过`git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git`克隆，则无需额外配置

#### 2.2 手动下载
2.2.1 MuseTalk

参考[这个链接](https://github.com/TMElyralab/MuseTalk/blob/main/README.md#download-weights)

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
2.2.2 GPT-SoVITS

参考[这个链接](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)


### 3. API-KEY
如果需要使用阿里云大模型服务平台百炼提供的[Qwen API](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-thousand-questions/?spm=a2c4g.11186623.0.0.581423edryZ54Q)和[CosyVoice API](https://help.aliyun.com/zh/dashscope/developer-reference/cosyvoice-large-model-for-speech-synthesis/?spm=a2c4g.11186623.0.0.79ce23ednMIj9m)，请在app.py(line 14)中配置API-KEY。

参考[这个链接](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?spm=a2c4g.11186623.0.0.7b7344b7jkORJj)完成API-KEY的获取与配置。

```python
os.environ["DASHSCOPE_API_KEY"] = "INPUT YOUR API-KEY HERE"
```

### 4. 启动服务


```bash
$ python app.py
```
### 5. 使用自定义的数字人形象（可选）
1. 在`/data/video/`中添加录制好的数字人形象视频
2. 修改`/src/thg.py`中`Muse_Talk`类的`avatar_list`，加入`(形象名, bbox_shfit)`，关于bbox_shift的说明参考[这个链接](https://github.com/TMElyralab/MuseTalk?tab=readme-ov-file#use-of-bbox_shift-to-have-adjustable-results)
3. 在`/app.py`中Gradio的avatar_name中加入数字人形象名后重新启动服务，等待完成初始化即可。
