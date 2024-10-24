# Digital Human Dialogue Demo
A digital human dialogue demo based on open-source ASR, LLM, TTS, and THG, with an first package latency of 3-5 seconds.

Online demo：https://www.modelscope.cn/studios/AI-ModelScope/video_chat

[**中文简体**](../README.md) | **English**

## Technology Stack
* ASR (Automatic Speech Recognition): [FunASR](https://github.com/modelscope/FunASR)
* LLM (Large Language Model): [Qwen](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api)
* TTS (Text to Speech): [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* THG (Talking Head Generation): [MuseTalk](https://github.com/TMElyralab/MuseTalk/tree/main)

## Local Deployment
### Environment Setup
Using the image version: ubuntu22.04-py310-torch2.1.2-tf2.14.0-modelscope1.14.0

```bash
$ git lfs install
$ git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git
$ conda create -n metahuman python=3.10
$ conda activate metahuman
$ cd video_chat
$ pip install -r requirement.txt
```

### Download Weights
#### 1. Clone from ModelScope (Recommended)
   
The Creative Space repository is already set up to track weight files with `git lfs`. 

If you clone via `git clone https://www.modelscope.cn/studios/AI-ModelScope/video_chat.git`, no additional configuration is required.

#### 2. Manual Download
   
2.1 MuseTalk weights

Reference: https://github.com/TMElyralab/MuseTalk/blob/main/README.md#download-weights

The weights should be organized as follows:
```plaintext
./weights/
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── musetalk
│   ├── musetalk.json
│   └── pytorch_model.bin
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

2.2 GPT-SoVITS weights

Reference: https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B

### Starting the Service
```bash
$ python app.py
```

### Using a Custom Digital Human Avatar
Add the recorded digital human avatar video to the /data/video/ directory.
Modify the avatar_list in the Muse_Talk class in /src/thg.py to include (avatar_name, bbox_shift). For details on bbox_shift, refer to this link.
Add the digital human avatar name to the avatar_name field in Gradio within app.py, then restart the service and wait for initialization to complete.

## TODO
1. Voice cloning ✅
2. Pipeline optimization (end-to-end voice API)