```
<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/logo.png" width="100%" alt="LongLive2.0 logo"/>
</p>

# 🎬 LongLive 2.0: An NVFP4 Parallel Infrastructure for Long Video Generation

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2605.18739)
[![Code](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/NVlabs/LongLive)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=7oQALy32fiU)
[![Models](https://img.shields.io/badge/Model-BF16-yellow)](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B)
[![Models](https://img.shields.io/badge/Model-NVFP4-orange)](https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4)
[![Demo](https://img.shields.io/badge/Demo-Page-brightgreen)](https://nvlabs.github.io/LongLive/LongLive2/)
[![Docs](https://img.shields.io/badge/Full-Documentation-green)](https://nvlabs.github.io/LongLive/LongLive2/docs/)

<div align="center">

<!-- TODO: replace this text block with the final project-page video/demo embed. -->

[![Watch the video](assets/longlive2/first-video-frame.png)](https://www.youtube.com/watch?v=7oQALy32fiU)

</div>

> **Personal fork note:** I'm using this repo to learn about NVFP4 quantization and sequence parallelism. Main branch tracks upstream; my experiments are in the `experiments/` branch.
>
> **Setup note for myself:** On my local machine I need to run `export CUDA_VISIBLE_DEVICES=0,1` before launching any multi-GPU scripts, otherwise it tries to grab all 4 GPUs and OOMs on the 5B model.

## 💡 TLDR: Infra with NVFP4 and parallelism for both training and inference

<p align="center" style="border-radius: 10px">
  <img src="assets/longlive2/teaser.jpg" width="100%" alt="LongLive2.0 teaser"/>
</p>

## News
- 🔥 [2026.05.13] We release **LongLive 2.0**, infra with NVFP4, parallelism and multi-shot for AR training, DMD distillation, and inference (⚡45.7 FPS). The original LongLive 1.0 is now in the [v1.0](https://github.com/NVlabs/LongLive/tree/v1.0) branch.
- 🔥 [2026.04.12] LongLive supports kv cache compression with [TriAttention](https://github.com/WeianMao/triattention), with 50% KV reduction and no quality drop. Check it [here](https://github.com/WeianMao/triattention/tree/main/longlive)
- 🎉 [2026.1.27] LongLive is accepted by **ICLR-2026**.
- 🔥 [2026.1.11] LongLive supports adapting LongLive's original RoPE into KV-cache relative RoPE and generates infinite long videos!
- 🔥 [2025.11.3] We implement LongLive on linear attention model [SANA-Video](https://nvlabs.github.io/Sana/Video/)! Now SANA-Video can generate 60s interactive videos in real-time.
- 🔥 [2025.9.29] We release [Paper](https://arxiv.org/abs/2509.22622), this GitHub repo [LongLive](https://github.com/NVlabs/LongLive) with all training and inference code, the model weight [LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B), and demo page [Website](https://nvlabs.github.io/LongLive).

## Introduction

**LongLive 1.0**: Real-time Interactive Long Video Generation. [You can find it here](https://github.com/NVlabs/LongLive/tree/v1.0) in our V1.0 branch.

**LongLive 2.0**: an NVFP4 Paralle
