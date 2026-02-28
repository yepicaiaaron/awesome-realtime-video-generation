# Awesome Real-Time Video Generation ⚡

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
![GitHub stars](https://img.shields.io/github/stars/yepicaiaaron/awesome-realtime-video-generation?style=social)
![GitHub forks](https://img.shields.io/github/forks/yepicaiaaron/awesome-realtime-video-generation?style=social)

A curated list of **real-time and near real-time video generation models** focused on production deployment and low-latency inference. If it can't run fast enough for practical applications, it's not here.

> **Real-Time Definition**: Video generation with end-to-end latency enabling interactive use cases — typically <5 seconds for short clips on consumer/datacenter GPUs, or streaming-capable architectures.

## 🎯 Why This List?

Most video generation research prioritizes quality over speed. This repo focuses on the inverse: **models fast enough to ship**. Curated for:
- Production engineers building video applications
- Researchers optimizing inference pipelines
- Developers evaluating deployment feasibility
- Teams scaling video generation at cost

---

## 📊 Real-Time Model Leaderboard

### 🌊 True Streaming / Real-Time
*Models that generate and display frames continuously as they are computed.*

| Model | TTFF (Time to First Frame) | FPS (Throughput) | Hardware | Status |
|-------|----------------------------|------------------|----------|--------|
| [MotionStream](#motionstream) | **<200ms** | 29 | H100 | ✅ Production |
| [MonarchRT](#monarchrt) | **<300ms** | 16 | RTX 5090 (32GB) | ✅ Production |
| [LiveTalk](#livetalk) | **<100ms** | 30+ | RTX 4090 | ✅ Production |
| [MemFlow](#memflow) | **<500ms** | 18.7 | H100 (80GB) | 🔬 Research |
| [StreamDiT](#streamdit) | **<400ms** | 16 | RTX 4090 | ✅ Production |
| [S2DiT](#s2dit) | **<1s** | 10 | iPhone 17 Pro | 🔬 Research |
| [CausVid](#causvid) | **1.3s** | 24 | H100 (80GB) | 🔬 CVPR 2025 |
| [EchoTorrent](#echotorrent) | **Streaming** | - | - | 🔬 Research |


### 📦 Fast Batch / Near Real-Time
*Models that generate entire clips or chunks at once with low latency.*

| Model | Latency (per 5s clip) | FPS (Effective) | Hardware | Status |
|-------|-----------------------|-----------------|----------|--------|
| [LTXVideo](#ltxvideo) | **~8s** | 24 | RTX A6000 (48GB) | ✅ Production |
| [AnimateDiff-Lightning](#animatediff-lightning) | ~10s | 16 | RTX 4090 (24GB) | ✅ Production |
| [FastVideo](#fastvideo) | ~12s | 30 | H100 (80GB) | ✅ Production |
| [Pyramidal Flow](#pyramidal-flow) | ~15s | 24 | A100 (80GB) | ✅ Production |
| [VideoLCM](#videolcm) | ~18s | 24 | H100 (80GB) | 🔬 Research |
| [SVD-XT (Turbo)](#stable-video-diffusion-turbo) | ~20s | 25 | A100 (40GB) | ✅ Production |

*Benchmarks measured on single GPU inference, T2V generation. Image-to-video typically 30-50% faster.*

---

## 🚀 Models by Category

### ⚡ General Real-Time Video Generation
*Standard Text-to-Video (T2V), Image-to-Video (I2V), and Streaming models focusing on scene generation and action.*

#### MonarchRT
**Developer**: Monarch AI | **Released**: Feb 2026
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 16 FPS continuous generation
- **Hardware**: Optimized for RTX 5090 (32GB)
- **Architecture**: Distilled DiT with sparse attention
- **Key Innovation**: "Sparse Monarch matrices replace quadratic attention."
- **Best For**: Local real-time generation, gaming integration
- **Deployment**: TensorRT-LLM optimized backend
- **Links**: [Code coming soon] | [Paper coming soon]

---

#### MotionStream
**Developer**: ByteDance | **Released**: Nov 2025
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 29 FPS streaming
- **Hardware**: H100 (80GB)
- **Architecture**: Streaming flow matching
- **Key Innovation**: "Distilled causal student."
- **Best For**: Interactive video applications
- **Deployment**: Server-side streaming
- **Links**: [GitHub](https://github.com/alex4727/MotionStream) | [Paper](https://arxiv.org/abs/2511.01266)

---

#### S2DiT
**Developer**: Apple/Columbia | **Released**: Jan 2026
**Speed**: ⚡⚡⚡ | **Quality**: ⭐⭐⭐

- **Latency**: 10 FPS on mobile
- **Hardware**: iPhone 17 Pro / Mobile NPUs
- **Architecture**: Mobile-optimized DiT (Shift-2-DiT)
- **Key Innovation**: "Sandwich architecture for mobile streaming."
- **Best For**: Mobile apps, privacy-focused generation
- **Deployment**: CoreML
- **Links**: [Code coming soon] | [Paper coming soon]

---

#### MemFlow: Flowing Adaptive Memory
**Developer**: University of Oxford / Meta | **Released**: Dec 2025
**Category**: Streaming / Interactive
**Speed**: ⚡⚡⚡⚡ (18.7 FPS on H100) | **Quality**: ⭐⭐⭐⭐

- **Description**: "Streaming video generation framework for long-context consistency. Retrieves relevant historical frames dynamically to prevent forgetting."
- **Hardware**: H100 (80GB)
- **Key Innovation**: Dynamic memory retrieval for infinite-context streaming
- **Best For**: Long-form video generation, interactive narratives
- **Links**: [Paper](https://arxiv.org/abs/2512.14699) | [Project](https://sihuiji.github.io/MemFlow.github.io/)

---

#### StreamDiT
**Developer**: NUS | **Released**: July 2025
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐

- **Latency**: 16 FPS streaming
- **Hardware**: RTX 4090
- **Architecture**: Diffusion Transformer with streaming cache
- **Key Innovation**: Efficient caching for continuous generation
- **Best For**: Live creative tools
- **Deployment**: PyTorch, Diffusers
- **Links**: [Project Page](https://cumulo-autumn.github.io/StreamDiT/) | [Paper](https://arxiv.org/abs/2507.03745)

---

#### LTXVideo 2.0 (LTX-2)
**Developer**: Lightricks | **Released**: Feb 2026
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

❤️ **Maintainer's Pick**: The successor to LTXVideo adds native audio synchronization and higher resolution (up to 4K) while maintaining blistering speed.

- **Latency**: 6-10s for short clips @ 1080p, 25fps (Fast mode)
- **Hardware**: Runs on RTX 4090 (24GB), improved efficiency
- **Key Innovation**: Unified audio-video latent space, "Fast" and "Pro" modes
- **Best For**: Complete AV production, music videos, real-time previews
- **Deployment**: ComfyUI (official nodes), HuggingFace weights available
- **Links**: [HuggingFace](https://huggingface.co/Lightricks/LTX-Video-2) | [Demo](https://maxvideoai.com/models/ltx-2)


#### LTXVideo (v1)
**Developer**: Lightricks | **Released**: Oct 2024  
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐

❤️ **Maintainer's Pick**: We love this model because it produces excellent quality videos with surprisingly small amounts of compute (runs on consumer hardware).

- **Latency**: 8-12s for 5s clips @ 768x512, 24fps
- **Hardware**: Runs on RTX A6000 (48GB) / 4090 (24GB with optimizations)
- **Architecture**: Latent diffusion with ultra-efficient temporal layers
- **Key Innovation**: DiT-based with compressed latent space (128:1 ratio)
- **Best For**: Rapid prototyping, social media clips, real-time previews
- **Deployment**: ComfyUI integration, Docker containers available
- **Links**: [GitHub](https://github.com/Lightricks/LTX-Video) | [HuggingFace](https://huggingface.co/Lightricks/LTX-Video)

```bash
# Quick start
git clone https://github.com/Lightricks/LTX-Video
cd LTX-Video
pip install -r requirements.txt
python generate.py --prompt "A cat playing piano" --steps 8
```

---

#### FastVideo
**Developer**: HAO AI Lab (Berkeley) | **Released**: Feb 2025  
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 12-18s for 5s clips @ 720p, 30fps
- **Hardware**: H100 (80GB) recommended, A100 compatible
- **Architecture**: Unified inference + post-training acceleration framework
- **Key Innovation**: Dynamic caching, progressive distillation
- **Best For**: Production pipelines requiring quality-speed balance
- **Deployment**: Supports HunyuanVideo, CogVideoX, Mochi backends
- **Links**: [GitHub](https://github.com/hao-ai-lab/FastVideo) | [Paper](https://arxiv.org/abs/2502.XXXXX)

```python
# FastVideo with HunyuanVideo backend
from fastvideo import FastVideoInference
model = FastVideoInference.from_pretrained("hunyuanvideo-turbo")
video = model.generate("Sunset over mountains", num_frames=120)
```

---

#### Pyramidal Flow
**Developer**: ByteDance | **Released**: Oct 2024  
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 15-22s for 5s clips @ 768x768, 24fps
- **Hardware**: A100 (80GB) or H100
- **Architecture**: Pyramidal flow matching (autoregressive-style)
- **Key Innovation**: Multi-resolution pyramid for efficient generation
- **Best For**: High-quality generation with acceptable latency
- **Deployment**: Diffusers integration, cloud-native
- **Links**: [GitHub](https://github.com/jy0205/Pyramid-Flow) | [Project](https://pyramid-flow.github.io/)

---

#### Stable Video Diffusion (SVD-XT Turbo)
**Developer**: Stability AI | **Released**: Nov 2023  
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 18-25s for 4s clips @ 576x1024, 25fps
- **Hardware**: A100 (40GB+), RTX 4090 (24GB with FP16)
- **Architecture**: Latent video diffusion, I2V-focused
- **Key Innovation**: Frame-by-frame consistency via temporal layers
- **Best For**: Animating static images, product demos
- **Deployment**: Replicate API, ComfyUI, Diffusers
- **Links**: [HuggingFace](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) | [Announcement](https://stability.ai/news/stable-video-diffusion-open-ai-video-model)

---

#### DynamiCrafter (Turbo)
**Developer**: CUHK | **Released**: Oct 2023  
**Speed**: ⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 20-30s for 2s clips @ 512x512, 16fps
- **Hardware**: RTX 4090 (24GB) minimum
- **Architecture**: Image animation via video diffusion priors
- **Key Innovation**: Open-domain image animation (no domain restriction)
- **Best For**: Animating photographs, concept art
- **Deployment**: Gradio demo, standalone script
- **Links**: [GitHub](https://github.com/Doubiiu/DynamiCrafter) | [Demo](https://huggingface.co/spaces/Doubiiu/DynamiCrafter)

---

#### Live2Diff
**Developer**: MMLAB (CUHK) | **Released**: Jul 2024  
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐

- **Latency**: **Streaming-capable** (frame-by-frame generation)
- **Hardware**: RTX 4090 (24GB) for 512x512 streaming
- **Architecture**: Uni-directional temporal attention for online generation
- **Key Innovation**: No future-frame dependency, true streaming
- **Best For**: Live video translation, real-time style transfer
- **Deployment**: Custom CUDA kernels, requires optimization
- **Links**: [GitHub](https://github.com/open-mmlab/Live2Diff) | [Paper](https://arxiv.org/abs/2407.08701)

---

#### CausVid
**Developer**: MIT/Adobe (Tianwei Yin, Xun Huang) | **Released**: CVPR 2025  
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: **1.3 seconds initial**, then streaming frame-by-frame
- **Hardware**: H100 (80GB) recommended
- **Architecture**: Autoregressive diffusion with causal forcing (distilled from bidirectional models)
- **Key Innovation**: Few-step autoregressive generation, dramatically reduces computational overhead
- **Best For**: Real-time video generation, interactive applications
- **Deployment**: CVPR 2025 paper, code TBD
- **Links**: [GitHub](https://github.com/tianweiy/CausVid) | [Paper](https://arxiv.org/abs/2412.07772) | [Project](https://causvid.github.io/)

---

#### Causal Forcing
**Developer**: Tsinghua ML Group | **Released**: Feb 2026  
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: **Real-time interactive** (10-15 FPS on H100)
- **Hardware**: H100 (80GB) for real-time, A100 for near-real-time
- **Architecture**: Autoregressive diffusion distillation
- **Key Innovation**: Frame-by-frame generation with user interaction support
- **Best For**: Interactive video games, live editing
- **Deployment**: Research code available, production integration TBD
- **Links**: [GitHub](https://github.com/thu-ml/Causal-Forcing) | [Project](https://thu-ml.github.io/CausalForcing.github.io/)

---

### 🗣️ Real-Time Avatar & Talking Head
*Specialized models for face animation, lip-sync, and full-body avatars.*

#### EchoTorrent
**Developer**: Research Community | **Released**: Feb 2026
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: True Real-Time (streaming)
- **Hardware**: Unknown
- **Architecture**: Streaming multi-modal video generation
- **Key Innovation**: Sustains high-quality video generation natively in a streaming format for less buffering and faster TTFF.
- **Best For**: Talking avatars, real-time conversational agents.
- **Links**: [Paper](https://arxiv.org/abs/2602.XXXXX) (Link TBD)

---


#### FasterLivePortrait (KlingAI / KwaiVGI)
**Developer**: KwaiVGI & Community | **Released**: Late 2024 / 2025
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: True Real-Time (TensorRT optimized)
- **Hardware**: Mid-to-high end consumer GPUs
- **Architecture**: Latent keypoint-based animation
- **Key Innovation**: Extreme inference optimization via TensorRT for an already efficient base model.
- **Best For**: Real-time video agents, virtual streamers.
- **Deployment**: ComfyUI (`ComfyUI-AdvancedLivePortrait`), C++ implementations, Gradio.
- **Links**: [GitHub (Original)](https://github.com/KwaiVGI/LivePortrait) | [TensorRT Implementation](https://github.com/StartHua/FasterLivePortrait)

---

#### Ditto
**Developer**: Ant Group | **Released**: Late 2024 / 2025
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: True Real-Time (sub-second streaming)
- **Hardware**: High-end consumer GPUs (RTX 4090/3090)
- **Architecture**: Streaming-native pipeline
- **Key Innovation**: Dedicated online pipeline (`stream_pipeline_online.py`) with TensorRT integration.
- **Best For**: Low-latency conversational agents.
- **Deployment**: TensorRT and streaming configurations available on GitHub.
- **Links**: [GitHub](https://github.com/antgroup/ditto)

---

#### MuseTalk
**Developer**: Tencent | **Released**: 2024
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 30fps+ (Real-Time)
- **Hardware**: RTX 3090+
- **Architecture**: Audio-driven lip-syncing
- **Key Innovation**: Applies highly accurate lip-sync to pre-existing base face videos in real-time.
- **Best For**: Adding speech to static/looping avatars dynamically.
- **Deployment**: Highly modular, often integrated into larger agent pipelines.
- **Links**: [GitHub](https://github.com/Tencent/MuseTalk)

---

#### LiveTalk
**Developer**: Tencent AI | **Released**: Dec 2025
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: **Real-time multimodal** (<100ms)
- **Hardware**: RTX 4090
- **Architecture**: Audio-driven latent diffusion
- **Key Innovation**: "On-policy distillation + identity sinks."
- **Best For**: Virtual assistants, live streaming avatars
- **Links**: [Code coming soon] | [Paper coming soon]

---

#### VACE (Real-Time Adaptation)
**Developer**: DayDream Live | **Released**: Feb 2026
**Type**: Avatar Control Framework
**Base Model**: VACE (Pretrained)

- **Nature**: **Technique applied to VACE**, not a standalone model
- **Mechanism**: Moves reference frames to parallel conditioning pathway
- **Latency**: Adds 20-30% overhead for streaming control
- **Key Innovation**: Enables real-time autoregressive control on existing weights
- **Best For**: Adapting VACE for streaming/interactive applications
- **Links**: [GitHub](https://github.com/daydreamlive/scope) | [Paper](https://arxiv.org/abs/2602.14381)

---

## 🔧 Optimization Techniques

### Inference Acceleration Methods

| Technique | Speedup | Quality Impact | Complexity |
|-----------|---------|----------------|------------|
| **FP16/BF16 Precision** | 1.5-2x | Minimal | Low |
| **Flash Attention** | 1.3-1.8x | None | Medium |
| **Dynamic Batching** | 1.2-1.5x | None | Low |
| **Distillation (4-step)** | 3-5x | Moderate | High |
| **Token Merging** | 1.5-2.5x | Slight | Medium |
| **TeaCache** | 1.3x | Minimal | Low |
| **xDiT (Multi-GPU)** | Linear | None | High |
| **MemFlow** | 7.9% overhead | **Significantly Improved** (Long Context) | Medium |

### Recommended Stack

**For Production:**
```
Model: LTXVideo / FastVideo
Serving: Modal / RunPod Serverless
GPU: H100 PCIe (burst) + A6000 (steady-state)
Framework: Diffusers + Flash Attention 2
Optimization: FP16 + xFormers + MemFlow (for long videos)
```

**For Research:**
```
Model: Pyramidal Flow / Causal Forcing
Serving: Local A100/H100
Framework: Native PyTorch
Optimization: Custom CUDA kernels
```

---

## 📈 Benchmarks & Comparisons

### Speed vs Quality Trade-off

```
Quality ▲
    5 |                    Sora/Veo (Commercial, slow)
    4 |        HunyuanVideo●          
    3 |   FastVideo●   Pyramidal Flow●
    2 | LTXVideo●              SVD-XT●
    1 |        AnimateLCM●
      └─────────────────────────────────► Speed
        5s   10s  15s  20s  25s  30s+ (latency)
```

### Hardware ROI Analysis

| GPU | $/hr (Cloud) | Throughput (clips/hr) | Cost per clip | Best Use Case |
|-----|--------------|----------------------|---------------|---------------|
| RTX 5090 | ~$0.50 (est) | ~960 (MonarchRT) | $0.0005 | Ultimate Local RT |
| RTX A6000 | $1.50 | 300 (LTXVideo) | $0.005 | High-volume, fast |
| A100 (40GB) | $2.50 | 180 (Pyramidal) | $0.014 | Balanced |
| H100 (80GB) | $4.50 | 250 (FastVideo) | $0.018 | Quality priority |
| RTX 4090 (Local) | $0 (capex) | 200 (SVD-XT) | $0 | Cost-sensitive |

---

## 🏗️ Deployment Guides

### Quick Deploy: LTXVideo on Modal

```python
# modal_ltxvideo.py
import modal
app = modal.App("ltxvideo-realtime")
image = modal.Image.debian_slim().pip_install("diffusers", "torch", "xformers")

@app.function(gpu="A10G", timeout=300)
def generate_video(prompt: str):
    from diffusers import LTXVideoPipeline
    pipe = LTXVideoPipeline.from_pretrained("Lightricks/LTX-Video").to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    video = pipe(prompt, num_inference_steps=8).frames
    return video

@app.local_entrypoint()
def main(prompt: str = "A dog running in a park"):
    result = generate_video.remote(prompt)
    print(f"Generated {len(result)} frames")
```

```bash
modal run modal_ltxvideo.py --prompt "Cyberpunk city at night"
```

---

### Docker: FastVideo Production Setup

```dockerfile
# Dockerfile.fastvideo
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip git
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN git clone https://github.com/hao-ai-lab/FastVideo /app/fastvideo
WORKDIR /app/fastvideo
RUN pip install -e .
EXPOSE 8000
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t fastvideo:latest -f Dockerfile.fastvideo .
docker run --gpus all -p 8000:8000 fastvideo:latest
```

---

## 🎓 Research Papers

### 2026
- **MonarchRT** (Feb 2026): 16 FPS on RTX 5090 [[arXiv]](https://arxiv.org/abs/2602.XXXXX)
- **S2DiT** (Jan 2026): Mobile streaming (10 FPS on iPhone) [[arXiv]](https://arxiv.org/abs/2601.XXXXX)
- **Adapting VACE for Real-Time Autoregressive Video Diffusion** (Feb 2026): Streaming control with 20-30% latency overhead [[arXiv]](https://arxiv.org/abs/2602.14381)
- **Causal Forcing** (Feb 2026): Autoregressive diffusion for real-time interactive video [[arXiv]](https://arxiv.org/abs/2602.02214)
- **Context Forcing** (Feb 2026): Long-context autoregressive with slow-fast memory [[arXiv]](https://arxiv.org/abs/2602.06028)

### 2025
- **MemFlow** (Dec 2025): Flowing Adaptive Memory for Consistent and Efficient Long Video Narratives [[arXiv]](https://arxiv.org/abs/2512.14699)
- **LiveTalk** (Dec 2025): Real-time multimodal avatar [[arXiv]](https://arxiv.org/abs/2512.XXXXX)
- **MotionStream** (Nov 2025): 29 FPS streaming [[arXiv]](https://arxiv.org/abs/2511.XXXXX)
- **StreamDiT** (July 2025): 16 FPS streaming [[arXiv]](https://arxiv.org/abs/2507.XXXXX)
- **CausVid** (CVPR 2025): Fast autoregressive video via diffusion forcing (1.3s latency) [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf)
- **URSA** (ICLR 2026, Oct 2025): Uniform discrete diffusion for unified video generation [[arXiv]](https://arxiv.org/abs/2510.24717)
- **FastVideo** (Feb 2025): Unified acceleration framework [[GitHub]](https://github.com/hao-ai-lab/FastVideo)
- **UltraViCo** (Oct 2025): Breaking extrapolation limits in DiTs [[arXiv]](https://arxiv.org/abs/2511.20123)
- **Pyramidal Flow** (Oct 2024): Flow matching for efficient generation [[arXiv]](https://arxiv.org/abs/2410.XXXXX)

### 2024
- **NOVA** (ICLR 2025, Dec 2024): Autoregressive video without vector quantization [[arXiv]](https://arxiv.org/abs/2412.14169)
- **LTXVideo** (Oct 2024): Real-time latent diffusion [[GitHub]](https://github.com/Lightricks/LTX-Video)
- **Live2Diff** (Jul 2024): Streaming video translation [[arXiv]](https://arxiv.org/abs/2407.08701)
- **AnimateDiff-Lightning** (Apr 2024): Few-step animation [[arXiv]](https://arxiv.org/abs/2404.XXXXX)

### 2023
- **Stable Video Diffusion** (Nov 2023): Latent video models at scale [[arXiv]](https://arxiv.org/abs/2311.15127)
- **AnimateDiff** (Jul 2023): Motion modules for T2I [[arXiv]](https://arxiv.org/abs/2307.04725)

---

## 🛠️ Tools & Frameworks

### Inference Frameworks
- **[Diffusers](https://github.com/huggingface/diffusers)**: HuggingFace standard, broad model support
- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**: Node-based workflow, excellent for prototyping
- **[FastVideo](https://github.com/hao-ai-lab/FastVideo)**: Specialized acceleration, production-grade
- **[VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys)**: PAB (Pyramid Attention Broadcast) optimization

### Serving Platforms
- **[Modal](https://modal.com)**: Serverless GPU, pay-per-second
- **[RunPod](https://runpod.io)**: Dedicated/serverless mix
- **[Replicate](https://replicate.com)**: Auto-scaling containers
- **[BentoML](https://bentoml.com)**: Self-hosted model serving

### Optimization Libraries
- **[xFormers](https://github.com/facebookresearch/xformers)**: Memory-efficient attention
- **[Flash Attention 2](https://github.com/Dao-AILab/flash-attention)**: Fastest attention kernels
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**: Distributed inference
- **[TensorRT](https://developer.nvidia.com/tensorrt)**: NVIDIA-optimized inference (requires export)

---

## 📦 Datasets for Fine-Tuning

### High-FPS / Real-Time Focused
- **[Panda-70M](https://github.com/snap-research/Panda-70M)**: 70M video-caption pairs (Snap)
- **[OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M)**: 1M high-quality clips (NJU)
- **[InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)**: 7M clips, multilingual
- **[WebVid-10M](https://github.com/m-bain/webvid)**: 10M stock footage clips

---

## 🔬 Experimental / Bleeding Edge

### Autoregressive Models (Video-Only)

#### NOVA
**Developer**: BAAIV | **Released**: Dec 2024 (ICLR 2025)  
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐

- **Latency**: 20-30s estimated (non-quantized autoregressive)
- **Hardware**: 0.6B params, runs on high-end consumer GPUs
- **Architecture**: Autoregressive without vector quantization
- **Key Innovation**: Frame-by-frame + set-by-set prediction, no VQ bottleneck
- **Best For**: Research baseline, predecessor to URSA
- **Status**: Code released, ICLR 2025 accepted
- **Links**: [GitHub](https://github.com/baaivision/NOVA) | [Paper](https://arxiv.org/abs/2412.14169) | [Project](https://bitterdhg.github.io/NOVA_page/)

---

### Multi-Modal Models (Video + Image Generation)

> **Note**: These models handle both video and image generation in a unified architecture. Included here because they support video generation at real-time speeds, but they're not video-only.

#### URSA (Uniform Discrete Diffusion)
**Developer**: BAAIV (Beijing Academy of AI) | **Released**: Oct 2025  
**Speed**: ⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐⭐  
**Modalities**: T2V, I2V, T2I, V2V

- **Latency**: Estimated 15-25s for short video clips (based on architecture)
- **Hardware**: 0.6B model on RTX 4090, 1.7B on H100
- **Architecture**: Discrete diffusion with metric path (successor to NOVA)
- **Key Innovation**: Single unified model replaces specialized T2V/T2I/I2V models, no vector quantization
- **Best For**: Production pipelines needing both image + video generation from one endpoint
- **Status**: ICLR 2026 accepted, models + code released
- **Links**: [GitHub](https://github.com/baaivision/URSA) | [Paper](https://arxiv.org/abs/2510.24717) | [HuggingFace](https://huggingface.co/collections/BAAI/ursa) | [Demo](https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480)

```python
# URSA models available:
# Video generation:
# - URSA-0.6B-FSQ320: 49 frames @ 512x320 (T2V, I2V, V2V)
# - URSA-1.7B-FSQ320: 49 frames @ 512x320 (higher quality)
# Image generation:
# - URSA-1.7B-IBQ1024: 1024x1024 images (T2I)
# Single pipeline handles all modalities
```

---

### Not Production-Ready (Yet)
- **MAGI-1** (Apr 2025): Autoregressive MoE, chunk-based [[GitHub]](https://github.com/SandAI-org/Magi-1)
- **Video-T1** (Mar 2025): Test-time scaling for quality boost [[arXiv]](https://arxiv.org/abs/2503.18942)
- **Context Forcing** (Feb 2026): Long-context autoregressive with slow-fast memory [[arXiv]](https://arxiv.org/abs/2602.06028)

---

## 💡 Contributing

We welcome contributions! Please:
1. Focus on models with **demonstrated real-time performance**
2. Include benchmarks (latency, GPU, quality metrics)
3. Provide deployment instructions where possible
4. Link to official repos/papers/demos

**PR Template:**
```markdown
## [Model Name]
- **Speed**: ⚡⚡⚡⚡ (1-5 lightning bolts)
- **Quality**: ⭐⭐⭐⭐ (1-5 stars)
- **Latency**: X seconds for Y frames @ Z resolution
- **Hardware**: GPU model + VRAM
- **Links**: [GitHub] [Paper] [Demo]
- **Deployment**: Quick start commands
```

---

## 📚 Additional Resources

### Blogs & Tutorials
- [Modal: Text-to-Video Models Comparison](https://modal.com/blog/text-to-video-ai-article)
- [HuggingFace: Video Diffusion State](https://huggingface.co/blog/video_gen)
- [Replicate: Video Generation Guide](https://replicate.com/blog)

### Community
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion) — Video generation threads
- [ComfyUI Discord](https://discord.gg/comfyui) — Video workflow sharing
- [Eleuther AI Discord](https://discord.gg/eleutherai) — Research discussions

---

## 📧 Maintainer

**Aaron Jones** ([@yepicaiaaron](https://github.com/yepicaiaaron))  
Expert in video generation infrastructure (Yepic AI, VideoStack)

Questions? Open an issue or PR!

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yepicaiaaron/awesome-realtime-video-generation&type=Date)](https://star-history.com/#yepicaiaaron/awesome-realtime-video-generation&Date)

---

## 📜 License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, Aaron Jones has waived all copyright and related rights to this work.
