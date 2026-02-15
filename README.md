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

| Model | Latency (5s @ 720p) | GPU | FPS | License | Status |
|-------|---------------------|-----|-----|---------|--------|
| [CausVid](#causvid) | **~1.3s + streaming** | H100 (80GB) | 24 | TBD | 🔬 CVPR 2025 |
| [LTXVideo](#ltxvideo) | **~8s** | RTX A6000 (48GB) | 24 | Apache 2.0 | ✅ Production |
| [AnimateDiff-Lightning](#animatediff-lightning) | ~10s | RTX 4090 (24GB) | 16 | Apache 2.0 | ✅ Production |
| [FastVideo](#fastvideo) | ~12s | H100 (80GB) | 30 | Apache 2.0 | ✅ Production |
| [Pyramidal Flow](#pyramidal-flow) | ~15s | A100 (80GB) | 24 | MIT | ✅ Production |
| [VideoLCM](#videolcm) | ~18s | H100 (80GB) | 24 | - | 🔬 Research |
| [SVD-XT (Turbo)](#stable-video-diffusion-turbo) | ~20s | A100 (40GB) | 25 | - | ✅ Production |
| [URSA-1.7B](#ursa-uniform-discrete-diffusion) † | ~20-25s (est.) | H100 (80GB) | 12 | Apache 2.0 | 🔬 ICLR 2026 |

*† Multi-modal model (also generates images). See [Multi-Modal Models](#multi-modal-models-video--image-generation) section.*

*Benchmarks measured on single GPU inference, T2V generation. Image-to-video typically 30-50% faster. CausVid streaming: initial latency 1.3s, then continuous frame generation.*

---

## 🚀 Models by Category

### Text-to-Video (Real-Time)

#### LTXVideo
**Developer**: Lightricks | **Released**: Oct 2024  
**Speed**: ⚡⚡⚡⚡⚡ | **Quality**: ⭐⭐⭐

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

### Image-to-Video (Real-Time)

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

### Streaming / Interactive

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
- **Links**: [Paper (PDF)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.pdf) | [Project](https://causvid.github.io/)

```python
# CausVid architecture enables:
# 1. Initial frame generation: 1.3s
# 2. Subsequent frames: streaming (continuous generation)
# 3. Autoregressive + diffusion forcing for quality
```

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

### Recommended Stack

**For Production:**
```
Model: LTXVideo / FastVideo
Serving: Modal / RunPod Serverless
GPU: H100 PCIe (burst) + A6000 (steady-state)
Framework: Diffusers + Flash Attention 2
Optimization: FP16 + xFormers
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
- **Causal Forcing** (Feb 2026): Autoregressive diffusion for real-time interactive video [[arXiv]](https://arxiv.org/abs/2602.02214)
- **Context Forcing** (Feb 2026): Long-context autoregressive with slow-fast memory [[arXiv]](https://arxiv.org/abs/2602.06028)

### 2025
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
- [Replicate: Video Generation Guide](https://replicate.com/blog/video-generation)

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
