# Contributing to Awesome Real-Time Video Generation

Thank you for considering contributing! This guide will help you add high-quality entries.

## Criteria for Inclusion

A model/tool/paper must meet **at least one** of these criteria:

### ✅ Must Have
1. **Demonstrated real-time or near real-time performance**
   - Text-to-video: <30s for 5s clips on consumer/datacenter GPUs
   - Image-to-video: <20s for 4s clips
   - Streaming: Frame-by-frame generation capability

2. **Reproducible results**
   - Open-source code OR publicly accessible API
   - Clear hardware requirements
   - Working demo or deployment instructions

3. **Production relevance**
   - Used in real products, OR
   - Cited by production systems, OR
   - Demonstrates novel optimization for inference speed

### ❌ Not Suitable
- Research-only models with no code/weights
- Models requiring >120s for standard clips
- Proprietary closed-source systems (unless exceptional)
- Quality-focused models without speed optimizations

---

## How to Add a Model

### 1. Choose the Right Section
- **Text-to-Video (Real-Time)**: T2V models with <30s latency
- **Image-to-Video (Real-Time)**: I2V models with <20s latency
- **Streaming / Interactive**: Frame-by-frame or online generation
- **Experimental / Bleeding Edge**: Promising but not production-ready

### 2. Use the Template

```markdown
#### [Model Name]
**Developer**: Organization/Lab | **Released**: Month Year  
**Speed**: ⚡⚡⚡⚡ (1-5 bolts) | **Quality**: ⭐⭐⭐⭐ (1-5 stars)

- **Latency**: X seconds for Y frames @ WxH resolution, Z fps
- **Hardware**: GPU model (VRAM) — minimum and recommended
- **Architecture**: Brief technical description (e.g., "Latent diffusion with...")
- **Key Innovation**: What makes it fast/unique (1 sentence)
- **Best For**: Primary use cases (3-5 words)
- **Deployment**: Available integrations (ComfyUI, Docker, etc.)
- **Links**: [GitHub](url) | [Paper](url) | [Demo](url)

\`\`\`bash
# Quick start example (if applicable)
git clone https://github.com/org/repo
python generate.py --prompt "..." --fast-mode
\`\`\`
```

### 3. Rating Guidelines

**Speed (⚡)**
- ⚡⚡⚡⚡⚡: <10s latency, streaming-capable
- ⚡⚡⚡⚡: 10-20s latency
- ⚡⚡⚡: 20-30s latency
- ⚡⚡: 30-60s latency (borderline)
- ⚡: >60s (not suitable for this list)

**Quality (⭐)**
- ⭐⭐⭐⭐⭐: Commercial-grade (Runway Gen-3 level)
- ⭐⭐⭐⭐: Production-ready (HunyuanVideo level)
- ⭐⭐⭐: Good for prototypes (LTXVideo level)
- ⭐⭐: Acceptable for demos
- ⭐: Proof-of-concept only

---

## How to Add Other Content

### Optimization Techniques
Add to the table in the **Optimization Techniques** section with:
- Technique name
- Speedup factor (empirical, cite source)
- Quality impact (None/Minimal/Slight/Moderate/Significant)
- Implementation complexity (Low/Medium/High)

### Tools & Frameworks
Add to appropriate subsection (**Inference Frameworks**, **Serving Platforms**, etc.) with:
- Name + link
- 1-sentence description
- When to use it

### Papers
Add to **Research Papers** under the appropriate year with:
- Title + month/year
- arXiv/conference link
- One-line summary

### Benchmarks
Update tables with **verifiable** numbers. Include:
- Hardware used
- Framework/optimizations applied
- Date of benchmark
- Source/citation

---

## Submission Process

### Via Pull Request
1. Fork the repository
2. Create a branch: `git checkout -b add-model-name`
3. Add your content following the template
4. Verify all links work
5. Commit: `git commit -m "Add [Model Name]"`
6. Push: `git push origin add-model-name`
7. Open a PR with a clear description

### Via Issue
If you don't want to create a PR:
1. Open an issue titled "Add [Model Name]"
2. Fill out the model template in the issue body
3. Include benchmarks and links
4. Maintainer will review and add

---

## Quality Standards

### Required for Acceptance
- [ ] All links are working
- [ ] Latency numbers are cited or tested
- [ ] Hardware requirements are specified
- [ ] Code/demo is accessible
- [ ] Writing is clear and concise

### Nice to Have
- [ ] Deployment instructions included
- [ ] Comparison to similar models
- [ ] Cost analysis ($/clip)
- [ ] Community adoption metrics

---

## Code of Conduct

- **Be respectful**: Constructive criticism only
- **No self-promotion spam**: Quality over quantity
- **Cite sources**: Don't claim benchmarks without evidence
- **Stay on topic**: This list is speed-focused, not general video gen

---

## Questions?

- Open an issue with the `question` label
- Tag `@ajonesdev` for maintainer attention
- Join discussions in existing PRs

---

**Thank you for contributing to the real-time video generation community!** 🚀
