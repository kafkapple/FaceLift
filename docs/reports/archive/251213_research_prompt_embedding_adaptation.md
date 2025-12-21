# Prompt Embedding Adaptation 연구 보고서

**날짜**: 2025-12-13
**주제**: MVDiffusion 프롬프트 임베딩 분석 및 단계별 적응 전략
**상태**: In Progress

---

## 1. 현황 분석

### 1.1 Prompt Embedding 흐름

```
[Dataset Init]
    └─ prompt_embed_path에서 clr_embeds.pt 로드 [6, 77, 1024]
           ↓
[Training Loop]
    └─ batch['color_prompt_embeddings'] → UNet
           ↓
[UNet Forward]
    └─ encoder_hidden_states=prompt_embeddings (Cross-Attention)
           ↓
[Generation]
    └─ 프롬프트 의미에 맞는 이미지 생성
```

### 1.2 프롬프트 비교

| 버전 | 프롬프트 예시 | Cosine Sim |
|------|---------------|------------|
| **FaceLift (Original)** | `"a rendering image of 3D models, front view, color map."` | 1.00 (baseline) |
| **Mouse (Current)** | `"a rendering image of a 3D model, top-front view, from above at an angle, color map."` | **0.70** |

### 1.3 문제점 식별

| 문제 | 현재 상태 | 실제 데이터 |
|------|-----------|-------------|
| **도메인** | "rendering image of 3D model" | 실제 촬영 영상 (real video) |
| **시점** | "front, right, back..." | 경사 시점 (elevated, ~20°) |
| **대상** | 일반 "3D model" | 특정 대상 (mouse) |

**결과**: Semantic Gap으로 인해 수렴 속도 저하 (1000 steps → 2000+ steps 예상)

---

## 2. 6x Augmentation 영향 분석

### 2.1 메커니즘
```yaml
reference_view_idx: "random"  # 매 iteration마다 6개 중 1개 무작위 선택
```

- **이전**: 고정된 1개 view만 input
- **현재**: 6개 view가 번갈아 input (6배 다양성)

### 2.2 학습 영향

| 측면 | 영향 | 설명 |
|------|------|------|
| **수렴 속도** | ⬇️ 느림 | 더 많은 variation 학습 필요 |
| **일반화 성능** | ⬆️ 향상 | 다양한 view에서 robust |
| **과적합 위험** | ⬇️ 감소 | Augmentation은 regularization 효과 |
| **최종 품질** | ⬆️ 향상 (예상) | 모든 view에서 일관된 생성 |

### 2.3 과적합 우려 해소
- Data augmentation은 일반적으로 과적합을 **감소**시킴
- 6x augmentation = 6배 더 다양한 학습 신호
- 예상 수렴 시점: 기존의 2-3배 step 필요

---

## 3. 단계별 프롬프트 Fine-tuning (Curriculum Learning)

### 3.1 문헌 근거

#### [Curriculum DPO for Diffusion Models](https://arxiv.org/abs/2405.13637) (May 2024)
- Easy-to-hard 순서로 학습 pairs 제공
- Preference ranking 기반 난이도 분류
- **첫 번째 curriculum learning 적용 사례** (text-to-image diffusion)

#### [DomainStudio](https://openreview.net/forum?id=ancAesl2LU)
- Few-shot domain adaptation for diffusion models
- Overfitting 완화를 위한 단계적 접근

#### [DRaFT+](https://developer.nvidia.com/blog/enhance-text-to-image-fine-tuning-with-draft-now-part-of-nvidia-nemo/) (NVIDIA, 2024)
- KL divergence term으로 mode collapse 방지
- Pre-trained model과의 유사성 유지

#### [Denoising Task Difficulty Curriculum](https://arxiv.org/abs/2403.10348) (Kim et al., 2024)
- Diffusion training에 curriculum learning 적용
- Task 난이도 기반 학습 순서 조정

### 3.2 제안: 3단계 프롬프트 적응 전략

```
Stage 1: Domain Bridge (Steps 0-2000)
    └─ "a rendering image of 3D models, {view} view, color map."
    └─ FaceLift 원본 프롬프트로 빠른 초기 수렴

Stage 2: View Adaptation (Steps 2000-5000)
    └─ "a rendering image of 3D models, {view} view, from above at an angle, color map."
    └─ 경사 시점 정보 추가

Stage 3: Domain Shift (Steps 5000-20000)
    └─ "a photograph of a mouse, {view} view, from above at an angle."
    └─ 실제 데이터 도메인으로 전환
```

### 3.3 대안: 단일 단계 (현재)

현재 설정 유지 시:
- **장점**: 구현 단순, 최종 목표 프롬프트로 직접 학습
- **단점**: 초기 수렴 느림, 더 많은 step 필요
- **예상 수렴**: ~3000-5000 steps

---

## 4. 개선된 프롬프트 제안

### 4.1 Option A: 현실적 프롬프트 (권장)

```python
views = ["front", "front-right", "right", "back", "left", "front-left"]
description = "from above at an angle"

color_prompts = [
    f"a photograph of a mouse, {view} view, {description}."
    for view in views
]
```

**예시**:
- `"a photograph of a mouse, front view, from above at an angle."`
- `"a photograph of a mouse, back view, from above at an angle."`

### 4.2 Option B: Hybrid 프롬프트

```python
color_prompts = [
    f"a multi-view image of a mouse, {view} view, elevated camera angle."
    for view in views
]
```

### 4.3 Option C: 단순화 (FaceLift 호환)

```python
views = ["front", "front_right", "right", "back", "left", "front_left"]

color_prompts = [
    f"a rendering image of 3D models, {view} view, color map."
    for view in views
]
```

빠른 수렴을 위해 원본 FaceLift 프롬프트 유지

---

## 5. 권장 사항

### 5.1 단기 (현재 학습)

**현재 설정 유지 + 모니터링**
- 2000-3000 steps까지 수렴 관찰
- WandB에서 view별 품질 비교
- 수렴 안 되면 프롬프트 변경 고려

### 5.2 중기 (다음 실험)

**실험 A**: 원본 FaceLift 프롬프트로 학습
```yaml
prompt_embed_path: "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
```

**실험 B**: 현실적 프롬프트 생성 후 학습
```bash
python scripts/generate_mouse_prompt_embeds_realistic.py
```

### 5.3 장기 (최적화)

- 3단계 curriculum learning 구현
- A/B 테스트로 최적 프롬프트 조합 탐색
- View별 개별 프롬프트 최적화

---

## 6. 결론

1. **현재 수렴 지연 원인**: 프롬프트 차이 (Cosine Sim 0.70) + 6x augmentation
2. **과적합 우려 없음**: Augmentation은 regularization 효과
3. **문헌 지지**: Curriculum learning이 diffusion 학습에 효과적
4. **권장**: 현재 학습 2000 steps까지 관찰 후 결정

---

## References

- [Curriculum Direct Preference Optimization for Diffusion and Consistency Models](https://arxiv.org/abs/2405.13637) (May 2024)
- [DomainStudio: Fine-Tuning Diffusion Models for Domain-Driven Image Generation](https://openreview.net/forum?id=ancAesl2LU)
- [DRaFT+: Enhance Text-to-Image Fine-Tuning](https://developer.nvidia.com/blog/enhance-text-to-image-fine-tuning-with-draft-now-part-of-nvidia-nemo/)
- [Separate-and-Enhance: Compositional Finetuning for T2I Diffusion](https://dl.acm.org/doi/10.1145/3641519.3657527) (SIGGRAPH 2024)
- [Self-Play Fine-Tuning of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/860c1c657deafe09f64c013c2888bd7b-Paper-Conference.pdf) (NeurIPS 2024)
