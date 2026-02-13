# MV-Adapter Technical Reference

> **목적**: MV-Adapter 아키텍처 분석 + FaceLift 통합 가능성 평가
> **출처**: [huanngzh/MV-Adapter](https://github.com/huanngzh/MV-Adapter) (ICCV 2025)
> **상태**: 📋 조사 완료 | **Created**: 2026-02-13
> **관련**: → [GENERALIZATION_ROADMAP](../hypotheses/GENERALIZATION_ROADMAP.md) | [MULTIVIEW_DIFFUSION_THEORY](./MULTIVIEW_DIFFUSION_THEORY.md)

---

## 1. 개요

MV-Adapter는 사전학습된 text-to-image diffusion 모델을 multi-view 생성기로 변환하는 **plug-and-play adapter**.

### 핵심 설계 원칙

1. **Parallel Decoupled Attention**: 기존 self-attention과 MV attention이 **병렬** 실행
2. **Zero-Init Strategy**: MV output projection을 0으로 초기화 → 학습 초기 기존 feature space 보존
3. **Weight Copying**: 기존 UNet Q/K/V → MV layers에 복사 → 사전학습 prior 활용
4. **T2IAdapter 재사용**: camera guider로 검증된 conditioning 메커니즘 활용

### FaceLift 호환성

| 항목 | MV-Adapter | FaceLift | 호환성 |
|------|------------|----------|:------:|
| Base model | SDXL / **SD2.1** | **SD2.1-UnCLIP** | ✅ SD2.1 config 존재 |
| Resolution | 768 (SDXL) / **512 (SD2.1)** | **512** | ✅ 동일 |
| Views | 6 | 6 | ✅ 동일 |
| Camera | Plücker rays (6ch) | M5 rig (w2c known) | ✅ 변환 가능 |

---

## 2. 아키텍처 상세

### 2.1 Attention Processor

**두 가지 프로세서**:
- `DecoupledMVRowSelfAttnProcessor2_0`: Row-wise attention (3D 물체용)
- `DecoupledMVRowColSelfAttnProcessor2_0`: Row + Column attention (6-view cubemap용)

**Row-wise attention 핵심**:

```python
# 높이(row) 기준으로 batch를 merge → 같은 행끼리만 attention
q_row = rearrange(q, "b v h hh ww c -> (b hh) h (v ww) c")
k_row = rearrange(k, "b v h hh ww c -> (b hh) h (v ww) c")
v_row = rearrange(v, "b v h hh ww c -> (b hh) h (v ww) c")

out_row = F.scaled_dot_product_attention(q_row, k_row, v_row)
```

- `(b hh)` = batch × height index → **같은 행(row)의 token끼리만 attend**
- `(v ww)` = view × width → 4개 뷰의 같은 행을 하나의 sequence로
- **FaceLift RMA와의 차이**: FaceLift는 dense (행 제한 없음), MV-Adapter는 **명시적 row mask**

**병렬 결합**:

```python
hidden_states = out_self + mv_scale * out_mv + ref_scale * out_ref + residual
```

### 2.2 Camera Guider (T2IAdapter)

**입력**: Plücker ray coordinates [B, 6, H, W]
- Channels 0-2: Ray direction (normalized)
- Channels 3-5: Ray moment (m = origin × direction)

**구조**: `FullAdapterXL`
```
Input: [B, 6, 512, 512]
  → PixelUnshuffle (16x): [B, 6×256, 32, 32]
  → Conv_in: [B, 320, 32, 32]
  → AdapterBlock 1: [B, 320, 32, 32]  → feature_1
  → AdapterBlock 2: [B, 640, 32, 32]  → feature_2
  → AdapterBlock 3 + Downsample: [B, 1280, 16, 16] → feature_3
  → AdapterBlock 4: [B, 1280, 16, 16] → feature_4
Output: 4 multi-scale features → UNet down_intrablock_additional_residuals
```

### 2.3 Reference Feature Extraction (I2MV)

Image-to-multi-view 변형에서 사용:

1. Reference image를 VAE encode (no noise)
2. Frozen UNet에 t=0으로 forward → **모든 self-attention layer에서 feature 캐시**
3. Denoising 시 캐시된 feature를 `ref_hidden_states`로 주입
4. MV attention processor가 `to_q_ref/to_k_ref/to_v_ref`로 reference cross-attention 수행

---

## 3. 학습 설정

### 3.1 Trainable Parameters

| Component | Params | Trainable | 비고 |
|-----------|:------:|:---------:|------|
| UNet (original) | ~865M (SD2.1) | ❌ Frozen | |
| UNet (_mv layers) | ~127M | ✅ | Duplicated Q/K/V + output proj |
| T2IAdapter | ~80M | ✅ | Camera guider CNN |
| VAE | ~83M | ❌ Frozen | |
| CLIP | ~340M | ❌ Frozen | |
| **Total trainable** | **~207M** | | ~15% of SD2.1 |

### 3.2 Config (SD2.1)

```yaml
# mvadapter_t2mv_sd21.yaml (FaceLift 관련)
system:
  pretrained_model_name_or_path: stabilityai/stable-diffusion-2-1
  init_adapter_kwargs:
    num_views: 6
    self_attn_processor: DecoupledMVRowSelfAttnProcessor2_0
    cond_in_channels: 6  # Plücker coordinates
    copy_attn_weights: true
    zero_init_module_keys: [to_out_mv]
  trainable_modules: [_mv]
  train_cond_encoder: true

optimizer:
  type: AdamW
  lr: 5.0e-5
  weight_decay: 0.01

trainer:
  max_epochs: 10
  gradient_clip_val: 1.0
  precision: 16-mixed
  accumulate_grad_batches: 4
```

### 3.3 Pretrained Checkpoints

HuggingFace: `huanngzh/mv-adapter`

| Checkpoint | Base | Resolution | 용도 |
|------------|------|:----------:|------|
| `mvadapter_t2mv_sd21.safetensors` | **SD2.1** | **512** | Text→6view (**FaceLift 호환**) |
| `mvadapter_i2mv_sd21.safetensors` | **SD2.1** | **512** | Image→6view (**FaceLift 호환**) |
| `mvadapter_t2mv_sdxl.safetensors` | SDXL | 768 | Text→6view (고해상도) |
| `mvadapter_i2mv_sdxl.safetensors` | SDXL | 768 | Image→6view (고해상도) |

---

## 4. FaceLift 통합 분석

### 4.1 통합 방안

**Option A: Attention Processor 교체** (권장)
- FaceLift의 `XFormersMVAttnProcessor` → `DecoupledMVRowSelfAttnProcessor2_0`
- Camera guider (T2IAdapter) 추가
- 기존 UNet weights 유지, MV layers만 학습

**Option B: Full Pipeline 교체**
- MV-Adapter의 pipeline을 FaceLift에 맞게 수정
- SD2.1-UnCLIP 특수 기능 (image conditioning) 보존 필요

### 4.2 통합 시 고려사항

| 이슈 | 상세 | 해결 |
|------|------|------|
| **SD2.1 vs SD2.1-UnCLIP** | FaceLift는 UnCLIP (image conditioning) 사용 | MV-Adapter SD2.1 config 기반 + UnCLIP 추가 |
| **Row-wise vs Dense** | MV-Adapter는 명시적 row, FaceLift는 dense | M5 비균일 카메라에는 dense가 유리할 수 있음 |
| **Plücker encoding** | M5 카메라의 w2c → c2w → Plücker 변환 필요 | `m5_cameras.json`에서 변환 가능 |
| **GS-LRM 연동** | MV-Adapter 출력 → GS-LRM 입력 호환 | 6뷰 512×512 동일 → 호환 |
| **학습 데이터** | Objaverse (대규모) vs M5 mouse (3,600장) | Fine-tune 필요 |

### 4.3 통합 우선순위

```
Phase 1: Pretrained SD2.1 adapter 다운로드 + inference 테스트
         → Objaverse 학습 모델이 mouse에 얼마나 일반화되는지 확인

Phase 2: M5 데이터로 adapter fine-tune
         → camera guider에 M5 Plücker ray 입력
         → 기존 GS-LRM checkpoint와 호환 확인

Phase 3: I2MV (image-conditioned) 변형 통합
         → reference feature extraction 구현
         → FaceLift의 UnCLIP image conditioning과 연동
```

### 4.4 기존 scaffolding 파일 업데이트 필요

`mouse_extensions/model/mv_adapter.py` (현재 scaffolding):
- `DecoupledMVRowSelfAttnProcessor2_0` 실제 구현으로 교체
- `CameraGuider` → T2IAdapter 사용하도록 수정
- `attach_to_unet()` 구현 완료

---

## 5. MV-Adapter vs FaceLift 현재 구현 비교

| 측면 | FaceLift RMA | MV-Adapter |
|------|-------------|------------|
| **Attention type** | Dense (all-to-all) | Row-wise (explicit mask) |
| **Camera input** | 없음 (text prompt만) | Plücker rays (6ch per-pixel) |
| **Reference injection** | CLIP image embedding (UnCLIP) | Frozen UNet features at t=0 |
| **Trainable params** | 전체 UNet fine-tune | ~15% only (_mv + T2IAdapter) |
| **Multi-view inductive bias** | Sparse star topology | Row correspondence + camera guider |
| **비균일 카메라 대응** | 가능 (dense) 하지만 implicit | 가능 (camera guider) + explicit row |
| **View 수 변경** | Config만 변경 | 재학습 필요 (attention 구조 의존) |

---

## 6. Critical Assessment

### 강점
1. **효율적**: 전체 모델의 15%만 학습 → 작은 데이터셋에도 적합
2. **Plug-and-play**: LoRA, ControlNet 등과 병용 가능
3. **Camera-aware**: Plücker ray로 명시적 카메라 정보 인코딩
4. **SD2.1 지원**: FaceLift base model과 직접 호환

### 약점
1. **Row attention의 한계**: 6-view cubemap (elevation 0°, equispaced) 전용 설계
   - M5 비균일 카메라에서는 row correspondence가 부정확할 수 있음
   - 해결: dense attention으로 변경하거나 epipolar-aware attention mask 사용
2. **num_views 고정**: 학습 시 정한 view 수 변경 어려움
3. **Objaverse bias**: 학습 데이터 = synthetic objects → mouse domain에 재학습 필수

### FaceLift 특수 고려사항
- M5 카메라의 18.1° elevation 편차 → MV-Adapter의 strict row attention에 부정적
- **권장**: Row-only 대신 dense attention variant 사용하거나, camera guider가 보상하도록 학습
- UnCLIP image conditioning은 MV-Adapter의 reference feature extraction과 유사한 역할 → 양쪽 통합 가능

---

*MV-Adapter Technical Reference v1.0 | 2026-02-13*
