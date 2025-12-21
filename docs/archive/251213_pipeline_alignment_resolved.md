# Mouse-FaceLift 파이프라인 정렬 문제 해결 보고서

**날짜**: 2024-12-13
**상태**: 문제 해결됨, 추가 최적화 진행 중

---

## 1. 문제 원인 확정

### 이전 상태 (문제)
```
MVDiffusion (FaceLift embeds) → 수평 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ❌ 카메라 불일치 → 이상한 3D 결과
```

### 현재 상태 (해결)
```
MVDiffusion (Mouse embeds) → 경사 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ✅ 카메라 일치 → 정상 3D 결과
```

---

## 2. 핵심 발견

| 구성요소 | 역할 | 발견 사항 |
|----------|------|-----------|
| prompt_embeds | MVDiffusion 출력 뷰 방향 결정 | **핵심 정렬 요소** |
| MVDiffusion UNet | 입력→멀티뷰 생성 | 이미 잘 학습됨 |
| GS-LRM | 멀티뷰→3D 복원 | 카메라 파라미터 민감 |

**결론**: prompt_embeds가 MVDiffusion 출력과 GS-LRM 입력 카메라를 연결하는 **핵심 bridge**

---

## 3. 현재 최적 설정

| 구성요소 | 경로 | 상태 |
|----------|------|------|
| MVDiffusion UNet | `mouse/original_facelift_embeds/checkpoint-6000` | ✅ 완료 |
| prompt_embeds | `mouse_prompt_embeds_6view/clr_embeds.pt` | ✅ 적용 |
| GS-LRM | `mouse_finetune/ckpt_0000000000020000.pt` | ⬆️ 개선 가능 |

---

## 4. GS-LRM 현재 학습 상태

### 체크포인트
- 최신: `ckpt_0000000000020000.pt` (20,000 steps)
- 시작점: FaceLift pretrained (`ckpt_0000000000021125.pt`)

### 학습 설정
```yaml
optimizer:
  lr: 2.0e-05
  weight_decay: 0.05

losses:
  l2_loss_weight: 1.0
  lpips_loss_weight: 0.5
  perceptual_loss_weight: 0.5
  ssim_loss_weight: 0.2

batch_size_per_gpu: 2
```

---

## 5. 다음 단계: GS-LRM 추가 학습

### 목표
- MVDiffusion + Mouse embeds 출력에 최적화된 GS-LRM

### 전략
1. 현재 체크포인트에서 추가 학습
2. Learning rate decay 적용
3. 더 많은 steps 학습

---

## 6. 관련 파일

- `mvdiffusion/data/mouse_prompt_embeds_6view/` - Mouse prompt embeddings
- `configs/mouse_config.yaml` - GS-LRM 학습 config
- `docs/guides/experiment_commands.md` - 실험 명령어 가이드
