---
date: 2024-12-21
context_name: "2_Research"
tags: [ai-assisted, bug-fix, mvdiffusion, gslrm, mouse-facelift]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# MVDiffusion 체크포인트 경로 버그 분석

## 요약

합성 데이터 생성 시 **잘못된 MVDiffusion 체크포인트 경로**를 사용하여, 생쥐 대신 **사람 얼굴**이 생성되는 치명적 버그 발견.

## 버그 상세

### 증상
- `data_mouse_synthetic_v2/`의 6-view 합성 이미지 중 일부가 **사람 얼굴**로 생성됨
- GS-LRM validation 시 GT 이미지에서 생쥐/사람 혼합 관찰
- PSNR 등 메트릭이 의미없는 값으로 계산됨

### 원인
`generate_synthetic_data.py` 실행 시 **존재하지 않는 체크포인트 경로** 지정:

```bash
# 잘못된 경로 (존재하지 않음)
--mvdiff_checkpoint checkpoints/mvdiffusion/mouse_centered_real/checkpoint-2000

# 올바른 경로 (실제 존재)
--mvdiff_checkpoint checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000
```

### 스크립트 동작
`scripts/generate_synthetic_data.py` 라인 162-202:
```python
unet_path = os.path.join(checkpoint_path, "unet")
if os.path.exists(unet_path):
    # Finetuned UNet 로드
    ...
else:
    # EMA 시도 후 없으면:
    print("Warning: No finetuned UNet found, using base model")  # 라인 202
```

체크포인트가 존재하지 않으면 **warning만 출력하고 base model (human pretrained)을 그대로 사용**.
이로 인해 생쥐 이미지를 입력해도 **사람 얼굴**이 생성됨.

### 실제 체크포인트 구조
```
checkpoints/mvdiffusion/
├── pipeckpts/              # Base model (human pretrained)
├── mouse_pixel_based/      # ✅ 실제 finetuned 체크포인트 위치
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   └── checkpoint-2000/    # ✅ 사용해야 할 체크포인트
│       └── unet/
│           ├── config.json
│           └── diffusion_pytorch_model.safetensors
├── mouse_uniform_v2/       # 다른 실험용
└── mouse_centered_real/    # ❌ 존재하지 않음!
```

## 해결 방법

### 1. 즉시 조치
```bash
# 잘못된 데이터/체크포인트 삭제
rm -rf data_mouse_synthetic_v2
rm -rf checkpoints/gslrm/mouse_synthetic_v2

# 올바른 체크포인트로 재생성
python scripts/generate_synthetic_data.py \
    --input_dir data_mouse_centered \
    --output_dir data_mouse_synthetic_v3 \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000 \
    --device cuda:1
```

### 2. 코드 개선 권장
```python
# generate_synthetic_data.py 수정 제안
def load_mvdiffusion_pipeline(...):
    unet_path = os.path.join(checkpoint_path, "unet")
    if not os.path.exists(unet_path):
        # Warning 대신 Error로 변경!
        raise FileNotFoundError(
            f"Finetuned UNet not found at {unet_path}. "
            f"Available checkpoints: {list_available_checkpoints()}"
        )
```

## 영향 범위

| 데이터/체크포인트 | 상태 | 조치 |
|-----------------|------|-----|
| `data_mouse_synthetic_v2/` | ❌ 오염 | 삭제 필요 |
| `checkpoints/gslrm/mouse_synthetic_v2/` | ❌ 무효 | 삭제 필요 |
| `logs/train_gslrm_synthetic_v2.log` | ⚠️ 참고용 | 보존 (분석용) |

## 교훈

1. **Fail-fast 원칙**: Warning 대신 Error로 조기 실패하여 잘못된 실험 방지
2. **경로 검증**: 체크포인트 경로는 실행 전 명시적 존재 확인 필수
3. **체크포인트 명명 규칙**: 실험별 체크포인트 위치를 명확히 문서화

## 관련 파일

- `scripts/generate_synthetic_data.py`: 합성 데이터 생성 스크립트
- `configs/mouse_gslrm_synthetic_v2.yaml`: GS-LRM 학습 설정
- 올바른 체크포인트: `checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000/`
