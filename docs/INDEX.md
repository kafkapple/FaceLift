# FaceLift Mouse Documentation

> **Map of Content (MoC)** - 모든 문서의 중앙 허브
> **Updated**: 260205

---

## Quick Links

| 목적 | 문서 |
|------|------|
| ⚡ 명령어 참조 | [[MOUSE_QUICK_REFERENCE]] |
| 🔬 실험 설정 | [[experiments/EXPERIMENT_REGISTRY]] |
| 📊 데이터셋 | [[datasets/PREPROCESSING_REGISTRY]] |

---

## 1. 운영 가이드 (guides/)

| 문서 | 내용 |
|------|------|
| [[guides/POSE_SPLATTER_GUIDE]] | Pose Splatter 비교 실험 설정 |
| [[guides/DEFORMATION_INTEGRATION_GUIDE]] | Temporal consistency 구현 |

---

## 2. 실험 문서 (experiments/)

| 문서 | 내용 |
|------|------|
| [[experiments/EXPERIMENT_REGISTRY]] | ⭐ 실험 레지스트리 (SSOT) |
| [[experiments/EXPERIMENT_QUICKSTART]] | 빠른 시작 가이드 |
| [[experiments/EXPERIMENT_CONFIG_GUIDE]] | Config 시스템 상세 |
| [[experiments/INFERENCE_E2E_GUIDE]] | E2E 추론 파이프라인 |
| [[experiments/TRAINING_LOGGING_GUIDE]] | WandB 로깅 가이드 |
| [[experiments/VISUALIZATION_SETTINGS]] | Turntable/시각화 설정 |

---

## 3. 데이터셋 문서 (datasets/)

| 문서 | 내용 |
|------|------|
| [[datasets/PREPROCESSING_REGISTRY]] | ⭐ 전처리 레지스트리 (SSOT) |
| [[datasets/M5_SERIES_SPEC]] | M5 시리즈 상세 스펙 |
| [[datasets/RAW_DATA]] | 원본 데이터 정보 |

---

## 4. 연구 노트 (research/)

| 날짜 | 주제 |
|------|------|
| [[research/260205_Research_Notes]] | H1 진단, 핵심 가설, Master Plan |
| [[research/260204_Research_Notes]] | 4D Gaussian, Temporal Methods |
| [[research/260203_Research_Notes]] | Deformation Network, CFG Ablation |
| [[research/260130_Debug_Turntable_Visualization]] | Turntable 버그 수정 |
| [[research/260129_Debug_NFS_Stale_Handle]] | NFS 에러 해결 |

---

## 5. Config 구조

```
configs/
├── base/
│   └── default.yaml              # 기본 설정
├── datasets/
│   ├── M5.yaml, M5t.yaml, M5t2.yaml
│   └── M0.yaml, M0_n.yaml        # Ablation용
├── experiments/
│   ├── E0_1_facelift.yaml        # Baseline
│   ├── E1_2_alpha.yaml           # GT Mask
│   └── view_ablation/            # View Ablation
│       ├── E0_1_1view.yaml
│       ├── E0_1_2view.yaml
│       ├── E0_1_3view.yaml
│       ├── E0_1_5view.yaml
│       └── E0_1_6view.yaml
└── mvdiffusion/
    ├── mouse_mvdiffusion_M5t2.yaml
    └── mouse_mvdiffusion_M5t2_cyclic.yaml
```

---

## 6. 핵심 인사이트 요약

### View Ablation (260205)
- **3-view > 4-view** (Inference-time PSNR 21.12 vs 19.58)
- 적은 뷰가 오히려 좋을 수 있음

### MVDiffusion (260205)
- **H1 진단**: M5t에서 GS-LRM only >> E2E (+1.41 PSNR gap)
- **결론**: MVDiffusion이 병목 (undertrained 시)

### 데이터 (260205)
- **다양성 > Epoch**: M5t2 (2880×6ep) > M5t (1198×20ep)

---

## 7. 자주 사용하는 명령어

### GS-LRM 학습
```bash
export CUDA_VISIBLE_DEVICES=X && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    > logs/gslrm_M5t2.log 2>&1 &
```

### MVDiffusion 학습
```bash
export CUDA_VISIBLE_DEVICES=X && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2.yaml \
    > logs/mvdiff_M5t2.log 2>&1 &
```

### E2E 추론
```bash
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --model M5t2 --num_frames 50 --prefer_ema
```

→ 상세: [[MOUSE_QUICK_REFERENCE]]

---

*FaceLift Mouse Documentation | MoC v2.0 | 260205*
