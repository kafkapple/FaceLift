# FaceLift Mouse Project

GS-LRM 기반 Multi-view Mouse 3D Reconstruction

---

## Research Goal

**목표**: Template-free, monocular input → 3D reconstruction 모델을 **small non-rigid moving object**로 확장

- **첫 대상**: 6개 카메라 뷰 생쥐 데이터
- **후속 계획**: Pose-Splatter와 비교, behavior analysis downstream task용 feature로 활용

### Success Criteria

| 유형 | 지표 |
|------|------|
| **정량** | PSNR, loss, IoU |
| **정성** | 3D Gaussian 재구성 품질, Turntable MP4, mask GT/pred 시각화, 전처리 검증 보고서 |

### Related Papers

- [FaceLift](https://arxiv.org/abs/2412.17812) - 원본 논문
- [GS-LRM](https://arxiv.org/abs/2404.19702) - 기반 아키텍처
- [3D Gaussian Splatting](https://arxiv.org/abs/2307.01097) - 렌더링 기법

### Original Repository

- **Upstream**: https://github.com/weijielyu/FaceLift
- 원본과 차이 비교 시 참조

---

## Environment

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift
```

### GPU Usage

| 규칙 | 상세 |
|------|------|
| **허용 GPU** | `CUDA_VISIBLE_DEVICES=4,5,6,7` (4~7번만 사용) |
| **동시 실행** | VRAM 허용 시 같은 GPU에서 2-3개 작업 가능 |

---

## Dataset Version Schema

### 핵심 개념

| 접미사 | 의미 | 설명 |
|--------|------|------|
| **_t** | **Temporal Split** | 시간순 3분할 (train→val→test). Data leak 방지 |
| 없음 | Random Split | 전 구간에서 무작위 샘플링. 오버핏 위험 있으나 학습 촉진 |

### D7 계열 관계도

```
D7 (기본: random split, fx_only scale)
├── D7_1: individual scale (별도 scale_x, scale_y)
│   └── D7_1_t: D7_1 + temporal split (★ 권장)
├── D7_2: average scale (동일 scale_x = scale_y)
├── D7_5: optimal scale
│   └── D7_5b: object-aware optimal
└── D7_t: D7 + temporal split
```

### Split 방식 비교

| 방식 | Train | Val | Test | 특징 |
|------|-------|-----|------|------|
| **Random** | 전구간 무작위 | 전구간 무작위 | - | Data leak 위험, 학습 촉진 |
| **Temporal (_t)** | 0-60% 시간 | 60-80% 시간 | 80-100% 시간 | Pose-splatter 호환, 공정 평가 |

### 버전별 상세

| 버전 | Scale | Split | PP | 상태 |
|------|-------|-------|-----|------|
| **D7_1_t** | individual | temporal | shift_to_256 | ★ 권장 |
| **D7_1** | individual | random | shift_to_256 | 활성 |
| **D7_t** | fx_only | temporal | shift_to_256 | 활성 |
| **D7** | fx_only | random | shift_to_256 | 기본 |

---

## Project Structure

```
FaceLift/
├── mouse_extensions/       # ★ 모든 확장 구현은 여기에
│   ├── preprocessing/      # 전처리 (preset 기반 통합)
│   ├── model/             # loss, visualization
│   └── scripts/           # 유틸리티, 진단
├── configs/mouse/          # 실험 설정 (base + dataset + experiment)
├── gslrm/                  # GS-LRM 코어 (최소 수정)
└── docs/                   # 기술 문서
```

### Extension Guidelines

> **원칙**: 모든 추가 구현은 `mouse_extensions/` 하위에 배치
> - 단일 기능 원칙, 간결한 모듈
> - 원본 코드(`gslrm/`) 수정 최소화

---

## Quick Commands

### Training
```bash
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_t_E1_1_paper_random.yaml
```

### Preprocessing
```bash
# Preset 기반 통합 전처리
python -m mouse_extensions.preprocessing.unified_preprocessor \
    --preset D7.1 \
    --input_dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

---

## Known Issues

1. **D4 PP Bug**: cx=cy=256 강제 → ghosting → D7에서 해결
2. **bf16 NaN**: opacity_reg + bf16 → `.float().clamp(1e-4)` 적용됨
3. **카메라 정규화**: fx→549, trans→2.7 필수

## Anti-Patterns (절대 금지)

- ⛔ `killall python` 사용 금지
- ⛔ 실험 자동 실행 금지 (명령어만 제공)
- ⛔ GPU 0~3번 사용 금지

---

## Documentation

### 문서 위치

| 유형 | 위치 |
|------|------|
| **연구 노트 (로컬)** | `/Users/joon/Documents/Obsidian/30_Projects/_CODES/code_Face_Lift/docs` |
| **기술 문서 (서버)** | `/home/joon/dev/FaceLift/docs/` |

### 작업 완료 시 필수

1. 연구 노트 작성 (로컬 Obsidian)
2. Git 커밋 (Small commits)
3. 교육용 설명 문서 (초보 연구자 대상)

---

## Conventions

| 항목 | 규칙 |
|------|------|
| **명명규칙** | `D{버전}_{split}_E{N}` (예: D7_1_t_E1) |
| **Config** | base + dataset + experiment 3단계 merge |
| **문서** | Markdown + LaTeX 수식 + 표 |
