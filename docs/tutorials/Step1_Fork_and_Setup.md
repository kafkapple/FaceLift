> **DEPRECATED (2026-01-29)**: This tutorial predates mouse_extensions refactoring. Does not reflect Modular config (-d -e), preset system, or mouse_extensions/ structure.

# Step 1: Fork & 환경 설정

> 원본 FaceLift 저장소를 fork 하고 개발 환경을 설정합니다.

## 1.1 Fork 및 Clone

### GitHub에서 Fork

1. [weijielyu/FaceLift](https://github.com/weijielyu/FaceLift) 방문
2. 우측 상단 "Fork" 버튼 클릭
3. 자신의 계정으로 fork 생성

### Clone

```bash
# Fork한 저장소 clone
git clone https://github.com/YOUR_USERNAME/FaceLift.git
cd FaceLift

# 원본 저장소를 upstream으로 추가
git remote add upstream https://github.com/weijielyu/FaceLift.git
```

---

## 1.2 환경 설정

### Conda 환경 생성

```bash
# 새 conda 환경 생성
conda create -n facelift python=3.10 -y
conda activate facelift

# CUDA 및 PyTorch 설치
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 필수 패키지 설치
pip install easydict rich wandb lpips einops
pip install accelerate transformers diffusers
```

### 또는 setup_env.sh 사용

```bash
bash setup_env.sh
```

---

## 1.3 체크포인트 다운로드

```bash
# 자동 다운로드 스크립트 실행
python scripts/download_checkpoints.py
```

**다운로드되는 파일:**
- `checkpoints/mvdiffusion/` - Multi-view Diffusion 모델
- `checkpoints/gslrm/ckpt_*.pt` - GS-LRM pretrained weights

---

## 1.4 개발 브랜치 생성

```bash
# Mouse 적용을 위한 새 브랜치 생성
git checkout -b feature/mouse-adaptation
```

---

## 1.5 원본 코드 구조 확인

```bash
# 핵심 파일 확인
ls -la train_gslrm.py
ls -la gslrm/data/dataset.py
ls -la configs/gslrm.yaml
```

### 원본 구조 설명

```
FaceLift/
├── train_gslrm.py           # GS-LRM 학습 메인 스크립트
├── train_diffusion.py       # MVDiffusion 학습 스크립트
├── inference.py             # 추론 스크립트
├── gslrm/
│   ├── data/
│   │   └── dataset.py       # RandomViewDataset 정의
│   └── model/
│       ├── gslrm.py         # GS-LRM 모델
│       ├── gaussians_renderer.py  # Gaussian Splatting 렌더러
│       └── utils_*.py       # 유틸리티 함수들
├── mvdiffusion/
│   ├── models/              # Diffusion 모델
│   └── pipelines/           # 추론 파이프라인
├── configs/
│   ├── gslrm.yaml           # GS-LRM 학습 config
│   └── mvdiffusion.yaml     # MVDiffusion 학습 config
└── data_sample/             # 데이터 구조 예시
```

---

## 1.6 테스트 실행 (선택적)

원본 코드가 정상 동작하는지 확인:

```bash
# 추론 테스트 (examples/ 폴더에 이미지 필요)
python inference.py --input_dir examples/ --output_dir outputs/

# 또는 Gradio 앱 실행
python gradio_app.py
```

---

## 다음 단계

✅ Fork 및 환경 설정 완료

→ [Step2: MouseViewDataset 구현](./Step2_Mouse_Dataset.md)

---

*Created: 2026-01-13*
