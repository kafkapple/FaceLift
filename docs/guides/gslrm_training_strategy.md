# GS-LRM 학습 전략 가이드

## 실행 방법

### 포그라운드 실행 (터미널에서 직접 확인)
```bash
python train_gslrm.py --config configs/mouse_gslrm_v2.yaml
```

### 백그라운드 실행 (터미널 종료해도 계속)
```bash
mkdir -p logs
nohup python train_gslrm.py --config configs/mouse_gslrm_v2.yaml \
    > logs/train_gslrm_v2.log 2>&1 &

# 로그 확인
tail -f logs/train_gslrm_v2.log
```

---

## 프로세스 관리

### 조회
```bash
# 학습 프로세스 확인
ps aux | grep train_gslrm

# GPU 사용 현황
nvidia-smi

# 현재 쉘의 백그라운드 작업
jobs -l
```

### 종료
```bash
# 특정 프로세스 종료
kill <PID>

# 강제 종료
kill -9 <PID>

# 모든 GS-LRM 학습 종료
pkill -f train_gslrm
```

---

## 현재 상태

| 버전 | 체크포인트 | Steps | 상태 |
|------|-----------|-------|------|
| pretrained | `ckpt_0000000000021125.pt` | 21,125 | FaceLift 인간 모델 |
| mouse_finetune v1 | `ckpt_0000000000020000.pt` | 20,000 | Mouse 1차 fine-tune |
| **mouse_v2** | (학습 예정) | +30,000 | 추가 최적화 |

---

## 학습 전략: Phase별 접근

### Phase 1: 빠른 POC (완료)
- MVDiffusion + Mouse embeds 적용
- 기존 GS-LRM mouse_finetune 사용
- **결과**: 성능 개선 확인됨

### Phase 2: GS-LRM 추가 학습 (현재)
```
목표: 30,000 steps 추가 학습
시작점: mouse_finetune/ckpt_20000
예상 시간: 6-12시간
```

### Phase 3: (선택) 더 긴 학습
- Phase 2 결과 확인 후 결정
- 필요시 50,000+ steps

---

## v2 학습 설정 변경 사항

| 파라미터 | v1 (기존) | v2 (신규) | 이유 |
|----------|----------|----------|------|
| Learning Rate | 2e-05 | **1e-05** | 안정적 fine-tuning |
| LPIPS weight | 0.5 | **0.8** | perceptual 품질 강화 |
| Perceptual weight | 0.5 | **0.8** | 시각적 품질 강화 |
| SSIM weight | 0.2 | **0.3** | 구조 유사성 강화 |
| Max steps | 20,000 | **50,000** | 충분한 학습 |
| Checkpoint every | 1,000 | **2,000** | 저장 공간 절약 |

---

## 학습 명령어

### 기본 실행
```bash
cd ~/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

python train_gslrm.py --config configs/mouse_gslrm_v2.yaml
```

### 백그라운드 실행 (권장)
```bash
nohup python train_gslrm.py --config configs/mouse_gslrm_v2.yaml \
    > logs/train_gslrm_v2.log 2>&1 &

# 진행 확인
tail -f logs/train_gslrm_v2.log

# WandB에서 확인
# Project: mouse_facelift, Group: gslrm
```

### 특정 GPU 지정
```bash
CUDA_VISIBLE_DEVICES=0 python train_gslrm.py --config configs/mouse_gslrm_v2.yaml
```

---

## 체크포인트 저장 위치

```
checkpoints/gslrm/
├── ckpt_0000000000021125.pt          # pretrained (인간)
├── mouse_finetune/                    # v1
│   ├── ckpt_0000000000018000.pt
│   ├── ckpt_0000000000019000.pt
│   └── ckpt_0000000000020000.pt       # 현재 최신
└── mouse_v2/                          # v2 (학습 예정)
    ├── ckpt_0000000000022000.pt
    ├── ckpt_0000000000024000.pt
    └── ...
```

---

## 테스트 명령어

### Full Pipeline 추론 (학습 완료 후)

**최종 조합: MVDiffusion + Mouse embeds + GS-LRM v2**
```bash
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_v2 \
    --output_dir outputs/pipeline_gslrm_v2
```

### 다른 샘플로 테스트
```bash
# sample_000001 테스트
python test_full_pipeline.py \
    --input_image data_mouse/sample_000001/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_v2 \
    --output_dir outputs/pipeline_gslrm_v2_sample001
```

### v1 vs v2 비교 테스트
```bash
# v1 (기존)
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_finetune \
    --output_dir outputs/compare_v1

# v2 (신규)
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_v2 \
    --output_dir outputs/compare_v2
```

### 학습 중간 테스트 (특정 체크포인트)
```bash
# 체크포인트 번호 직접 지정
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_v2/ckpt_0000000000030000.pt \
    --output_dir outputs/test_v2_30k
```

---

## 성능 비교 기준

| 메트릭 | 측정 방법 |
|--------|----------|
| 시각적 품질 | 렌더링 이미지 확인 |
| PSNR/SSIM | validation 로그 |
| 3D 일관성 | turntable 영상 확인 |

---

## 예상 일정

| 단계 | 소요 시간 | 결과물 |
|------|----------|--------|
| v2 학습 시작 | 즉시 | - |
| 첫 체크포인트 | ~1시간 | ckpt_22000 |
| 중간 확인 | ~3시간 | ckpt_30000 |
| 완료 | ~6-12시간 | ckpt_50000 |
