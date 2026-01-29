# NFS Stale File Handle Error - Debug Guide

## 문제 개요

**오류**: `OSError: [Errno 116] Stale file handle`

**발생 환경**: gpu03 서버에서 NFS 마운트(`10.2.11.1:/home/joon`)에 wandb 로그 저장 시

**증상**:
- wandb 이미지 로깅 중 파일 핸들 유효하지 않음
- 학습 중간에 랜덤하게 발생 (step 2000~3000 사이)
- 주로 `wandb.Image()` 호출 시 temp 파일 → media 폴더 이동 과정에서 발생

## 원인

NFS (Network File System)에서 파일 핸들이 "stale" 상태가 되는 경우:
1. 서버 측에서 파일/디렉토리가 삭제되거나 이동됨
2. NFS 클라이언트 캐시와 서버 상태 불일치
3. 네트워크 일시적 불안정으로 핸들 유효성 손실
4. wandb의 빈번한 temp 파일 생성/이동이 NFS와 충돌

## 해결책

### 핵심: wandb 로그를 로컬 디스크에 저장

**NFS 마운트** (느림, stale handle 위험):
```
/home/joon/dev/FaceLift/wandb_logs/  ← NFS (10.2.11.1:/home/joon)
```

**로컬 디스크** (빠름, 안정적):
```
/node_data/joon/wandb_logs/          ← 로컬 SSD (/dev/nvme1n1p1)
```

### 적용된 수정 (train_gslrm.py)

```python
# Create wandb directory - prefer local storage to avoid NFS stale file handle errors
local_wandb_dir = "/node_data/joon/wandb_logs"
if os.path.isdir("/node_data/joon"):
    wandb_dir = local_wandb_dir
else:
    wandb_dir = "wandb_logs"
os.makedirs(wandb_dir, exist_ok=True)
```

### 프로젝트별 디렉토리 구조

```
/node_data/joon/wandb_logs/
├── FaceLift/          # GS-LRM, MVDiffusion 학습
├── MAMMAL/            # MAMMAL fitting
├── pose-splatter/     # Pose Splatter 학습
└── flux_cond_img/     # FLUX 이미지 생성
```

### 심볼릭 링크 설정 (각 프로젝트에서)

```bash
# FaceLift
cd /home/joon/dev/FaceLift
rm -rf wandb_logs 2>/dev/null
ln -s /node_data/joon/wandb_logs/FaceLift wandb_logs

# MAMMAL
cd /home/joon/dev/MAMMAL_mouse
rm -rf wandb_logs 2>/dev/null
ln -s /node_data/joon/wandb_logs/MAMMAL wandb_logs

# pose-splatter
cd /home/joon/dev/pose-splatter
rm -rf wandb_logs 2>/dev/null
ln -s /node_data/joon/wandb_logs/pose-splatter wandb_logs
```

## 확인 방법

```bash
# 디스크 유형 확인
df -h /node_data/joon/wandb_logs
# 결과: /dev/nvme1n1p1 (로컬 SSD)

# NFS 마운트 확인
df -h /home/joon
# 결과: 10.2.11.1:/home/joon (NFS)
```

## 관련 파일

- `train_gslrm.py`: wandb_dir 설정 (line 558-563)
- `/node_data/joon/wandb_logs/`: 로컬 wandb 저장소

## 참고

- Errno 116 = ESTALE (Stale NFS file handle)
- wandb는 temp 파일 생성 후 rename/move 수행 → NFS에서 race condition 발생 가능
- 로컬 디스크 사용 시 I/O 속도도 향상됨

---
*Created: 2026-01-29 | FaceLift M5 ablation 학습 중 발생*
