# 260129 Debug: NFS Stale File Handle

> **Date**: 2026-01-29
> **Type**: Debug / Infrastructure
> **Status**: ✅ Resolved

---

## 문제

**오류**: `OSError: [Errno 116] Stale file handle`

**발생 환경**: gpu03에서 NFS 마운트에 wandb 로그 저장 시

**증상**:
- wandb 이미지 로깅 중 파일 핸들 유효하지 않음
- 학습 중간에 랜덤하게 발생 (step 2000~3000)
- `wandb.Image()` 호출 시 temp 파일 → media 폴더 이동 과정에서 발생

---

## 원인

NFS (Network File System)에서 파일 핸들이 "stale" 상태가 되는 경우:
1. 서버 측에서 파일/디렉토리가 삭제되거나 이동됨
2. NFS 클라이언트 캐시와 서버 상태 불일치
3. 네트워크 불안정으로 핸들 유효성 손실
4. wandb의 빈번한 temp 파일 생성/이동이 NFS와 충돌

---

## 해결책

### 핵심: wandb 로그를 로컬 디스크에 저장

| 경로 | 유형 | 결과 |
|------|------|------|
| /home/joon/.../wandb_logs/ | NFS | ❌ stale handle 위험 |
| **/node_data/joon/wandb_logs/** | 로컬 SSD | ✅ 안정적, 빠름 |

### 적용된 수정 (train_gslrm.py)

```python
local_wandb_dir = "/node_data/joon/wandb_logs"
if os.path.isdir("/node_data/joon"):
    wandb_dir = local_wandb_dir
else:
    wandb_dir = "wandb_logs"
os.makedirs(wandb_dir, exist_ok=True)
```

### 프로젝트별 심볼릭 링크

```bash
# FaceLift
cd /home/joon/dev/FaceLift
ln -s /node_data/joon/wandb_logs/FaceLift wandb_logs

# MAMMAL, pose-splatter도 동일하게 설정
```

---

*FaceLift Debug Notes | 2026-01-29*
