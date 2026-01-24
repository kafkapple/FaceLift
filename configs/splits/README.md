# Split Configuration System

## 개요

데이터셋 하위 버전으로 다양한 split 전략 관리.

## 버전 체계

```
M3      ← 기본 (samples/ 폴더)
M3.r    ← Random split (90/10)
M3.t    ← Temporal 3분할 (train/val/test)
M3.s    ← Temporal Stratified (Pose Splatter style)
```

## Split 전략

| 전략 | 설명 | 용도 |
|------|------|------|
| `random` | 무작위 셔플 | 일반적 실험 |
| `temporal` | 시간순 분할 (앞/중간/뒤) | 시간 일반화 테스트 |
| `temporal_stratified` | 시간 구간별 균등 샘플링 | 전체 시간 분포 유지 |

## 사용법

```bash
# Split 생성
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 \
    --config configs/splits/temporal_stratified.yaml

# 또는 프리셋 사용
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 \
    --preset pose_splatter
```
