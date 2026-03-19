# Session Handoff: 260319 → Next Session

> Copy-paste this as the first message in the next Claude Code session at `~/dev/FaceLift`

---

## 다음 세션 시작 메시지

```
FaceLift BehaviorSplatter 프로젝트 이어서 진행.

## 현재 상태
- GPU 4: 6v alpha=0.3 학습 중 (~15K steps)
- GPU 5: 6v alpha=1.0 학습 중 (~15K steps)
- GPU 6: 6v alpha=0.5 full dataset (3600 frames) 학습 중
- 3개 모두 완료 여부 확인 필요

## 이전 세션 완료 (260319, 20+ 작업)
- ERR (Effective Rank Regularization) 구현 완료 → `configs/experiments/6view_alpha05_err.yaml`
- Strategy C temporal features 추출 완료 (129d, 3585 frames)
- 비교 모듈 구축 (`mouse_extensions/visualization/comparison/`)
- 카메라 좌표계 버그 수정 (orbit azimuth 270° 시작, clockwise 방향, gimbal lock guard)
- evaluation/visualization.py 복원 (학습 에러 해결)
- 문헌 조사 25편 → `docs/experiments/RELATED_WORK_SURVEY.md`
- Novel view quality 전략 → `docs/experiments/NOVEL_VIEW_QUALITY_STRATEGY.md`

## 오늘 할 일 (우선순위)
1. **GPU 학습 완료 확인 + 결과 분석** — alpha sweep Pareto front, comparison grid
2. **NeurIPS 논문 윤곽 + 벤치마크 태스크 정의** — 논문이 모든 실험 방향 결정
3. **Difix3D+ zero-shot PoC** — nvidia/difix 모델로 PoC 3프레임 테스트 (30분)
4. **ERR sweep 실험 실행** — GPU 비면 즉시 (weight 0.001/0.01/0.1)
5. **Dataset hosting/licensing 계획 시작** — NeurIPS Dataset Track 필수

## 전략 전환
- 코드 인프라 → 논문/데이터셋 준비로 전환 (3-model deliberation 합의)
- W1-2: 데이터셋 준비 50%, W3-4: 논문 작성 50%

## 주의
- 서버에서만 변경된 파일 있음 (스크립트 이동, visualization.py 복원) → 필요시 서버→로컬 pull
- MV-Adapter는 P3 후순위 보류
- `configs/mouse/uniform/`에 레거시 config 23개 → 정리 필요 시 `_archive/`로
```

---

## 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `docs/experiments/NOVEL_VIEW_QUALITY_STRATEGY.md` | ERR→Difix→2DGS 전략 |
| `docs/experiments/RELATED_WORK_SURVEY.md` | 25편 문헌 서베이 |
| `docs/experiments/DIFIX_TRAINING_STRATEGY.md` | DiFix 2.5-stage curriculum |
| `mouse_extensions/model/mask_losses.py` | ERR loss (`compute_effective_rank_loss`) |
| `mouse_extensions/behavior/extract_temporal_features.py` | Strategy C (129d) |
| `mouse_extensions/visualization/comparison/` | 비교 모듈 (5 presets + grid) |
| `configs/experiments/6view_alpha05_err.yaml` | ERR 실험 config |
| `configs/experiments/6view_alpha05_full.yaml` | Full dataset config |
| `configs/comparison/example_alpha_sweep.yaml` | 비교 모듈 예시 config |

## GPU 학습 완료 후 즉시 실행

```bash
# 1. 학습 완료 확인
ssh gpu03 "for f in alpha03 alpha10 alpha05_full; do echo === $f ===; tail -3 /home/joon/dev/FaceLift/outputs/train_6view_${f}*.log; done"

# 2. Best checkpoint 확인
ssh gpu03 "find /node_data/joon/checkpoints/FaceLift/gslrm/ -name 'best_psnr.pt' -newer /home/joon/dev/FaceLift/outputs/train_6view_alpha03_v3.log"

# 3. ERR 실험 실행 (GPU 비면)
ssh gpu03 "cd /home/joon/dev/FaceLift && CUDA_VISIBLE_DEVICES=4 nohup ... train_gslrm.py -d M5t2 -e 6view_alpha05_err > outputs/train_6view_alpha05_err.log 2>&1 &"
```

---

*Created: 2026-03-19*
