# `baselines/pose_splatter/` — 무효 아티팩트 정리 (260723)

## 삭제된 파일 2건

| 파일 | 값 | 삭제 사유 |
|---|---|---|
| `paper_standard_evaluation.json` | PSNR 24.68 / IoU 0.829 | 산출 run = `facelift_compare_5cam` — `fl_vs_ps_comparison.md` 가 **"different camera"로 폐기 선언**한 실험. full-image 프로토콜이라 FL과 비교 불가 |
| `posesplatter_fair.json` | psnr_gt_masked 16.80 / IoU 0.827 | 동일 폐기 run 의 fair eval. SSOT 인용값 `13.78` 과 무관 |

**복구**: `git checkout HEAD -- baselines/pose_splatter/<file>` (삭제 시점 git 추적 중이었음)

## 왜 무효인가

- 두 파일 모두 `exp_dir = output/facelift_compare_5cam/latest` 산출. 해당 output 디렉토리는 **gpu03에서 삭제됨**(260619) → 재계산 불가
- SSOT가 인용해 온 **`13.78 / 0.846` 은 이 파일들에 없음.** 그 값은 `m5_baseline_gs` 산출인데 체크포인트·전처리 데이터·전처리 코드가 모두 소실
- gpu03 `~/dev/pose-splatter/experiments/fair/` 의 3개 파일도 전부 같은 폐기 run 산출 (16.71~16.80). **`m5_baseline_gs` 산출은 0개**
  - 해당 파일들은 PS 프로젝트 자체 이력이므로 **미삭제** (본 정리 범위 밖)

## 현재 상태

**유효한 PS baseline 아티팩트 = 0개.** 신규 산출 전까지 PS 비교 수치 인용 금지.

- 상세: `docs/FACELIFT_SSOT.md` §2.1 **C5** · §3 **C5** · §5.1 · Drift Ledger **D10~D14**
- 복구 계획: `outputs/reports/260723_ps_recovery_plan.md`
- 감사 기록: `outputs/reports/260723_ps_baseline_integrity_audit.md`
