# Session Handoff: 260319 PM → Next Session

> Copy-paste this as the first message in the next Claude Code session at `~/dev/FaceLift`

---

## 다음 세션 시작 메시지

```
FaceLift BehaviorSplatter 프로젝트 이어서 진행.

## 이전 세션 완료 (260319 PM, 대규모 논문+PoC 세션)
- 논문 개요 v2.0 작성 (Obsidian neurips/PAPER_OUTLINE_v2.0.md)
- Baselines & Metrics 상세 레퍼런스 (neurips/BASELINES_AND_METRICS.md)
- PoC 실행 계획 + Hypothesis Matrix (neurips/POC_EXECUTION_PLAN.md)
- Sparse PoC 완료 (500 exp): S1(skeleton)>S3(engineered), Sil=0.43, K=4, 5K sweet spot
- BAMS 학습 완료 (100 epochs, loss 47→16, embedding 179K×128d)
- BAMS vs S1 비교: BAMS_long Sil≈S1(0.359 vs 0.362), 그러나 TPI=0.90, bout=60ms (flickering)
- Ethogram + cluster distribution 시각화 생성
- GS-LRM Gaussian count 실측: foreground ~15-20K (usage ~1%), PS ~8.5K → 2x 차이
- 환경 사고 복구: facelift(torch2.6+cu124), bams(torch2.0) 별도 env 구성
- environment-{facelift,bams}.yml 저장

## 현재 상태
- GPU 4/5/6: alpha sweep 3개 ~30% 진행 중 (alpha 0.3/1.0/0.5)
- bams env: gpu03에 설치 완료 (nerdslab/bams)
- sdannce env: joon 서버에 설치 완료
- s-DANNCE PoC 데이터: joon:/home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4
- Sparse features: gpu03:outputs/sdannce_poc/features/sparse_features.npz
- BAMS embeddings: gpu03:outputs/sdannce_poc/bams/bams_embeddings.npz

## 오늘 할 일 (우선순위)
1. **SocialMapper 실행** → 9 HLACs pseudo-GT 생성 (joon 서버, sdannce env)
   - SocialMapper 과정 상세 조사: 어떤 feature → 어떤 clustering → 어떤 기준으로 naming
   - pseudo-GT 검증 방법 문서화
2. **Classification PoC** — S1 vs S3 vs BAMS → 9 HLACs, linear probe (가장 중요한 비교)
3. **BAMS temporal smoothing** — Gaussian σ=10-20f 적용 후 clustering 재평가
4. **Per-cluster skeleton GIF** + HTML report (기존 visualize_clusters.py 활용, s-DANNCE 23kp 호환 필요)
5. **종합 결과 문서** (Obsidian neurips/)
6. **Alpha sweep 완료 확인** → best checkpoint → GS-LRM inference → Dense features

## 주의사항
- facelift env: torch 2.6.0+cu124, numpy 1.26.4 (복구 완료, 검증됨)
- bams env: torch 2.0.0+cu117 (별도 env, BAMS 전용)
- sdannce env: joon 서버 miniconda3 (SocialMapper 전용)
- 환경 간 pip install로 dependency 파괴 금지! 반드시 별도 env 사용
- s-DANNCE behavior labels = SocialMapper pseudo-GT (수동 annotation 아님)
```

---

## 핵심 파일 위치

### Obsidian (논문/전략)
| 파일 | 역할 |
|------|------|
| `neurips/PAPER_OUTLINE_v2.0.md` | 논문 개요 v2.1 (최신) |
| `neurips/BASELINES_AND_METRICS.md` | 3-Stage 파이프라인별 비교 대상 상세 |
| `neurips/POC_EXECUTION_PLAN.md` | Hypothesis Matrix + 실행 계획 |
| `neurips/POC_RESULTS_SPARSE_CLUSTERING.md` | Sparse clustering 결과 |
| `docs/INDEX.md` | 문서 허브 (v7.3) |

### gpu03 (코드/데이터)
| 파일 | 역할 |
|------|------|
| `outputs/sdannce_poc/features/sparse_features.npz` | S1+S3 features (90K frames) |
| `outputs/sdannce_poc/bams/bams_embeddings.npz` | BAMS 128d embeddings (179K) |
| `outputs/sdannce_poc/bams/best_model.pt` | BAMS trained model |
| `outputs/sdannce_poc/bams_analysis/` | Ethogram + cluster distribution PNGs |
| `outputs/sdannce_poc/clustering_poc/` | 500-experiment clustering results |
| `mouse_extensions/behavior/extract_sdannce_features.py` | S1/S3 추출 |
| `mouse_extensions/behavior/run_bams_baseline.py` | BAMS 학습/embedding |
| `mouse_extensions/behavior/run_clustering_poc.py` | Clustering PoC |
| `mouse_extensions/behavior/analyze_bams_embeddings.py` | BAMS 분석 |
| `mouse_extensions/behavior/plot_bams_comparison.py` | 시각화 |
| `environment-facelift.yml` | facelift env spec |
| `environment-bams.yml` | bams env spec |

### joon 서버
| 파일 | 역할 |
|------|------|
| `/home/joon/dev/sdannce-poc/data/2022_09_22_M3_M4/` | s-DANNCE PoC 세션 |
| `/home/joon/dev/sdannce-poc/sdannce/socialmapper/` | SocialMapper 파이프라인 |
| `/home/joon/dev/sdannce-poc/outputs/` | Feature + clustering 결과 |

---

## SocialMapper 실행 명령 (다음 세션 즉시)

```bash
# joon 서버에서
ssh joon
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sdannce
cd /home/joon/dev/sdannce-poc

# SocialMapper demo를 참고하여 우리 세션에 적용
python sdannce/socialmapper/examples/demo_analysis.py  # 먼저 demo 확인
# 또는 직접 파이프라인 실행 (session path 지정)
```

---

*Created: 2026-03-19*
