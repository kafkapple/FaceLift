# 전처리 요소별 영향 분석 실험 계획서

> **목적**: 각 전처리 요소가 3D 재구성 품질에 미치는 영향 정량화
> **작성일**: 2026-01-25
> **버전**: v2.0

---

## 1. 핵심 가설 (Hypotheses)

### H1: FG Coverage가 PSNR을 결정한다
**가설**: Coverage↑ → Gradient 신호↑ → 학습 효과↑ → PSNR↑

| 근거 | Dataset | Coverage | Val PSNR |
|------|---------|----------|----------|
| 관찰 1 | D3_normalized | ~6% | **27.0** |
| 관찰 2 | D7_1 | 50% | 20.7 |
| 관찰 3 | D8 | 50% | 19.7 |

**예측**: M3_norm (Coverage ~6%) → Val PSNR ≥ 27

### H2: fx 정규화는 Pretrained 호환성에 필수
**가설**: fx=549 (pretrained 분포) 필수, fx≠549 → 학습 불안정/실패

| 설정 | Dataset | fx | 예상 결과 |
|------|---------|-----|-----------|
| 정규화 | D7_1, D8, M3_norm | 549 | 안정적 학습 |
| 미정규화 | M3 (버그) | 739 | 불안정/낮은 PSNR |

### H3: PP 가변은 기하학적으로 정확하며 문제없다
**가설**: GS-LRM이 fxfycxcy 텐서로 가변 PP 처리 가능

| 설정 | Dataset | PP | 예상 결과 |
|------|---------|-----|-----------|
| 가변 (정확) | M3_norm | std~60 | 정상 학습 |
| 강제 256 | 가상 M3_norm_v1 | 0 | ray 오류 가능 |

### H4: fx/fy 비율(Aspect Ratio) 보존이 유리할 수 있다
**가설**: 원본 센서 비율 유지 → 기하학적 왜곡 감소 → PSNR↑

| 설정 | Dataset | fx/fy | Val PSNR |
|------|---------|-------|----------|
| 비정방형 유지 | D3_normalized | 0.9937 | 27.0 |
| 정방형 강제 | D7_1, D8 | 1.0000 | ~20 |

**주의**: D3_norm의 높은 PSNR이 비율 때문인지 Coverage 때문인지 분리 필요

---

## 2. 비교 데이터셋 (Experimental Groups)

### 2.1 주요 비교군

| ID | Dataset | Coverage | fx | PP | fx/fy | 역할 |
|----|---------|----------|-----|-----|-------|------|
| **A** | D7_1 | 50% | 549 | 256 | 1.0 | 기준선 (Baseline) |
| **B** | D3_normalized | ~6% | 549 | 256 | 0.99 | 고Coverage+비정방형 |
| **C** | M3 | 80% | 739❌ | 가변 | 1.0 | 버그 버전 |
| **D** | **M3_norm** | 80% | 549✅ | 가변 | 1.0 | **★ 최적 후보** |

### 2.2 가설별 비교쌍

| 가설 | 비교 | 통제 변수 | 실험 변수 |
|------|------|-----------|-----------|
| **H1** | A vs D | fx=549, PP정책 | Coverage (~3% vs ~6%) |
| **H2** | C vs D | Coverage, PP | fx (739 vs 549) |
| **H3** | D vs D_pp256 | Coverage, fx | PP (가변 vs 강제256) |
| **H4** | A vs B | fx=549 | fx/fy (1.0 vs 0.99) |

### 2.3 추가 실험 (선택적)

| ID | Dataset | 목적 |
|----|---------|------|
| E | D7_1_aspect | H4 검증: D7_1 + 비정방형 비율 |
| F | M3_norm_pp256 | H3 검증: M3_norm + PP 강제 256 |

---

## 3. 평가 지표 (Metrics)

### 3.1 정량 지표 (Quantitative)

| 지표 | 측정 대상 | 성공 기준 |
|------|-----------|-----------|
| **Val PSNR** | 재구성 품질 | ≥25 (양호), ≥27 (우수) |
| **Train PSNR** | 학습 수렴 | 수렴 확인 |
| **PSNR Gap** | Overfitting | Gap < 5 (정상) |
| **Val SSIM** | 구조적 유사도 | ≥0.85 |
| **Val LPIPS** | 지각적 품질 | ≤0.15 |

### 3.2 정성 지표 (Qualitative)

| 지표 | 확인 방법 | 판단 기준 |
|------|-----------|-----------|
| **Ghosting** | Turntable 영상 | 잔상/이중상 없음 |
| **Shape 정확도** | Novel view | 형태 왜곡 없음 |
| **Texture 선명도** | 확대 검사 | 털 디테일 보존 |
| **배경 분리** | Alpha mask | 깔끔한 경계 |

### 3.3 학습 안정성 지표

| 지표 | 측정 방법 | 문제 징후 |
|------|-----------|-----------|
| Loss 수렴 | WandB 그래프 | 발산, 진동 |
| NaN 발생 | 로그 확인 | 학습 중단 |
| 수렴 속도 | Steps to plateau | >50k는 느림 |

---

## 4. 결과 해석 기준 (Decision Rules)

### 4.1 H1 검증: Coverage → PSNR

```
비교: A (50%, PSNR ~20) vs D (80%, PSNR ?)

IF D.PSNR ≥ 25:
    → H1 지지: Coverage가 PSNR 결정
    → 결론: Adaptive zoom 유효
    
IF D.PSNR < 22:
    → H1 기각: Coverage 외 다른 요인
    → 추가 분석 필요
```

### 4.2 H2 검증: fx 정규화 필수성

```
비교: C (fx=739) vs D (fx=549)

IF C 학습 실패/불안정:
    → H2 지지: fx 정규화 필수
    
IF C.PSNR ≈ D.PSNR:
    → H2 기각: GS-LRM이 fx 적응 가능
    → (놀라운 결과, 추가 검증 필요)
```

### 4.3 H3 검증: PP 가변 허용

```
비교: D (PP 가변) vs F (PP=256 강제)

IF D.PSNR ≥ F.PSNR:
    → H3 지지: PP 가변이 더 정확하거나 동등
    → 결론: PP 강제 불필요
    
IF D.PSNR < F.PSNR - 2:
    → H3 기각: GS-LRM이 PP=256 선호
    → 결론: PP 강제 필요
```

### 4.4 H4 검증: Aspect Ratio 영향

```
비교: A (fx/fy=1.0) vs B (fx/fy=0.99)

주의: B는 Coverage도 다름 (~6% vs ~3%)

IF B.PSNR > A.PSNR:
    → 비정방형 OR Coverage 효과 (분리 불가)
    → E (D7_1_aspect) 실험으로 분리 필요
```

---

## 5. 실험 우선순위 및 일정

### Phase 1: 핵심 검증 (P0)

| 순서 | 실험 | 목적 | 소요 |
|------|------|------|------|
| 1 | M3_norm 전처리 | 최적 설정 생성 | ~2h |
| 2 | M3_norm + E1_2_gt_alpha | H1, H3 검증 | ~6h |
| 3 | M3 + E1_2_gt_alpha | H2 검증 (버그 상태) | ~6h |

### Phase 2: 추가 분석 (P1)

| 순서 | 실험 | 목적 | 조건 |
|------|------|------|------|
| 4 | M3_norm_pp256 | H3 상세 검증 | Phase 1 완료 후 |
| 5 | D7_1_aspect | H4 분리 검증 | H4 관심 시 |

---

## 6. 예상 결과 시나리오

### Best Case (가설 모두 지지)
```
M3_norm: Val PSNR 28+, Ghosting 없음
→ Coverage 80% + fx=549 + PP 정확 = 최적
→ Production 설정으로 채택
```

### Moderate Case (H1, H2 지지)
```
M3_norm: Val PSNR 25-27
M3: 학습 불안정
→ Coverage + fx 정규화 중요 확인
→ M3_norm 사용, 추가 최적화 여지
```

### Worst Case (가설 기각)
```
M3_norm: Val PSNR < 22 또는 Ghosting 심각
→ Coverage가 아닌 다른 요인 탐색 필요
→ D3_normalized 재분석, Split 특성 점검
```

---

## 7. 체크리스트

### 전처리 전
- [ ] M3_norm preset 설정 확인 (normalize_after_zoom=True, force_pp=False)
- [ ] 입력 데이터 경로 확인
- [ ] 출력 디렉토리 여유 공간 확인 (~50GB)

### 전처리 후
- [ ] 샘플 수 확인 (3597개 예상)
- [ ] fx 분포 확인 (549 ± 0.1)
- [ ] cx/cy 분포 확인 (가변, 평균 ~256)
- [ ] Coverage 확인 (~6%)
- [ ] 시각적 검사 (clipping 없음)

### 학습 후
- [ ] Val PSNR 기록
- [ ] Train/Val gap 확인
- [ ] Turntable 영상 검사
- [ ] WandB 로그 저장

---

*Created: 2026-01-25*
*FaceLift Mouse Preprocessing Experiment Plan v2.0*

---

## 8. 추가 가설: H5 (Per-sample Zoom)

### H5: 샘플별 가변 zoom이 일반화 성능을 높인다

**가설**: 다양한 zoom 레벨로 학습 → 다양한 크기의 생쥐에 대한 일반화 능력 향상

| 설정 | Dataset | Zoom | fx 분포 | 예상 |
|------|---------|------|---------|------|
| Global zoom | M3_norm | 1.35 고정 | 549 고정 | 일관적 |
| **Per-sample zoom** | M3_persample | 1.0~2.5 가변 | 549 고정 | 다양성↑ |

**비교 방법**: M3_norm vs M3_persample
- 동일 조건: fx=549 (정규화), PP=가변 (정확)
- 차이점: zoom 범위 (고정 vs 샘플별 적응)

**평가 기준**:
- Val PSNR: 일반화 성능
- Variance across samples: 일관성
- Unseen data 성능: 새로운 영상 테스트 (optional)

**전처리 명령어**:
```bash
# M3_persample (per-sample adaptive zoom)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_persample \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_persample
```

---

*Updated: 2026-01-25 - Added H5 (Per-sample Zoom)*

---

## 9. 최종 데이터셋 비교표 (Summary)

| Dataset | Zoom 방식 | fx | PP | 검증 가설 | 상태 |
|---------|-----------|-----|-----|-----------|------|
| D7_1 | 없음 | 549 | 256 | 기준선 | ✅ 존재 |
| D3_normalized | 알 수 없음 | 549 | 256 | 참조용 | ✅ 존재 |
| M3 | Global 1.35x | **739** ❌ | 가변 | H2 (버그 상태) | ✅ 존재 |
| **M3_norm** | Global 1.35x | **549** ✅ | 가변 | H1, H2, H3 | ⏳ 전처리 필요 |
| **M3_persample** | **Per-sample** | **549** ✅ | 가변 | **H5** | ⏳ 전처리 필요 |

---

## 10. 전처리 명령어 (Quick Reference)

```bash
cd /home/joon/dev/FaceLift

# M3_norm: Global zoom + fx 정규화 (H1, H2, H3)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_norm \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_norm

# M3_persample: Per-sample zoom + fx 정규화 (H5)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_persample \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_persample
```

---

## 11. 학습 명령어

```bash
# M3_norm 학습
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_norm -e E1_2_gt_alpha

# M3_persample 학습
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_persample -e E1_2_gt_alpha

# M3 (버그 상태) 학습 - H2 검증용
CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3 -e E1_2_gt_alpha
```

---

*Updated: 2026-01-25 - Added Summary, Commands, H5*
*Version: v2.1*
