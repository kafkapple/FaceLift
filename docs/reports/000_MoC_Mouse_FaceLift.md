---
date: 2026-01-05
context_name: "2_Research"
tags: [moc, mouse-facelift, research-summary, gslrm, mvdiffusion]
project: Mouse-FaceLift
status: living-document
generator: ai-assisted
generator_tool: claude-code
---

# 🗺️ Map of Content: Mouse-FaceLift 연구 여정

> **목적**: FaceLift 파이프라인을 마우스 도메인에 적응하여 Single-view → Multi-view 3D 재구성
>
> **기간**: 2025-12-13 ~ 현재
>
> **핵심 파이프라인**: `Input Image → MVDiffusion → 6-view Images → GS-LRM → 3D Gaussians`

---

## 📊 연구 진행 타임라인

```
12/13 ─────── 12/14 ─────── 12/18 ─────── 12/19 ─────── 12/20 ─────── 12/21
  │             │             │             │             │             │
  ▼             ▼             ▼             ▼             ▼             ▼
2단계 전략   Mode Collapse  카메라 정규화  이슈 종합    Pixel-based   GS-LRM 한계
수립         해결           Z-up vs Y-up   문서화       전처리        분석

12/30 ─────── 12/31 ─────── 01/04 ─────── 01/05
  │             │             │             │
  ▼             ▼             ▼             ▼
뷰 순서      Loss 시스템    전처리        MoC 작성
Masked Loss  분석           종합 분석
```

---

## 🔬 Phase 1: 전략 수립 및 초기 디버깅 (12/13-12/14)

### 가설
> FaceLift pretrained 모델을 마우스 데이터에 fine-tuning하면 3D 재구성 가능

### 실험 결과

| 테스트 | GS-LRM 모델 | 카메라 | 결과 |
|--------|-------------|--------|------|
| pretrained (human) | pretrained | FaceLift | ✅ 성공 |
| mouse_finetune | fine-tuned | FaceLift | ❌ 실패 |
| mouse_finetune | fine-tuned | mouse | ❌ 실패 |

### 발견된 문제 & 해결

| 문제 | 원인 | 해결 | 관련 노트 |
|------|------|------|-----------|
| MVDiffusion Mode Collapse | Random reference view → Identity mapping | `reference_view_idx: 0` 고정 | [[251214_research_daily]] |
| Gradient Explosion | LR 2e-5가 너무 높음 | LR: 5e-6, warmup: 500 | [[251214_research_daily]] |

### 핵심 교훈
1. **Random Reference View 위험**: Identity mapping 학습의 지름길
2. **도메인 전이 시 LR 조정 필수**: 4-10배 낮추기

📝 **관련 노트**: [[251213_research_daily]], [[251214_research_daily]]

---

## 🔬 Phase 2: 카메라 정규화 (12/18-12/19)

### 가설
> 카메라 파라미터 불일치가 GS-LRM 실패의 원인

### 실험 결과

| 항목 | 인간 | 마우스 | 영향 |
|------|------|--------|------|
| 카메라 거리 | 2.7 고정 | 2.0~3.4 가변 | 🔴 Critical |
| 좌표계 | Z-up | ~Z-up | 🟡 Minor |
| `num_input_views` | 6 (학습) | 1 (잘못 설정) | 🔴 Critical |

### 발견된 문제 & 해결

| 문제 | 원인 | 해결 | 관련 노트 |
|------|------|------|-----------|
| White prediction | 카메라 거리 불일치 | 거리 2.7로 정규화 | [[251219_known_issues_and_solutions]] |
| PSNR 정체 (13-15) | `num_input_views=1` | `num_input_views=5` | [[251218_research_daily]] |
| Gradient explosion | LPIPS/Perceptual loss | weight=0 설정 | [[251218_research_daily]] |

### 핵심 교훈
1. **GS-LRM은 Z-up 좌표계** 사용 (Y-up 정규화가 오히려 역효과)
2. **Pretrained 분포 유지가 핵심**: 카메라 포맷은 협상 불가

📝 **관련 노트**: [[251218_research_daily]], [[251219_known_issues_and_solutions]]

---

## 🔬 Phase 3: 전처리 개선 (12/20-12/21)

### 가설
> Bbox 기반 전처리가 불규칙 형상(꼬리)에서 실패 → Pixel-based 전처리 필요

### 실험 결과

| Dataset | Size CV | CoM Offset | Grade |
|---------|---------|------------|-------|
| **data_mouse_pixel_based** | **0.16%** | **1.5px** | **A** |
| data_mouse_uniform | 8.92% | 38.3px | D |
| data_mouse_centered | 12.82% | 36.0px | D |

### 발견된 문제 & 해결

| 문제 | 원인 | 해결 | 관련 노트 |
|------|------|------|-----------|
| 전처리 품질 불균일 | Bbox가 꼬리에 왜곡 | CoM + pixel count 방식 | [[251220_research_daily]] |
| 합성 데이터 오류 | 원본 카메라 복사 | FaceLift 표준 카메라 생성 | [[251220_critical_synthetic_data_pipeline]] |
| **체크포인트 버그** | 잘못된 경로 → human 모델 사용 | 경로 검증 필수 | [[251221_research_gslrm_mvdiffusion_analysis]] |

### 🔴 GS-LRM Fine-tuning 실패 분석

**핵심 발견**: GS-LRM은 **Plücker Ray Encoding**으로 카메라 정보 전달
- `ray_direction = normalize(K_inv @ pixel_coord)`
- Intrinsics가 다르면 → 같은 픽셀도 완전히 다른 ray로 인식

**Fine-tuning 실험 결과**:
- Real Camera: Step 1부터 흰색 (즉시 실패)
- Freeze All: Catastrophic Forgetting (점진적 품질 저하)

### 핵심 교훈
1. **Fine-tuning보다 데이터 적응**: 모델 수정 < 데이터를 모델에 맞추기
2. **Fail-fast 원칙**: Warning보다 Error로 조기 실패

📝 **관련 노트**: [[251220_research_daily]], [[251220_critical_synthetic_data_pipeline]], [[251221_research_gslrm_mvdiffusion_analysis]]

---

## 🔬 Phase 4: Masked Loss & 뷰 순서 (12/30-12/31)

### 가설
> 배경이 95%인 이미지에서 일반 Loss가 배경을 지배 → Masked Loss 필요

### 실험 결과

| 문제 | 증상 | 해결 |
|------|------|------|
| 뷰 순서 랜덤화 | 출력이 점점 흐릿해짐 | `use_mouse_dataset: true` |
| 배경 지배 | 흰색 출력 학습 | Masked L2/SSIM loss |
| SSIM 음수 | 상수 영역 수치 불안정 | `clamp(ssim, 0, 1)` |

### 핵심 교훈
1. **뷰 순서 일관성**: 불균등 카메라 배열에서 랜덤 샘플링 위험
2. **Masked Loss**: 배경 비율 높을 때 필수

📝 **관련 노트**: [[251230_research_view_order_and_masked_loss]], [[251231_facelift_loss_system_analysis]]

---

## 🔬 Phase 5: 전처리 종합 분석 (01/04-01/05)

### 가설
> 샘플별 균일 scale이 뷰별 균일보다 3D 일관성 유지에 유리

### 실험 결과

| 전략 | 장점 | 단점 | 권장 |
|------|------|------|------|
| 뷰별 균일 | 시각적 일관성 | Plücker 불균일 | ❌ |
| **샘플별 균일** | 3D 일관성 | 뷰별 크기 차이 | ✅ |

### Clipping 문제 해결
- **원인**: 이미지 중심 기준 스케일링 (CoM 무시)
- **해결**: **Center of Mass 기반** 스케일링

📝 **관련 노트**: [[260104_research_preprocessing_comprehensive]], [[251231_mouse_data_preprocessing_analysis]]

---

## 📚 문서 색인

### 연구 일지 (Daily)
- [[251213_research_daily]] - 2단계 학습 전략 수립
- [[251214_research_daily]] - Mode Collapse & Gradient Explosion 해결
- [[251218_research_daily]] - 카메라 정규화 & num_input_views 수정
- [[251220_research_daily]] - Pixel-based 전처리 도입

### 기술 분석
- [[251219_known_issues_and_solutions]] - 🔴 **Living Document**: 모든 이슈 종합
- [[251220_critical_synthetic_data_pipeline]] - 합성 데이터 파이프라인 핵심
- [[251221_research_gslrm_mvdiffusion_analysis]] - GS-LRM & MVDiffusion 종합 분석
- [[251230_research_view_order_and_masked_loss]] - 뷰 순서 & Masked Loss
- [[251231_facelift_loss_system_analysis]] - Loss 시스템 상세 분석
- [[251231_mouse_data_preprocessing_analysis]] - 전처리 상태 점검
- [[260104_research_preprocessing_comprehensive]] - 전처리 종합 (Scaling & Clipping)

---

## ⚠️ 알려진 제한사항

### MVDiffusion 한계
- **고정된 뷰 인덱스**: 균등 60° 간격 가정
- 비균등 카메라 배열에서 **실제 6-view 직접 사용** 권장

### GS-LRM 한계
- **카메라 intrinsics 강한 의존**: Plücker Ray Encoding
- Fine-tuning 매우 취약 → **Zero-shot 또는 LoRA** 권장

### 전처리 주의사항
- **샘플별 균일 scale** 사용
- **CoM 기반 centering** 필수
- Intrinsics 변환 시 동기화 필수

---

## 🎯 현재 상태 & 다음 단계

### 완료됨 ✅
- [x] 2단계 학습 전략 수립
- [x] MVDiffusion mode collapse 해결
- [x] 카메라 정규화 파이프라인
- [x] Pixel-based 전처리 도입
- [x] Masked Loss 구현
- [x] 뷰 순서 고정

### 진행 중 🔄
- [ ] CoM 기반 스케일링 적용
- [ ] GS-LRM 학습 재개 (개선된 파이프라인)

### 향후 계획 📋
- [ ] Camera Pose Conditioning (Plücker Ray Embedding) 도입
- [ ] Zero123++ / SV3D 등 대안 모델 탐색
- [ ] 다양한 카메라 환경 일반화

---

*🤖 Generated with Claude Code*
*📝 Last updated: 2026-01-05*
