---
date: 2024-12-15
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, 3d-reconstruction, overview]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Mouse-FaceLift 프로젝트 연구 개요

## 1. 프로젝트 목표

**FaceLift 파이프라인을 마우스 도메인에 적용하여 단일 뷰 이미지에서 3D 재구성 수행**

### 원본 FaceLift 파이프라인
```
Single Image → MVDiffusion (6-view 생성) → GS-LRM (3D Gaussian)
```

### 적용 대상
- **도메인**: 실험실 마우스 (laboratory mouse)
- **입력**: 단일 뷰 마우스 이미지
- **출력**: 360° 회전 가능한 3D Gaussian 모델

---

## 2. 실험 단계 구조

### Stage 1: MVDiffusion (Multi-View Generation)
| 항목 | 내용 |
|------|------|
| **목적** | 단일 이미지 → 6개 뷰 이미지 생성 |
| **베이스 모델** | Zero123++ (Objaverse pretrained) |
| **주요 실험** | 카메라 정보 주입, 프롬프트 임베딩 |
| **결과** | 비교적 성공적, Fine-tuning 효과 있음 |

### Stage 2: GS-LRM / LGM (3D Reconstruction)
| 항목 | 내용 |
|------|------|
| **목적** | 6개 뷰 이미지 → 3D Gaussian Splatting |
| **베이스 모델** | GS-LRM (Human pretrained), LGM (Objaverse pretrained) |
| **주요 실험** | Freeze layer, Conservative training, Ray embedding |
| **결과** | **여전히 도전적**, 2D-only supervision의 한계 |

---

## 3. 핵심 발견사항

### 성공 요인 (Stage 1)
1. Zero123++의 일반화 능력으로 마우스 도메인 학습 가능
2. 카메라 컨디셔닝이 view consistency 개선
3. 적절한 augmentation으로 overfitting 방지

### 실패 요인 (Stage 2)
1. **2D-only supervision**: 깊이(depth) 정보 없이 3D 학습의 한계
2. **Mode collapse**: GS-LRM 예측값이 회색 평균으로 수렴
3. **Catastrophic forgetting**: Fine-tuning 시 Objaverse 3D prior 손실
4. **View-aligned Gaussian**: 진정한 3D geometry 학습 불가

### 근본적 차이 (FaceLift vs Mouse-FaceLift)
| 요소 | FaceLift | Mouse-FaceLift |
|------|----------|----------------|
| Pretrain 데이터 | Objaverse (3D 모델) | Human/Objaverse |
| 3D Supervision | Depth loss 포함 | **2D rendering loss만** |
| Fine-tune | 적은 양으로 충분 | Fine-tune 시 prior 손실 |

---

## 4. 결론

### 현재 상태
1. **Stage 1 (MVDiffusion)**: 어느 정도 성공적
   - Fine-tuning으로 마우스 6개 뷰 생성 가능
   - View consistency는 개선 여지 있음

2. **Stage 2 (3D Reconstruction)**: 여전히 도전적
   - 2D supervision만으로는 진정한 3D 학습 어려움
   - **Pretrained 모델 직접 사용이 가장 나은 결과**

### 권장 접근법
```
현재 최선: Pretrained LGM/GS-LRM + 최소한의 Fine-tuning (or No Fine-tuning)
```

### 향후 방향
1. **Depth supervision 추가**: 마우스 3D 스캔 데이터 활용
2. **View-aligned 구조 개선**: 진정한 3D prior 학습
3. **Domain gap 줄이기**: Objaverse에 마우스 유사 객체 추가

---

## 5. 관련 연구 노트

| 문서 | 내용 |
|------|------|
| `251215_research_stage1_mvdiffusion.md` | Stage 1 상세 실험 기록 |
| `251215_research_stage2_3d_reconstruction.md` | Stage 2 상세 실험 기록 |
| `251213_research_two_phase_training_strategy.md` | 2단계 학습 전략 |
| `251212_pipeline_analysis_report.md` | 파이프라인 분석 |

---

## 6. 주요 Confounding Factors

> 아래 요소들은 실험 과정에서 복합적으로 작용하여 개별 효과 분리가 어려움

1. **카메라 파라미터 vs 데이터 품질**
   - 카메라 수정과 데이터 전처리가 동시에 변경된 경우 있음

2. **Freeze layer vs Learning rate**
   - Conservative training에서 두 요소가 함께 적용됨

3. **Mode collapse vs Domain gap**
   - Human pretrained 모델의 domain gap인지, 학습 자체의 문제인지 분리 필요

4. **Ray embedding bug vs Architecture limitation**
   - 버그 수정 후에도 view-aligned Gaussian의 근본적 한계 존재
