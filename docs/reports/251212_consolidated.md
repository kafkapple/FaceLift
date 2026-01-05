# 251212: 파이프라인 분석 및 도메인 갭 이슈

**날짜:** 2025-12-12
**주제:** MVDiffusion + GS-LRM 파이프라인 문제점 분석
**상태:** ✅ 분석 완료

---

## 핵심 요약

| 테스트 | GS-LRM 모델 | 카메라 | 결과 |
|--------|-------------|--------|------|
| pretrained (human) | pretrained | FaceLift | ✅ 성공 |
| mouse_finetune | finetuned | FaceLift | ❌ 흐릿함 |
| mouse_finetune | finetuned | mouse | ❌ 실패 |

**핵심 발견**: MVDiffusion (FaceLift embeds) → 수평 뷰 출력 ↔ GS-LRM mouse (경사 뷰 기대) 불일치

---

## 1. 문제 진단

### 카메라 설정 불일치 체인
```
MVDiffusion prompt_embeds: FaceLift 기본 (수평, elevation=0°)
            ↓
MVDiffusion 출력: 수평 6뷰 생성
            ↓
GS-LRM mouse_finetune: Mouse 카메라 기대 (경사 20°)
            ↓
결과: 카메라 불일치 → 3D 복원 실패
```

### 데이터 흐름 분석
```
[입력 이미지]
      ↓ ⚠️ 임의 각도 입력
[MVDiffusion] ← 학습: top-down view only
      ↓
6x views (512x512) ⚠️ reference view 불일치
      ↓
[GS-LRM] ← 카메라 파라미터 불일치
      ↓
❌ 기형적인 3D 결과
```

---

## 2. 도메인 갭 분석

### 학습 vs 추론 특성
| 항목 | 학습 (합성) | 추론 (실제) |
|------|------------|------------|
| 배경 | 완벽히 분리 (alpha) | Segmentation 필요 |
| 노이즈 | 없음 | 센서 노이즈 |
| 조명 | 일관됨 | 다양함 |
| 카메라 포즈 | 정확함 | 추정 오차 |

### 해결책 옵션
1. **End-to-End (권장)**: MVDiffusion이 도메인 변환 수행
2. **데이터 증강**: 합성 이미지에 노이즈/블러 추가
3. **Mixed 학습**: 합성 70% + 실제 30%

---

## 3. 발견된 이슈 및 해결

### 체크포인트 경로 버그 (Critical)
```python
# Before (버그) - 경로 중복
model.save_pretrained(os.path.join(cfg.checkpoint_prefix, output_dir, "unet"))
# → checkpoints/checkpoints/experiments/.../unet

# After (수정)
model.save_pretrained(os.path.join(output_dir, "unet"))
```

### GSLRM Index Tensor 순서 버그
```python
# Before (잘못됨) - [scene_idx, view_idx]
index = torch.stack([
    torch.zeros(num_views).long(),   # scene_idx
    torch.arange(num_views).long()   # view_idx
], dim=-1)

# After (수정됨) - [view_idx, scene_idx]
index = torch.stack([
    torch.arange(num_views).long(),  # view_idx (첫 번째!)
    torch.zeros(num_views).long()    # scene_idx (두 번째)
], dim=-1)
```

---

## 4. 실험 우선순위

### Phase 1: MVDiffusion 학습
- [x] MVDiffusion fine-tune 진행
- [ ] 생성 품질 검증

### Phase 2: End-to-End 검증
- [ ] Real → MVDiffusion → GSLRM(pretrained) 테스트
- [ ] Real 6-view → GSLRM(finetuned) baseline 비교

### Phase 3: GSLRM 전략 결정
- [ ] MVDiffusion 출력으로 GSLRM 학습 (권장)

---

## 5. 최종 실험 결과

### GS-LRM 모델 비교
| 조합 | prompt_embeds | GS-LRM | 결과 |
|------|---------------|--------|------|
| 이전 | FaceLift | mouse_finetune | ❌ 흐릿 |
| Baseline | FaceLift | pretrained (human) | ✅ 선명 |

**결론**: MVDiffusion 출력이 FaceLift prompt_embeds 기준이므로, pretrained 모델이 더 잘 작동

---

## 관련 파일
- `test_full_pipeline.py` - 전체 파이프라인 테스트
- `inference_mouse.py` - 추론 스크립트

---

*통합 문서: 251212_pipeline_analysis_report.md + 251212_research_mouse_facelift_daily.md*
