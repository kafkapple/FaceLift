

---

## MVG-Correct Presets (2026-01-25)

### 문제 발견

기존 M3_norm, M3_persample 프리셋의 **Object-Centered Zoom**이 MVG 부정합 야기:
- M3_norm: Ray Error 13.62° (cx=197±46, cy=137±22)
- M3_persample: Ray Error 16.15° (cx=235±32, cy=212±40)

### 해결책: Center-Aligned Zoom

이미지 중심 기준 crop으로 PP=256 자동 보장.

| 프리셋 | zoom_center_mode | PP 결과 | Ray Error |
|--------|------------------|---------|-----------|
| M3_1 | image | 256±0 | 0° |
| M3_2 | image | 256±0 | 0° |

### 사용법

```bash
# M3_1 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_1

# M3_2 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 검증

```bash
python mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py --verbose
```

### 관련 문서

- [[PP_MVG_COMPREHENSIVE_ANALYSIS]] - 종합 분석
- [[PP_FIX_MVG_THEORY]] - MVG 이론 분석
- [[EXPERIMENT_REGISTRY]] - 실험 레지스트리

---

## 버그 수정 이력

### 2026-01-25: normalize_after_zoom PP 버그 수정

**영향**: M3_1, M3_2 (center-aligned zoom 사용)

**문제**: `normalize_after_zoom` 활성화 시 center-aligned zoom에서도 PP가 스케일링되어 256에서 190으로 변경됨

**수정**: `preprocess.py:656-663` - zoom_center_mode 조건 추가

**재전처리 필요**: M3_1, M3_2 데이터셋 삭제 후 재생성

**상세**: [[theory/PP_FX_MVG_ANALYSIS#11-버그-발견-및-수정-2026-01-25]]
