
---

## Frame Context Inspection Tool

프레임 주변 컨텍스트를 슬로우모션으로 추출하여 검사하는 도구.

### 위치
`mouse_extensions/scripts/extract_frame_context.py`

### 사용법
```bash
# 기본: 3초 전후, 0.25x 속도
python extract_frame_context.py --video input.mp4 --frames 5900

# 슬로우모션 (10x 느리게)
python extract_frame_context.py --video input.mp4 --frames 5900,11800 --speed 0.1

# 윈도우 확장 (5초 전후)
python extract_frame_context.py --video input.mp4 --frames 5900 --window 5.0
```

### 파라미터
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video, -v` | 입력 비디오 | (필수) |
| `--frames, -f` | 타겟 프레임 (콤마 구분) | (필수) |
| `--window, -w` | 전후 윈도우 (초) | 3.0 |
| `--speed, -s` | 재생 속도 배율 | 0.25 |
| `--output, -o` | 출력 디렉토리 | 비디오 위치 |

---

## Raw Data: markerless_mouse_1_nerf

### 프레임 정보
- **총 프레임**: 18,000 (6개 카메라 동일)
- **FPS**: 100
- **해상도**: 1152×1024
- **frame_interval=5** → 3,600 샘플

### ⚠️ Frame Discontinuity (불연속 위치)
```
DISCONTINUITY_FRAMES = {5900, 11800, 17700}
```

**의미**: 해당 위치에서 원본 녹화의 프레임 불연속(갭) 발생
- 해당 프레임 자체는 **정상** (제외 불필요)
- temporal smoothness 가정하는 알고리즘 사용 시 주의
- 현재 전처리: **모든 프레임 사용** (제외 없음)
