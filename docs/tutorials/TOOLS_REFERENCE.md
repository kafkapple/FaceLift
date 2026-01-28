# Tools Reference

> 진단 및 분석 도구 가이드

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

*Tools Reference | 2026-01-28*
