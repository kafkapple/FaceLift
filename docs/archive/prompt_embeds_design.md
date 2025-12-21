# MVDiffusion Prompt Embeddings 설계 가이드

## 개요

MVDiffusion은 **prompt embeddings**를 사용해서 각 출력 뷰의 방향을 제어합니다.
이 문서는 prompt embeddings의 역할, 구조, 생성 방법을 설명합니다.

---

## 1. Prompt Embeddings 역할

```
┌─────────────────────────────────────────────────────────────────────┐
│  MVDiffusion 추론 과정                                               │
│                                                                     │
│  입력: [같은 이미지 × 6] + [다른 prompt_embeds × 6]                   │
│         └─ reference view    └─ 각 뷰 방향 정보                      │
│                                                                     │
│  UNet 내부:                                                          │
│  - Cross-attention에서 prompt_embeds 사용                            │
│  - 각 위치(0~5)마다 다른 prompt_embeds → 다른 뷰 생성                 │
│                                                                     │
│  출력: [다른 뷰 × 6]                                                  │
│         └─ 각각 다른 카메라 방향에서 본 이미지                        │
└─────────────────────────────────────────────────────────────────────┘
```

**핵심**: prompt_embeds가 "이 위치에서는 어떤 방향의 뷰를 생성해라"라는 신호를 제공

---

## 2. Prompt Embeddings 구조

### 파일 구조
```
mvdiffusion/data/fixed_prompt_embeds_6view/  # FaceLift 기본
├── clr_embeds.pt      # Color map용 embeddings
└── normal_embeds.pt   # Normal map용 embeddings

mvdiffusion/data/mouse_prompt_embeds_6view/  # Mouse 카메라용
├── clr_embeds.pt
├── normal_embeds.pt
└── metadata.json      # 설정 정보
```

### Tensor Shape
```python
prompt_embeds.shape = [6, 77, 1024]
#                      │   │    │
#                      │   │    └─ CLIP embedding dimension
#                      │   └─ Token sequence length (CLIP max)
#                      └─ Number of views
```

### Dtype
- `torch.float16` (half precision)

---

## 3. 텍스트 프롬프트 설계

### FaceLift 기본 (수평 6방향, elevation 0°)
```python
views = ["front", "front_right", "right", "back", "left", "front_left"]

color_prompts = [
    "a rendering image of 3D models, front view, color map.",
    "a rendering image of 3D models, front_right view, color map.",
    "a rendering image of 3D models, right view, color map.",
    "a rendering image of 3D models, back view, color map.",
    "a rendering image of 3D models, left view, color map.",
    "a rendering image of 3D models, front_left view, color map.",
]
```

### Mouse 카메라용 (경사 6방향, elevation ~20°)
```python
views = ["top-front", "top-front-right", "top-right",
         "top-back", "top-left", "top-front-left"]

color_prompts = [
    "a rendering image of a 3D model, top-front view, from above at an angle, color map.",
    "a rendering image of a 3D model, top-front-right view, from above at an angle, color map.",
    # ...
]
```

### 프롬프트 설계 원칙
1. **방향 명시**: front, back, left, right 등 방향 키워드 포함
2. **elevation 표현**: top-, from above 등으로 경사 표현
3. **일관성**: 모든 뷰에 동일한 구조의 문장 사용
4. **짧고 명확**: CLIP이 잘 이해할 수 있는 간결한 문장

---

## 4. Prompt Embeddings 생성 방법

### 생성 스크립트
```bash
# FaceLift 기본 (수평)
python mvdiffusion/data/generate_fixed_text_embeds.py

# Mouse 카메라용 (경사)
python scripts/generate_mouse_prompt_embeds.py \
    --output_dir mvdiffusion/data/mouse_prompt_embeds_6view \
    --elevation 20
```

### 생성 과정
```python
# 1. CLIP 모델 로드
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

# 2. 텍스트 토큰화
text_inputs = tokenizer(prompts, padding="max_length", max_length=77, ...)

# 3. CLIP 인코딩
with torch.no_grad():
    prompt_embeds = text_encoder(text_inputs.input_ids)
    prompt_embeds = prompt_embeds[0]  # [6, 77, 1024]

# 4. 저장
torch.save(prompt_embeds.half(), "clr_embeds.pt")
```

---

## 5. Config에서 설정

### mouse_dataset.py에서 로딩
```python
# config에서 경로 지정 가능
prompt_embed_path = config.get(
    "prompt_embed_path",
    "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"  # 기본값
)
self.color_prompt_embedding = torch.load(prompt_embed_path)
```

### YAML config 예시
```yaml
# FaceLift 카메라용
prompt_embed_path: 'mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt'

# Mouse 카메라용
prompt_embed_path: 'mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt'
```

---

## 6. 실험별 Prompt Embeddings

| 실험 | Prompt Embeds | 카메라 설정 | 설명 |
|------|---------------|-------------|------|
| FaceLift 기본 | fixed_prompt_embeds_6view | 수평 6방향 | Human face용 |
| Option A | fixed_prompt_embeds_6view | 수평 6방향 | Mouse → FaceLift 카메라 변환 |
| Option B | mouse_prompt_embeds_6view | 경사 6방향 | Mouse 원본 카메라 |

---

## 7. 주의사항

1. **학습-추론 일치**: 학습 시 사용한 prompt_embeds와 추론 시 사용하는 것이 같아야 함
2. **카메라 매칭**: prompt_embeds의 뷰 방향과 실제 데이터 카메라가 일치해야 함
3. **GS-LRM 연계**: MVDiffusion 출력 뷰의 카메라 ≈ GS-LRM에 전달하는 카메라
