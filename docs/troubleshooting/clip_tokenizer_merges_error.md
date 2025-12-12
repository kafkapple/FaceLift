# CLIPTokenizer merges.txt Missing Error

## 증상

```
Traceback (most recent call last):
  File "train_diffusion.py", line XXX, in <module>
    main(cfg)
  ...
  File ".../tokenization_clip.py", line 312, in __init__
    with open(merges_file, encoding="utf-8") as merges_handle:
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

## 원인

CLIPTokenizer는 BPE(Byte Pair Encoding) 토크나이저로, 다음 두 파일이 **필수**입니다:

| 파일 | 용도 |
|------|------|
| `vocab.json` | 토큰 → ID 매핑 사전 |
| `merges.txt` | BPE merge 규칙 (서브워드 병합 순서) |

일부 pretrained checkpoint 다운로드 시 `merges.txt`가 누락될 수 있습니다.

## 해결 방법

### 자동 해결 (v2024.12.10+)

`train_diffusion.py`에 자동 다운로드 로직이 포함되어 있습니다:

```python
# load_models() 호출 시 자동으로 확인 및 다운로드
ensure_tokenizer_files(cfg.pretrained_model_name_or_path)
```

### 수동 해결

```bash
# tokenizer 디렉토리로 이동
cd checkpoints/mvdiffusion/pipeckpts/tokenizer

# merges.txt 다운로드
wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt
```

## 검증

tokenizer 디렉토리에 다음 파일들이 있는지 확인:

```bash
ls -la checkpoints/mvdiffusion/pipeckpts/tokenizer/
# 필수 파일:
# - vocab.json (~1MB)
# - merges.txt (~500KB)
# - tokenizer_config.json
# - special_tokens_map.json
```

## 관련 코드

- `train_diffusion.py:ensure_tokenizer_files()` - 자동 다운로드 로직
- 소스: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

## 참고

이 문제는 CLIP 계열 모델(CLIP-ViT-L/14 등)을 사용하는 모든 프로젝트에서 발생할 수 있습니다.
BPE 토크나이저는 `vocab.json`만으로는 동작하지 않으며, merge 규칙이 있어야 입력 텍스트를 올바르게 토큰화할 수 있습니다.
