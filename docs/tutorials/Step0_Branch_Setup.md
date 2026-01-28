> **DEPRECATED (2026-01-29)**: This tutorial predates mouse_extensions refactoring. Does not reflect Modular config (-d -e), preset system, or mouse_extensions/ structure.

# Step 0: 원본 Repo 기반 새 브랜치 생성

> 현재 저장소에서 원본 FaceLift main을 받아 새 브랜치로 만들고, 단계별로 mouse 모듈을 추가하는 방법

## 0.1 현재 상황 이해

```
현재 저장소 상태:
├── main      ← 원본 FaceLift (upstream)
├── dev       ← mouse 수정본 (현재 위치)
└── (새 브랜치) ← 원본 기반으로 처음부터 구현 테스트
```

---

## 0.2 새 브랜치 생성 (원본 main 기반)

### Step A: 원본 main 브랜치 확인

```bash
cd /home/joon/dev/FaceLift

# main 브랜치로 이동
git checkout main

# main이 원본과 같은지 확인
git log --oneline -5
# weijielyu/FaceLift 원본 커밋들이 보여야 함
```

### Step B: 새 구현 브랜치 생성

```bash
# main 기반으로 새 브랜치 생성
git checkout -b implement/mouse-from-scratch

# 현재 위치 확인
git branch
# * implement/mouse-from-scratch   ← 여기!
#   dev
#   main
```

이제 이 브랜치는 **원본 FaceLift 상태**입니다.

---

## 0.3 단계별 구현 테스트

### 구현 순서

```
Step 1: MouseViewDataset 추가
   ↓
Step 2: train_gslrm.py 수정 (use_mouse_dataset 플래그)
   ↓
Step 3: configs/mouse/ 생성
   ↓
Step 4: 전처리 스크립트 추가 (선택)
   ↓
Step 5: 테스트 실행
```

---

## Step 1: MouseViewDataset 추가

### 1-1. 파일 생성

```bash
# gslrm/data/mouse_dataset.py 생성
touch gslrm/data/mouse_dataset.py
```

### 1-2. dev 브랜치에서 코드 복사

```bash
# dev 브랜치의 mouse_dataset.py 내용 확인
git show dev:gslrm/data/mouse_dataset.py > gslrm/data/mouse_dataset.py
```

또는 수동으로 `docs/tutorials/Step2_Mouse_Dataset.md`의 코드를 복사.

### 1-3. 커밋

```bash
git add gslrm/data/mouse_dataset.py
git commit -m "feat: Add MouseViewDataset for 6-view mouse data"
```

---

## Step 2: train_gslrm.py 수정

### 2-1. load_datasets() 메서드 수정

`train_gslrm.py`의 `load_datasets()` 메서드를 찾아서 수정:

```python
def load_datasets(self):
    """Load training and validation datasets."""
    
    # NEW: Check if mouse dataset should be used
    use_mouse_dataset = self.config.get("mouse", {}).get("use_mouse_dataset", False)
    
    if use_mouse_dataset:
        from gslrm.data.mouse_dataset import MouseViewDataset
        print("Using MouseViewDataset with camera normalization")
        self.dataset = MouseViewDataset(self.config, split="train")
        if self.config.validation.enabled:
            self.val_dataset = MouseViewDataset(self.config, split="val")
        else:
            self.val_dataset = None
    else:
        # Original: use RandomViewDataset
        from gslrm.data.dataset import RandomViewDataset
        self.dataset = RandomViewDataset(self.config, split="train")
        if self.config.validation.enabled:
            self.val_dataset = RandomViewDataset(self.config, split="val")
        else:
            self.val_dataset = None
    
    self._log_dataset_examples()
    self._setup_dataloaders()
```

### 2-2. 커밋

```bash
git add train_gslrm.py
git commit -m "feat: Add use_mouse_dataset flag to train_gslrm.py"
```

---

## Step 3: Mouse Config 생성

### 3-1. 디렉토리 생성

```bash
mkdir -p configs/mouse
```

### 3-2. Config 파일 복사

```bash
# dev 브랜치에서 복사
git show dev:configs/mouse/gslrm_recommended.yaml > configs/mouse/gslrm_recommended.yaml
```

### 3-3. 데이터 경로 수정

`configs/mouse/gslrm_recommended.yaml` 열고 경로 수정:

```yaml
training:
  dataset:
    dataset_path: /your/actual/path/data_mouse_train.txt  # 실제 경로!

validation:
  dataset_path: /your/actual/path/data_mouse_val.txt  # 실제 경로!
```

### 3-4. 커밋

```bash
git add configs/mouse/
git commit -m "feat: Add mouse config files"
```

---

## Step 4: 테스트 실행

### 4-1. Dataset 로드 테스트

```bash
python -c "
from gslrm.data.mouse_dataset import MouseViewDataset
print('MouseViewDataset imported successfully!')
"
```

### 4-2. 학습 테스트 (Dry-run)

```bash
# Config 파싱 테스트
python -c "
import yaml
from easydict import EasyDict as edict

with open('configs/mouse/gslrm_recommended.yaml') as f:
    config = edict(yaml.safe_load(f))

print('Config loaded!')
print(f'  use_mouse_dataset: {config.mouse.use_mouse_dataset}')
print(f'  num_views: {config.training.dataset.num_views}')
"
```

### 4-3. 실제 학습 테스트 (데이터 준비 후)

```bash
python train_gslrm.py --config configs/mouse/gslrm_recommended.yaml
```

---

## 0.4 dev 브랜치와 비교

각 단계 완료 후 dev 브랜치와 비교:

```bash
# 현재 브랜치와 dev 비교
git diff dev -- gslrm/data/mouse_dataset.py
git diff dev -- train_gslrm.py
git diff dev -- configs/mouse/
```

---

## 0.5 문제 해결

### "ModuleNotFoundError: No module named 'gslrm'"

```bash
# PYTHONPATH 설정
export PYTHONPATH=/home/joon/dev/FaceLift:$PYTHONPATH
```

### "FileNotFoundError: data_mouse_train.txt"

- `configs/mouse/gslrm_recommended.yaml`의 `dataset_path`를 실제 경로로 수정

### 원본으로 되돌리기

```bash
# 모든 변경 취소하고 main으로 돌아가기
git checkout main
git branch -D implement/mouse-from-scratch  # 브랜치 삭제
```

---

## 요약: 실행 명령어 모음

```bash
# 1. 새 브랜치 생성
git checkout main
git checkout -b implement/mouse-from-scratch

# 2. MouseViewDataset 추가
git show dev:gslrm/data/mouse_dataset.py > gslrm/data/mouse_dataset.py
git add gslrm/data/mouse_dataset.py
git commit -m "feat: Add MouseViewDataset"

# 3. train_gslrm.py 수정 (수동 편집 후)
git add train_gslrm.py
git commit -m "feat: Add use_mouse_dataset flag"

# 4. Config 추가
mkdir -p configs/mouse
git show dev:configs/mouse/gslrm_recommended.yaml > configs/mouse/gslrm_recommended.yaml
# 경로 수정 후
git add configs/mouse/
git commit -m "feat: Add mouse config"

# 5. 테스트
python train_gslrm.py --config configs/mouse/gslrm_recommended.yaml
```

---

*Created: 2026-01-13*
