#!/bin/bash
#
# run_all_experiments.sh
#
# 전체 위치 변형 실험 실행
# 저 coverage 비정형 물체의 PP 영향 검증
#

set -e  # 에러 시 중단

# =============================================================================
# 설정
# =============================================================================

# Blender 경로 (환경에 맞게 수정)
BLENDER="${BLENDER:-$HOME/blender-4.0.2-linux-x64/blender}"

# 스크립트 경로
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CREATE_MODEL_SCRIPT="${SCRIPT_DIR}/create_mouse_model.py"
RENDER_SCRIPT="${SCRIPT_DIR}/render_position_experiments.py"
VALIDATE_SCRIPT="${SCRIPT_DIR}/validate_synthetic_dataset.py"

# 출력 경로
OUTPUT_BASE="${OUTPUT_BASE:-/home/joon/data/synthetic/position_experiments}"
MODEL_FILE="${OUTPUT_BASE}/mouse_model.blend"

# 실험 설정
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"

# 실험 목록
EXPERIMENTS=(
    "POS_C"       # 기준선
    "POS_R"       # 오른쪽 오프셋
    "POS_R_PP"    # 오른쪽 + PP 보정
    "POS_U"       # 위쪽 오프셋
    "POS_RU"      # 대각선
    "POS_RU_PP"   # 대각선 + PP 보정
    "POS_RANDOM"  # 랜덤 위치
    "POS_RANDOM_PP"  # 랜덤 + PP 보정
)

# =============================================================================
# 함수
# =============================================================================

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"
}

check_blender() {
    if ! command -v "$BLENDER" &> /dev/null; then
        echo "Error: Blender not found at: $BLENDER"
        echo "Set BLENDER environment variable or install Blender"
        exit 1
    fi
    log "Blender: $($BLENDER --version | head -1)"
}

# =============================================================================
# 메인
# =============================================================================

main() {
    log "========================================"
    log "Position Experiments Pipeline"
    log "========================================"
    log "Output: $OUTPUT_BASE"
    log "Samples: $NUM_SAMPLES"
    log "Experiments: ${#EXPERIMENTS[@]}"
    log "========================================"
    
    # Blender 확인
    check_blender
    
    # 출력 디렉토리 생성
    mkdir -p "$OUTPUT_BASE"
    
    # Step 1: 모델 생성
    log ""
    log "Step 1: Creating mouse model..."
    log "----------------------------------------"
    
    if [ ! -f "$MODEL_FILE" ]; then
        $BLENDER --background -P "$CREATE_MODEL_SCRIPT" -- \
            --output "$MODEL_FILE" \
            --size 0.25
        log "Model created: $MODEL_FILE"
    else
        log "Model exists: $MODEL_FILE (skipping)"
    fi
    
    # Step 2: 각 실험 렌더링
    log ""
    log "Step 2: Rendering experiments..."
    log "----------------------------------------"
    
    for EXP in "${EXPERIMENTS[@]}"; do
        EXP_DIR="${OUTPUT_BASE}/${EXP}"
        
        log ""
        log "Experiment: $EXP"
        log "Output: $EXP_DIR"
        
        if [ -d "$EXP_DIR" ] && [ -f "$EXP_DIR/data_list.txt" ]; then
            log "Already exists (skipping)"
            continue
        fi
        
        $BLENDER --background "$MODEL_FILE" -P "$RENDER_SCRIPT" -- \
            --output_dir "$EXP_DIR" \
            --experiment "$EXP" \
            --num_samples "$NUM_SAMPLES" \
            --seed "$SEED"
        
        log "Completed: $EXP"
    done
    
    # Step 3: 검증
    log ""
    log "Step 3: Validating datasets..."
    log "----------------------------------------"
    
    for EXP in "${EXPERIMENTS[@]}"; do
        EXP_DIR="${OUTPUT_BASE}/${EXP}"
        
        if [ -d "$EXP_DIR" ]; then
            log "Validating: $EXP"
            python "$VALIDATE_SCRIPT" "${EXP_DIR}/sample_00000" || true
        fi
    done
    
    # Step 4: 요약
    log ""
    log "========================================"
    log "Pipeline Complete"
    log "========================================"
    log "Output: $OUTPUT_BASE"
    log ""
    log "Generated datasets:"
    for EXP in "${EXPERIMENTS[@]}"; do
        EXP_DIR="${OUTPUT_BASE}/${EXP}"
        if [ -d "$EXP_DIR" ]; then
            COUNT=$(ls -d "$EXP_DIR"/sample_* 2>/dev/null | wc -l)
            log "  $EXP: $COUNT samples"
        fi
    done
    log ""
    log "Next steps:"
    log "  1. Review rendered images"
    log "  2. Create GS-LRM dataset configs"
    log "  3. Run finetuning experiments"
    log "========================================"
}

# 실행
main "$@"
