#!/bin/bash
# 전체 단계별 테스트 실행 스크립트
# Usage: bash run_all_stage_tests.sh

set -e

echo "============================================================"
echo "Mouse-FaceLift 단계별 테스트"
echo "============================================================"

# Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

cd ~/FaceLift

OUTPUT_BASE="outputs/stage_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_BASE

echo ""
echo "출력 디렉토리: $OUTPUT_BASE"
echo ""

# ============================================================
# Stage 1: MVDiffusion 테스트
# ============================================================
echo "============================================================"
echo "Stage 1: MVDiffusion 테스트"
echo "============================================================"

# Test 1a: Mouse 이미지 + Fine-tuned UNet
echo "[1a] Mouse 이미지 + Fine-tuned UNet"
python test_stage1_mvdiffusion.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --output_dir $OUTPUT_BASE/stage1 \
    --unet checkpoints/mvdiffusion/mouse/checkpoint-2000/unet

# Test 1b: Mouse 이미지 + Pretrained UNet (비교용)
echo "[1b] Mouse 이미지 + Pretrained UNet"
python test_stage1_mvdiffusion.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --output_dir $OUTPUT_BASE/stage1 \
    --unet "" \
    --test_pretrained

# Test 1c: Human 이미지 + Pretrained UNet (baseline)
echo "[1c] Human 이미지 + Pretrained UNet"
python test_stage1_mvdiffusion.py \
    --input_image data_sample/gslrm/sample_000/images/cam_000.png \
    --output_dir $OUTPUT_BASE/stage1 \
    --unet "" \
    --test_pretrained

echo ""
echo "Stage 1 완료: $OUTPUT_BASE/stage1"
echo ""

# ============================================================
# Stage 2: GS-LRM 테스트
# ============================================================
echo "============================================================"
echo "Stage 2: GS-LRM 테스트"
echo "============================================================"

# MVDiffusion 출력으로 GS-LRM 테스트
MVDIFF_OUTPUT="$OUTPUT_BASE/stage1/cam_000_finetuned"

if [ -d "$MVDIFF_OUTPUT" ]; then
    # Test 2a: MVDiffusion 출력 + GS-LRM Pretrained + FaceLift 카메라
    echo "[2a] MVDiffusion 출력 + GS-LRM Pretrained + FaceLift 카메라"
    python test_stage2_gslrm.py \
        --image_dir $MVDIFF_OUTPUT \
        --camera_json utils_folder/opencv_cameras.json \
        --output_dir $OUTPUT_BASE/stage2 \
        --ckpt_pretrained checkpoints/gslrm/ckpt_0000000000021125.pt \
        --test_both

    # Test 2b: MVDiffusion 출력 + GS-LRM Pretrained + Mouse 카메라 (비교)
    echo "[2b] MVDiffusion 출력 + GS-LRM Pretrained + Mouse 카메라"
    python test_stage2_gslrm.py \
        --image_dir $MVDIFF_OUTPUT \
        --camera_json data_mouse/sample_000000/opencv_cameras.json \
        --output_dir $OUTPUT_BASE/stage2_mouse_cam \
        --ckpt_pretrained checkpoints/gslrm/ckpt_0000000000021125.pt
fi

# Human 데이터로 baseline 테스트
echo "[2c] Human 6-view 직접 입력 + GS-LRM Pretrained (baseline)"
python test_stage2_gslrm.py \
    --image_dir data_sample/gslrm/sample_000/images \
    --camera_json utils_folder/opencv_cameras.json \
    --output_dir $OUTPUT_BASE/stage2_human_baseline \
    --ckpt_pretrained checkpoints/gslrm/ckpt_0000000000021125.pt

echo ""
echo "Stage 2 완료: $OUTPUT_BASE/stage2"
echo ""

# ============================================================
# 결과 요약
# ============================================================
echo "============================================================"
echo "테스트 완료! 결과 요약:"
echo "============================================================"
echo ""
echo "Stage 1 (MVDiffusion):"
ls -la $OUTPUT_BASE/stage1/ 2>/dev/null || echo "  (결과 없음)"
echo ""
echo "Stage 2 (GS-LRM):"
ls -la $OUTPUT_BASE/stage2/ 2>/dev/null || echo "  (결과 없음)"
echo ""
echo "전체 출력 디렉토리: $OUTPUT_BASE"
echo ""
