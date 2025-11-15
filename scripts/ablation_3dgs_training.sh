#!/usr/bin/env bash
# 3DGS训练优化的消融实验
# 测试不同配置：baseline, 只用edge loss, 只用SSIM调整, 两者都用

set -euo pipefail

# Configuration
SCENE_ROOT="${SCENE_ROOT:-/home/bygpu/data/video_scene}"
ITERATIONS="${ITERATIONS:-30000}"
TEST_LAST="${TEST_LAST:-40}"  # 测试集数量

# Baseline参数（默认）
LAMBDA_DSSIM_BASELINE="${LAMBDA_DSSIM_BASELINE:-0.2}"  # 默认SSIM权重
LAMBDA_EDGE_BASELINE="${LAMBDA_EDGE_BASELINE:-0.0}"   # 无edge loss

# Optimized参数
LAMBDA_DSSIM_OPT="${LAMBDA_DSSIM_OPT:-0.25}"          # 调整后的SSIM权重
LAMBDA_EDGE_OPT="${LAMBDA_EDGE_OPT:-0.1}"            # Edge loss权重

BASE_OUTPUT_DIR="./output/ablation_3dgs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BASE_OUTPUT_DIR}"

echo "==============================================="
echo "3DGS Training Optimization Ablation Study"
echo "==============================================="
echo "Scene root: ${SCENE_ROOT}"
echo "Iterations: ${ITERATIONS}"
echo "Test set: Last ${TEST_LAST} frames"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo ""

# 实验配置
declare -a EXPERIMENTS=(
    "baseline:${LAMBDA_DSSIM_BASELINE}:${LAMBDA_EDGE_BASELINE}:Baseline (no edge loss, default SSIM)"
    "edge_only:${LAMBDA_DSSIM_BASELINE}:${LAMBDA_EDGE_OPT}:Edge Loss Only (λ_edge=${LAMBDA_EDGE_OPT})"
    "ssim_only:${LAMBDA_DSSIM_OPT}:${LAMBDA_EDGE_BASELINE}:SSIM Adjustment Only (λ_dssim=${LAMBDA_DSSIM_OPT})"
    "both:${LAMBDA_DSSIM_OPT}:${LAMBDA_EDGE_OPT}:Both (λ_dssim=${LAMBDA_DSSIM_OPT}, λ_edge=${LAMBDA_EDGE_OPT})"
)

RESULTS_FILE="${BASE_OUTPUT_DIR}/ablation_results.txt"
echo "Experiment Results" > "${RESULTS_FILE}"
echo "==================" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name lambda_dssim lambda_edge exp_desc <<< "${exp_config}"
    
    MODEL_PATH="${BASE_OUTPUT_DIR}/${exp_name}"
    
    echo "----------------------------------------"
    echo "Experiment: ${exp_name}"
    echo "Description: ${exp_desc}"
    echo "λ_dssim: ${lambda_dssim}, λ_edge: ${lambda_edge}"
    echo "----------------------------------------"
    
    # 训练模型
    echo "[1/3] Training model..."
    if [[ "${lambda_edge}" == "0.0" ]]; then
        # 如果edge loss为0，使用baseline训练脚本
        python train_scene.py \
          -s "${SCENE_ROOT}" \
          --model_path "${MODEL_PATH}" \
          --iterations "${ITERATIONS}" \
          --eval \
          --test_last_n "${TEST_LAST}" \
          --lambda_dssim "${lambda_dssim}" \
          --save_iterations "${ITERATIONS}"
    else
        # 使用optimized训练脚本（支持edge loss）
        python train_scene_optimized.py \
          -s "${SCENE_ROOT}" \
          --model_path "${MODEL_PATH}" \
          --iterations "${ITERATIONS}" \
          --eval \
          --test_last_n "${TEST_LAST}" \
          --lambda_dssim "${lambda_dssim}" \
          --lambda_edge "${lambda_edge}" \
          --save_iterations "${ITERATIONS}"
    fi
    
    # 渲染测试集
    echo "[2/3] Rendering test set..."
    python render.py \
      -m "${MODEL_PATH}" \
      -s "${SCENE_ROOT}" \
      --iteration "${ITERATIONS}" \
      --skip_train \
      --eval \
      --test_last_n "${TEST_LAST}"
    
    # 计算metrics
    echo "[3/3] Computing metrics..."
    if [[ -f "scripts/compute_metrics.py" ]]; then
        python scripts/compute_metrics.py \
          --model_path "${MODEL_PATH}" \
          --set test \
          --iteration "${ITERATIONS}" || echo "Metrics computation failed, continuing..."
    else
        echo "Warning: compute_metrics.py not found, skipping metrics computation"
    fi
    
    # 尝试从输出中提取PSNR（如果metrics文件存在）
    METRICS_FILE="${MODEL_PATH}/test/ours_${ITERATIONS}/metrics.txt"
    if [[ -f "${METRICS_FILE}" ]]; then
        PSNR=$(grep -i "psnr" "${METRICS_FILE}" | head -1 | awk '{print $NF}' || echo "N/A")
        SSIM=$(grep -i "ssim" "${METRICS_FILE}" | head -1 | awk '{print $NF}' || echo "N/A")
        LPIPS=$(grep -i "lpips" "${METRICS_FILE}" | head -1 | awk '{print $NF}' || echo "N/A")
    else
        PSNR="N/A"
        SSIM="N/A"
        LPIPS="N/A"
    fi
    
    # 记录结果
    echo "${exp_name}|${exp_desc}|${lambda_dssim}|${lambda_edge}|${PSNR}|${SSIM}|${LPIPS}" >> "${RESULTS_FILE}"
    
    echo "✓ Experiment ${exp_name} complete"
    echo "  PSNR: ${PSNR}, SSIM: ${SSIM}, LPIPS: ${LPIPS}"
    echo ""
done

echo "==============================================="
echo "Ablation Study Complete!"
echo "==============================================="
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Summary:"
echo "--------"
cat "${RESULTS_FILE}"
echo ""
echo "Model paths:"
for exp_config in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name _ _ _ <<< "${exp_config}"
    echo "  ${exp_name}: ${BASE_OUTPUT_DIR}/${exp_name}"
done

