#!/usr/bin/env bash
# Helper script to run COLMAP mapper with DENSE solver when SuiteSparse is unavailable
# Usage: ./scripts/fix_colmap_mapper.sh <database_path> <image_path> <output_path>

set -euo pipefail

DB_PATH="${1:-}"
IMG_PATH="${2:-}"
OUT_PATH="${3:-}"

if [[ -z "${DB_PATH}" ]] || [[ -z "${IMG_PATH}" ]] || [[ -z "${OUT_PATH}" ]]; then
  echo "Usage: $0 <database_path> <image_path> <output_path>"
  exit 1
fi

# Create a temporary COLMAP config file to force DENSE solver
CONFIG_FILE=$(mktemp)
cat > "${CONFIG_FILE}" <<EOF
# COLMAP configuration to use DENSE solver instead of SuiteSparse
Mapper.ba_global_sparse_linear_algebra_library_type=DENSE_SCHUR
Mapper.ba_global_function_tolerance=1e-6
EOF

echo "Running COLMAP mapper with DENSE_SCHUR solver..."
echo "Config: ${CONFIG_FILE}"

# Try with config file first
colmap mapper \
  --project_path "${CONFIG_FILE}" \
  --database_path "${DB_PATH}" \
  --image_path "${IMG_PATH}" \
  --output_path "${OUT_PATH}" || {
  echo "Failed with config file, trying direct parameters..."
  # Fallback: try setting environment variable
  export CERES_USE_DENSE_SCHUR=1
  colmap mapper \
    --database_path "${DB_PATH}" \
    --image_path "${IMG_PATH}" \
    --output_path "${OUT_PATH}" \
    --Mapper.ba_global_function_tolerance=1e-6
}

rm -f "${CONFIG_FILE}"
echo "Done."


