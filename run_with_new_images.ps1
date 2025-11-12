# PowerShell 一键运行脚本 - 新照片路径配置
# 使用方法：修改下面的路径，然后运行: .\run_with_new_images.ps1

# ========== 请在这里填写你的照片路径 ==========
$PHOTOS_PATH = "C:\path\to\your\photos"  # 你的照片文件夹路径（或视频文件路径）
$OUTPUT_DIR = ".\output_scene"           # 输出目录（会自动创建）
$INPUT_TYPE = "images"                   # "images" 或 "video"
# =============================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "一键运行 - 新照片训练" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 检查路径是否存在
if (-not (Test-Path $PHOTOS_PATH)) {
    Write-Host "错误: 照片路径不存在: $PHOTOS_PATH" -ForegroundColor Red
    Write-Host "请修改脚本中的 PHOTOS_PATH 变量" -ForegroundColor Yellow
    exit 1
}

Write-Host "照片路径: $PHOTOS_PATH" -ForegroundColor Green
Write-Host "输出目录: $OUTPUT_DIR" -ForegroundColor Green
Write-Host "输入类型: $INPUT_TYPE" -ForegroundColor Green
Write-Host ""

# 设置环境变量（bash脚本会读取这些）
$env:DATA_RAW = $PHOTOS_PATH
$env:OUTPUT_DIR = $OUTPUT_DIR
$env:INPUT_TYPE = $INPUT_TYPE

Write-Host "正在运行一键脚本..." -ForegroundColor Yellow
Write-Host ""

# 检查是否有WSL或Git Bash
if (Get-Command bash -ErrorAction SilentlyContinue) {
    # 使用bash运行脚本
    bash scripts/run_baseline_pipeline.sh
} else {
    Write-Host "未找到bash，请使用以下方式之一：" -ForegroundColor Yellow
    Write-Host "1. 安装Git Bash或WSL" -ForegroundColor Yellow
    Write-Host "2. 或手动运行以下命令：" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   # 在Git Bash或WSL中运行：" -ForegroundColor Cyan
    Write-Host "   export DATA_RAW='$PHOTOS_PATH'" -ForegroundColor White
    Write-Host "   export OUTPUT_DIR='$OUTPUT_DIR'" -ForegroundColor White
    Write-Host "   export INPUT_TYPE='$INPUT_TYPE'" -ForegroundColor White
    Write-Host "   bash scripts/run_baseline_pipeline.sh" -ForegroundColor White
    Write-Host ""
}

