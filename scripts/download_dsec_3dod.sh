#!/bin/bash

# ==============================================================================
# 脚本名称: download_dsec_3dod.sh
# 作用: 从 HuggingFace 下载 DSEC-3DOD 数据集并解压
# 目标路径: AWS EC2 数据盘独立目录
# ==============================================================================

# 设置严格模式，遇到错误立刻退出
set -e

# 定义下载的隔离路径
TARGET_DIR="/home/ubuntu/mmood3d/dsec_workspace/data/DSEC-3DOD"
HF_REPO="mickeykang/DSEC-3DOD"

echo "============================================================"
echo "开始配置 DSEC-3DOD 数据集下载..."
echo "目标隔离路径: $TARGET_DIR"
echo "============================================================"

# 创建目标隔离目录
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# 检查是否安装了 huggingface_hub（使用 python/uv）
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli 未在全局找到。我们将尝试通过 uv 环境或 pip 安装..."
    # 假设服务器已配置好 uv 环境
    if command -v uv &> /dev/null; then
        echo "发现 uv，正在通过 uv tool 安装 huggingface_hub..."
        uv tool install huggingface_hub
        export PATH="$HOME/.cargo/bin:$PATH" # 确保 uv tool 路径在环境变量中
    else
        echo "未发现 uv，正在使用 pip 安装 huggingface_hub..."
        pip install -U "huggingface_hub[cli]"
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

echo "============================================================"
echo "开始使用 huggingface-cli 下载数据集 (约 37GB)..."
echo "============================================================"

# 使用 huggingface-cli 下载，自动支持断点续传和多线程
huggingface-cli download $HF_REPO --repo-type dataset --local-dir "$TARGET_DIR"

echo "============================================================"
echo "下载完成。开始解压主要的 zip 文件 (若存在)..."
echo "这可能需要较长时间并消耗解压空间..."
echo "============================================================"

# 解压文件
if [ -f "DSEC.zip" ]; then
    # -u 选项：只更新已改变的文件，如果中断后再次执行可节省时间
    unzip -u DSEC.zip
    echo "解压成功！"
    # 提示用户是否删除原始 zip 包
    echo "提示: 解压后若确认文件无误，可以手动执行 'rm DSEC.zip' 释放 37GB 空间。"
else
    echo "未发现 DSEC.zip，请检查下载日志。"
fi

echo "============================================================"
echo "DSEC-3DOD 数据集获取阶段执行完毕！"
echo "请继续执行阶段一的数据切片转换。"
echo "============================================================"
