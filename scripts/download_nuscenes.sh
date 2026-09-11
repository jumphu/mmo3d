#!/bin/bash
# ============================================================
# nuScenes LiDAR 数据下载脚本（通过 Cognito + S3 直接下载）
#
# 使用方式:
#   1. 填入你的 nuScenes 账号（下方 EMAIL 和 PASSWORD）
#   2. 上传到 poc-jump-001: scp scripts/download_nuscenes.sh poc-jump-001:/data/mmood3d/
#   3. 后台运行: ssh poc-jump-001 "nohup bash /data/mmood3d/download_nuscenes.sh &"
#   4. 查看进度: ssh poc-jump-001 "tail -f /data/mmood3d/download.log"
# ============================================================

# !! 填入你的 nuScenes 账号 !!
EMAIL="YOUR_EMAIL"
PASSWORD="YOUR_PASSWORD"

# Cognito 配置（从 nuScenes 前端 JS 提取，已验证）
USER_POOL_ID="us-east-1_G55ePzusp"
CLIENT_ID="7fq5jvs5ffs1c50hd3toobb3b9"
IDENTITY_POOL_ID="us-east-1:8693bbd9-67a2-43fa-bb9b-2c699507b79f"
S3_BUCKET="data.nuscenes.org"
COGNITO_REGION="us-east-1"

DATA_DIR="/data/mmood3d/code/data/nuscenes"
CKPT_DIR="/data/mmood3d/code/checkpoints"
TEMP_DIR="/data/mmood3d/tmp"
LOG_FILE="/data/mmood3d/download.log"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$TEMP_DIR"
> "$LOG_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "========== 开始下载 nuScenes 数据 =========="
log "数据目录: $DATA_DIR"
df -h /data | tee -a "$LOG_FILE"

# ----------------------------------------------------------
# 1. 下载基础检测器权重
# ----------------------------------------------------------
log "[Step 1/4] 下载 CenterPoint 基础检测器权重..."
CKPT_FILE="$CKPT_DIR/base_centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d_known.pth"
if [ -f "$CKPT_FILE" ]; then
    log "  权重文件已存在 ($(ls -lh "$CKPT_FILE" | awk '{print $5}')), 跳过"
else
    wget -c -q --show-progress \
        "https://github.com/uulm-mrm/mmood3d/releases/download/base_centerpoint/base_centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d_known.pth" \
        -O "$CKPT_FILE" 2>&1 | tee -a "$LOG_FILE"
    log "  权重下载完成: $(ls -lh "$CKPT_FILE" | awk '{print $5}')"
fi

# ----------------------------------------------------------
# 2. Cognito 认证获取临时 AWS 凭证
# ----------------------------------------------------------
log "[Step 2/4] Cognito 认证..."

# Step 2a: 用户名密码认证，获取 ID Token
AUTH_RESULT=$(aws cognito-idp initiate-auth \
    --region "$COGNITO_REGION" \
    --auth-flow USER_PASSWORD_AUTH \
    --client-id "$CLIENT_ID" \
    --auth-parameters USERNAME="$EMAIL",PASSWORD="$PASSWORD" \
    --no-sign-request \
    --output json 2>&1)

ID_TOKEN=$(echo "$AUTH_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['AuthenticationResult']['IdToken'])" 2>/dev/null)

if [ -z "$ID_TOKEN" ]; then
    log "  ✗ 认证失败！请检查邮箱和密码"
    log "  错误信息: $AUTH_RESULT"
    exit 1
fi
log "  ✓ 获取 ID Token 成功"

# Step 2b: 用 ID Token 换取 Identity ID
PROVIDER_KEY="cognito-idp.${COGNITO_REGION}.amazonaws.com/${USER_POOL_ID}"

IDENTITY_ID=$(aws cognito-identity get-id \
    --region "$COGNITO_REGION" \
    --identity-pool-id "$IDENTITY_POOL_ID" \
    --logins "{\"$PROVIDER_KEY\":\"$ID_TOKEN\"}" \
    --no-sign-request \
    --query "IdentityId" --output text 2>&1)

if [ -z "$IDENTITY_ID" ] || [[ "$IDENTITY_ID" == *"error"* ]]; then
    log "  ✗ 获取 Identity ID 失败: $IDENTITY_ID"
    exit 1
fi
log "  ✓ Identity ID: $IDENTITY_ID"

# Step 2c: 获取临时 AWS 凭证
CREDENTIALS=$(aws cognito-identity get-credentials-for-identity \
    --region "$COGNITO_REGION" \
    --identity-id "$IDENTITY_ID" \
    --logins "{\"$PROVIDER_KEY\":\"$ID_TOKEN\"}" \
    --no-sign-request \
    --output json 2>&1)

export AWS_ACCESS_KEY_ID=$(echo "$CREDENTIALS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['AccessKeyId'])" 2>/dev/null)
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDENTIALS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SecretKey'])" 2>/dev/null)
export AWS_SESSION_TOKEN=$(echo "$CREDENTIALS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SessionToken'])" 2>/dev/null)

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    log "  ✗ 获取 AWS 临时凭证失败"
    log "  $CREDENTIALS"
    exit 1
fi
log "  ✓ 获取 AWS 临时凭证成功"

# ----------------------------------------------------------
# 3. 从 S3 下载 nuScenes 数据
# ----------------------------------------------------------
log "[Step 3/4] 从 S3 下载 nuScenes trainval 数据..."

# 需要下载的文件（仅 LiDAR）
FILES=(
    "public/v1.0/v1.0-trainval_meta.tgz"
    "public/v1.0/v1.0-trainval01_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval02_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval03_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval04_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval05_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval06_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval07_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval08_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval09_blobs_lidar.tgz"
    "public/v1.0/v1.0-trainval10_blobs_lidar.tgz"
)

TOTAL=${#FILES[@]}
CURRENT=0

for S3_KEY in "${FILES[@]}"; do
    FILENAME=$(basename "$S3_KEY")
    CURRENT=$((CURRENT + 1))
    FILEPATH="$TEMP_DIR/$FILENAME"

    log "  [$CURRENT/$TOTAL] 下载 $FILENAME ..."

    # 如果已解压过关键目录，跳过
    if [ "$FILENAME" = "v1.0-trainval_meta.tgz" ] && [ -d "$DATA_DIR/v1.0-trainval" ]; then
        log "  [$CURRENT/$TOTAL] $FILENAME 已处理，跳过"
        continue
    fi

    # 从 S3 下载（带进度）
    aws s3 cp "s3://$S3_BUCKET/$S3_KEY" "$FILEPATH" \
        --region "$COGNITO_REGION" \
        2>&1 | tee -a "$LOG_FILE"

    if [ ! -f "$FILEPATH" ]; then
        log "  [$CURRENT/$TOTAL] ✗ 下载 $FILENAME 失败!"
        continue
    fi

    FILE_SIZE=$(ls -lh "$FILEPATH" | awk '{print $5}')
    log "  [$CURRENT/$TOTAL] 下载完成: $FILE_SIZE, 开始解压..."

    # 解压到数据目录
    tar xzf "$FILEPATH" -C "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"

    # 删除压缩包释放空间
    rm -f "$FILEPATH"

    DISK_AVAIL=$(df -h /data | tail -1 | awk '{print $4}')
    log "  [$CURRENT/$TOTAL] ✓ $FILENAME 完成. 磁盘剩余: $DISK_AVAIL"
done

rmdir "$TEMP_DIR" 2>/dev/null || true

# ----------------------------------------------------------
# 4. 验证数据完整性
# ----------------------------------------------------------
log "[Step 4/4] 验证数据目录结构..."

check_dir() {
    if [ -d "$1" ]; then
        COUNT=$(find "$1" -type f | wc -l)
        SIZE=$(du -sh "$1" | awk '{print $1}')
        log "  ✓ $2: $COUNT 个文件, $SIZE"
    else
        log "  ✗ $2 不存在!"
    fi
}

check_dir "$DATA_DIR/v1.0-trainval" "v1.0-trainval (元数据)"
check_dir "$DATA_DIR/samples/LIDAR_TOP" "samples/LIDAR_TOP (keyframes)"
check_dir "$DATA_DIR/sweeps/LIDAR_TOP" "sweeps/LIDAR_TOP (sweeps)"

log ""
log "========== 全部完成 =========="
du -sh "$DATA_DIR" 2>/dev/null | tee -a "$LOG_FILE"
df -h /data | tee -a "$LOG_FILE"
log ""
log "下一步: 生成 annotation 文件"
log "  cd /data/mmood3d/code"
log "  source /data/mmood3d/.venv/bin/activate"
log "  python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes"
