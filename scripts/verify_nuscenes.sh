#!/bin/bash
# 验证 nuScenes S3 文件可访问性

REGION="us-east-1"
USER_POOL_ID="us-east-1_G55ePzusp"
CLIENT_ID="7fq5jvs5ffs1c50hd3toobb3b9"
IDENTITY_POOL_ID="us-east-1:8693bbd9-67a2-43fa-bb9b-2c699507b79f"
PROVIDER_KEY="cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}"

# 从下载脚本读取账号
eval $(grep '^EMAIL=' /data/mmood3d/download_nuscenes.sh)
eval $(grep '^PASSWORD=' /data/mmood3d/download_nuscenes.sh)

echo "=== Step 1: Cognito Auth ==="
AUTH_RESULT=$(aws cognito-idp initiate-auth \
    --region "$REGION" \
    --auth-flow USER_PASSWORD_AUTH \
    --client-id "$CLIENT_ID" \
    --auth-parameters USERNAME="$EMAIL",PASSWORD="$PASSWORD" \
    --no-sign-request \
    --output json 2>&1)

ID_TOKEN=$(echo "$AUTH_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['AuthenticationResult']['IdToken'])" 2>/dev/null)
if [ -z "$ID_TOKEN" ]; then
    echo "✗ Auth failed: $AUTH_RESULT"
    exit 1
fi
echo "✓ Auth OK"

echo "=== Step 2: Get Identity ==="
IDENTITY_ID=$(aws cognito-identity get-id \
    --region "$REGION" \
    --identity-pool-id "$IDENTITY_POOL_ID" \
    --logins "{\"$PROVIDER_KEY\":\"$ID_TOKEN\"}" \
    --no-sign-request \
    --query "IdentityId" --output text 2>&1)
echo "✓ Identity: $IDENTITY_ID"

echo "=== Step 3: Get Credentials ==="
CREDS=$(aws cognito-identity get-credentials-for-identity \
    --region "$REGION" \
    --identity-id "$IDENTITY_ID" \
    --logins "{\"$PROVIDER_KEY\":\"$ID_TOKEN\"}" \
    --no-sign-request \
    --output json 2>&1)

export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['AccessKeyId'])" 2>/dev/null)
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SecretKey'])" 2>/dev/null)
export AWS_SESSION_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['Credentials']['SessionToken'])" 2>/dev/null)

if [ -z "$AWS_ACCESS_KEY_ID" ]; then
    echo "✗ Credentials failed"
    exit 1
fi
echo "✓ Credentials OK"

echo ""
echo "=== Step 4: Verify all files ==="
TOTAL_BYTES=0
COUNT_OK=0
COUNT_FAIL=0

for f in \
    v1.0-trainval_meta.tgz \
    v1.0-trainval01_blobs_lidar.tgz \
    v1.0-trainval02_blobs_lidar.tgz \
    v1.0-trainval03_blobs_lidar.tgz \
    v1.0-trainval04_blobs_lidar.tgz \
    v1.0-trainval05_blobs_lidar.tgz \
    v1.0-trainval06_blobs_lidar.tgz \
    v1.0-trainval07_blobs_lidar.tgz \
    v1.0-trainval08_blobs_lidar.tgz \
    v1.0-trainval09_blobs_lidar.tgz \
    v1.0-trainval10_blobs_lidar.tgz
do
    SIZE=$(aws s3api head-object \
        --bucket data.nuscenes.org \
        --key "public/v1.0/$f" \
        --region us-east-1 \
        --query "ContentLength" \
        --output text 2>&1)

    if [[ "$SIZE" =~ ^[0-9]+$ ]]; then
        SIZE_GB=$(python3 -c "print(f'{$SIZE/1024/1024/1024:.2f}')")
        echo "✓ $f  ${SIZE_GB} GB"
        TOTAL_BYTES=$((TOTAL_BYTES + SIZE))
        COUNT_OK=$((COUNT_OK + 1))
    else
        echo "✗ $f  FAILED"
        COUNT_FAIL=$((COUNT_FAIL + 1))
    fi
done

TOTAL_GB=$(python3 -c "print(f'{$TOTAL_BYTES/1024/1024/1024:.2f}')")
echo ""
echo "=== 结果: $COUNT_OK 成功, $COUNT_FAIL 失败, 总计 ${TOTAL_GB} GB ==="
