#!/bin/bash
# 查看下载进度
echo "===== 下载日志（最近 20 行）====="
tail -20 /data/mmood3d/download.log 2>/dev/null || echo "日志文件不存在"

echo ""
echo "===== 磁盘使用 ====="
df -h /data

echo ""
echo "===== 数据目录大小 ====="
du -sh /data/mmood3d/code/data/nuscenes/* 2>/dev/null || echo "数据目录为空"

echo ""
echo "===== 后台进程 ====="
ps aux | grep -E "download_nuscenes|wget|tar" | grep -v grep || echo "没有下载进程在运行"
