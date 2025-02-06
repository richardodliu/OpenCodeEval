#!/bin/bash

# 进入当前目录下的 data 文件夹
cd OpenCodeEval/benchmark/data || { echo "data 文件夹不存在！"; exit 1; }

# 遍历 data 文件夹下的所有文件夹
for dir in */; do
    # 检查是否是目录
    if [ -d "$dir" ]; then
        # 移除尾部的斜杠（/）并创建压缩文件
        dir_name=$(basename "$dir")
        tar -czf "$dir_name.tar.gz" "$dir_name"
        echo "$dir_name 压缩完成！"
    fi
done