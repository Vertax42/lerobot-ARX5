#!/bin/bash
# 包装脚本：运行set_can.sh而不显示libtinfo警告

# 使用env -i创建干净的环境，避免conda库冲突
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash -c "
cd /home/ubuntu/lerobot-ARX5
./set_can.sh
" 2>/dev/null
