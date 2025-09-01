#!/usr/bin/env bash
set -euo pipefail

DEST="${HOME}/.aws/credentials"

# 2) 备份已有文件（如果存在）
if [ -f "$DEST" ]; then
  cp "$DEST" "${DEST}.bak.$(date +%Y%m%d%H%M%S)"
fi

# 3) 写入你要保存的内容 —— 把下面这段替换为“下面的文字”
cat > "$DEST" <<'EOF'

EOF

# 4) 设定权限（AWS 推荐 600）
chmod 600 "$DEST"

CONFIG="${HOME}/.aws/config"

if [ -f "$CONFIG" ]; then
  cp "$CONFIG" "${CONFIG}.bak.$(date +%Y%m%d%H%M%S)"
fi

cat > "$CONFIG" <<'EOF'
[default]
credential_process = python3 /workspace/AgentRL/api/get_credentials.py -s greenland -p greenland "arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole"


[profile greenland]
role_arn = arn:aws:iam::684288478426:role/GreenlandCrossAccountAccessRole
credential_source = EcsContainer
EOF

chmod 600 "$CONFIG"

echo "写入完成：$DEST $CONFIG "

PROFILE_FILE="${HOME}/.zshrc"
if [ -n "${BASH_VERSION-}" ] || [ "$(basename "${SHELL:-}")" = "bash" ]; then
  PROFILE_FILE="${HOME}/.bashrc"
fi
if [ ! -f "$PROFILE_FILE" ]; then
  if   [ -f "${HOME}/.bashrc" ]; then PROFILE_FILE="${HOME}/.bashrc"
  elif [ -f "${HOME}/.zshrc"  ]; then PROFILE_FILE="${HOME}/.zshrc"
  else PROFILE_FILE="${HOME}/.profile"; fi
fi

add_line() { grep -qxF "$1" "$PROFILE_FILE" 2>/dev/null || echo "$1" >> "$PROFILE_FILE"; }
add_line 'export AWS_PROFILE=default'
add_line 'export AWS_CONFIG_FILE="$HOME/.aws/config"'

source "$PROFILE_FILE"


