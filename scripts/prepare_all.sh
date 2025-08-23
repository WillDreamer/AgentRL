#!/usr/bin/env bash
# 安全/健壮性
set -Eeuo pipefail
IFS=$'\n\t'

# ---- 小工具 ----
log() { printf "\n\033[1;32m[+] %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m[!] %s\033[0m\n" "$*"; }
die() { printf "\033[1;31m[x] %s\033[0m\n" "$*"; exit 1; }
trap 'die "脚本在第 $LINENO 行出错（exit=$?）。"' ERR

as_root() {
  if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
    sudo -H bash -lc "$*"
  else
    bash -lc "$*"
  fi
}

# ---- 0) 预检查 ----
log "检查 sudo 可用性（非 root 情况）"
if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  if ! sudo -n true 2>/dev/null; then
    warn "当前用户需要sudo密码；确保以可sudo的用户运行。"
  fi
fi

# ---- 1) 基础准备：tmux / git-lfs / workspace 权限 ----
log "更新 APT 并安装 tmux / git-lfs（非交互）"
as_root 'export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y tmux git-lfs >/dev/null 2>&1 || apt-get install -y tmux git-lfs
'
log "初始化 git-lfs（用户级）"
git lfs install --skip-repo >/dev/null 2>&1 || true

log "确保 /workspace 存在并可写"
as_root 'mkdir -p /workspace'
as_root 'chmod -R a+w /workspace/'

# ---- 2) Conda 初始化 ----
log "初始化 conda 环境"
if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  warn "未检测到 conda；将尝试运行用户提供的 setup_conda.sh 进行安装。"
fi

# ---- 3) 运行项目脚本（按存在与先后顺序）----
log "运行 setup_conda.sh（若存在）"
[[ -f setup_conda.sh ]] && bash setup_conda.sh || warn "未找到 setup_conda.sh，跳过"

log "运行 setup_env.sh / setuo_env.sh（择其一）"
if [[ -f setup_env.sh ]]; then
  bash setup_env.sh
elif [[ -f setuo_env.sh ]]; then        # 用户原脚本的拼写
  bash setuo_env.sh
else
  warn "未找到 setup_env.sh 或 setuo_env.sh，跳过"
fi

log "运行 recipe/webshop/setup_webshop.sh（若存在）"
[[ -f recipe/webshop/setup_webshop.sh ]] && bash recipe/webshop/setup_webshop.sh || warn "未找到 recipe/webshop/setup_webshop.sh，跳过"

# 再次确保 conda 激活（有些脚本可能改变了环境）
if command -v conda >/dev/null 2>&1; then
  conda activate base || true
fi


log "校验登录状态（whoami）"
if command -v hf >/dev/null 2>&1; then
  hf auth whoami || die "whoami 失败"
else
  huggingface-cli whoami || die "whoami 失败"
fi

log "全部完成 🎉"
