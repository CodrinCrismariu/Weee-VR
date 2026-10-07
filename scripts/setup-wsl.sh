#!/usr/bin/env bash
# One-time toolchain setup inside WSL Ubuntu 24.04: Swift (via swiftly), usbmuxd/libimobiledevice, xtool.
# Safe to re-run. Usage (from Windows): wsl -d Ubuntu-24.04 -- bash "/mnt/c/Users/Useru/Documents/Weee Sports - VR/scripts/setup-wsl.sh"
set -euo pipefail

echo "==> apt packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  binutils git gnupg2 libc6-dev libcurl4-openssl-dev libedit2 libgcc-13-dev libncurses-dev \
  libpython3-dev libsqlite3-0 libstdc++-13-dev libxml2-dev libz3-dev pkg-config tzdata unzip zlib1g-dev \
  curl ca-certificates zip rsync usbmuxd libimobiledevice-utils libimobiledevice6 usbutils socat libfuse2t64 file

echo "==> swiftly + Swift toolchain"
export SWIFTLY_HOME_DIR="$HOME/.local/share/swiftly"
if ! command -v swiftly >/dev/null 2>&1 && [ ! -x "$SWIFTLY_HOME_DIR/bin/swiftly" ]; then
  tmp="$(mktemp -d)"
  (cd "$tmp" && curl -fsSLO "https://download.swift.org/swiftly/linux/swiftly-$(uname -m).tar.gz" \
    && tar zxf "swiftly-$(uname -m).tar.gz" \
    && ./swiftly init --assume-yes --quiet-shell-followup)
  rm -rf "$tmp"
fi
# shellcheck disable=SC1091
. "$SWIFTLY_HOME_DIR/env.sh"
hash -r
swift --version

echo "==> xtool"
if ! command -v xtool >/dev/null 2>&1; then
  curl -fL "https://github.com/xtool-org/xtool/releases/latest/download/xtool-$(uname -m).AppImage" -o /tmp/xtool
  chmod +x /tmp/xtool
  sudo mv /tmp/xtool /usr/local/bin/xtool
fi
xtool --help >/dev/null 2>&1 || {
  echo "xtool AppImage could not run via FUSE; enabling extract-and-run fallback"
  grep -q APPIMAGE_EXTRACT_AND_RUN "$HOME/.bashrc" || echo 'export APPIMAGE_EXTRACT_AND_RUN=1' >> "$HOME/.bashrc"
  export APPIMAGE_EXTRACT_AND_RUN=1
}
xtool --help | head -20

echo "==> fast build dir on the Linux filesystem"
mkdir -p "$HOME/weee-build"

echo "==> done. Next: run 'xtool setup' interactively (Apple ID + Xcode.xip)."
