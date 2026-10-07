#!/usr/bin/env bash
# Routes WSL's usbmux socket to Apple Mobile Device Service on Windows (from the "Apple Devices" app).
# Needed because usbmuxd over usbipd breaks on large transfers ("message was too large (65536 bytes)"),
# so app installs hang. Requires networkingMode=mirrored in %USERPROFILE%\.wslconfig.
# The iPhone stays connected to *Windows* (do not usbipd-attach it). Run once inside WSL.
set -euo pipefail

sudo systemctl disable --now usbmuxd 2>/dev/null || true
sudo systemctl mask usbmuxd   # udev would otherwise start it if the iPhone ever gets attached to WSL

sudo tee /etc/systemd/system/usbmux-relay.service >/dev/null <<'EOF'
[Unit]
Description=Relay /var/run/usbmuxd to Apple Mobile Device Service on Windows (127.0.0.1:27015)
After=network-online.target

[Service]
ExecStartPre=/bin/rm -f /var/run/usbmuxd
ExecStart=/usr/bin/socat UNIX-LISTEN:/var/run/usbmuxd,fork,mode=666 TCP:127.0.0.1:27015
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now usbmux-relay
sleep 1
idevice_id -l
