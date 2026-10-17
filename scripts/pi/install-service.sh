#!/usr/bin/env bash
set -euo pipefail
project="$(cd "$(dirname "$0")/../.." && pwd)"
config="${1:-configs/pilot.yaml}"
unit=/etc/systemd/system/robustness-gap-coordinator.service
sudo tee "$unit" >/dev/null <<UNIT
[Unit]
Description=robustness-gap coordinator ($config)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$project
ExecStart=$project/scripts/pi/coordinator.sh $config
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now robustness-gap-coordinator
echo "Installed. Status: systemctl status robustness-gap-coordinator"
echo "Log and access code: journalctl -u robustness-gap-coordinator -n 20"
