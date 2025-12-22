#!/usr/bin/env bash
set -euo pipefail

RULE_PATH="/etc/udev/rules.d/99-metabonk-uinput.rules"
GROUP_NAME="${METABONK_UINPUT_GROUP:-uinput}"
USER_NAME="${METABONK_UINPUT_USER:-$USER}"

echo "[uinput] Ensuring uinput kernel module is loaded..."
sudo modprobe uinput

echo "[uinput] Creating group ${GROUP_NAME} (if missing)..."
sudo groupadd -f "${GROUP_NAME}"

echo "[uinput] Writing udev rule to ${RULE_PATH}..."
echo "KERNEL==\"uinput\", SUBSYSTEM==\"misc\", MODE=\"0660\", GROUP=\"${GROUP_NAME}\", OPTIONS+=\"static_node=uinput\"" | sudo tee "${RULE_PATH}" >/dev/null

echo "[uinput] Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput || true

echo "[uinput] Adding ${USER_NAME} to ${GROUP_NAME}..."
sudo usermod -aG "${GROUP_NAME}" "${USER_NAME}"

echo "[uinput] Done."
echo "[uinput] IMPORTANT: log out/in (or run 'newgrp ${GROUP_NAME}') before starting MetaBonk."
