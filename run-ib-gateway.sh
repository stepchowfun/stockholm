#!/usr/bin/env bash

# Make Bash log commands and not silently ignore errors.
set -euxo pipefail

# Use a stable container name so cleanup targets only this service.
readonly CONTAINER_NAME=ib-gateway

# Stop the gateway container without masking the wrapper's exit status.
cleanup() {
  docker container stop "$CONTAINER_NAME" || true
}

# Stop the container after termination and before starting a replacement instance.
trap cleanup EXIT
cleanup

# Run the gateway.
docker container run \
  --env "TWS_USERID=$(cat "$HOME/ib-username")" \
  --env 'AUTO_LOGOFF_TIME=10:00 PM' \
  --env ALLOW_BLIND_TRADING=yes \
  --env BYPASS_WARNING=yes \
  --env EXISTING_SESSION_DETECTED_ACTION=primary \
  --env READ_ONLY_API=no \
  --env RELOGIN_AFTER_TWOFA_TIMEOUT=yes \
  --env TIME_ZONE=America/New_York \
  --env TRADING_MODE=live \
  --env TWOFA_TIMEOUT_ACTION=exit \
  --env TWS_ACCEPT_INCOMING=accept \
  --env TWS_PASSWORD_FILE=/home/ibgateway/ib-password \
  --env VNC_SERVER_PASSWORD=vnc_password \
  --mount "type=bind,src=$HOME/ib-password,dst=/home/ibgateway/ib-password,ro" \
  --publish 127.0.0.1:4001:4003 \
  --publish 127.0.0.1:5900:5900 \
  --name "$CONTAINER_NAME" \
  --rm \
  ghcr.io/gnzsnz/ib-gateway@sha256:8b1106efea6c27c14d1a53c881e149a124224e90b4565575334b7f305f7d35b3 # :10.45.1i
