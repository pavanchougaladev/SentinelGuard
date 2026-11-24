#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 22.04/24.04 EC2 instance for SentinelGuard.
#
#   sudo REPO_URL=https://github.com/<you>/sentinelguard.git bash deploy/ec2-bootstrap.sh
#
# Without REPO_URL the project must already be copied to APP_DIR (default
# /opt/sentinelguard), e.g. with:
#   scp -r ./sentinelguard ubuntu@<host>:/tmp && ssh ubuntu@<host> 'sudo mv /tmp/sentinelguard /opt/sentinelguard'
#
# Installs Docker Engine + compose plugin, registers a systemd unit that brings
# the stack up on boot, and optionally reverse-proxies it through nginx.
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/sentinelguard}"
REPO_URL="${REPO_URL:-}"
WITH_NGINX="${WITH_NGINX:-1}"

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo, e.g.: sudo REPO_URL=... bash deploy/ec2-bootstrap.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg git
if [[ "$WITH_NGINX" == "1" ]]; then
  apt-get install -y nginx
fi

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# shellcheck disable=SC1091
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

if [[ -n "$REPO_URL" && ! -d "$APP_DIR/.git" ]]; then
  mkdir -p "$APP_DIR"
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi

if [[ ! -f "$APP_DIR/docker-compose.yml" ]]; then
  echo "No docker-compose.yml in $APP_DIR. Copy the project there or set REPO_URL." >&2
  exit 1
fi

if [[ ! -f "$APP_DIR/.env" && -f "$APP_DIR/.env.example" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "Created $APP_DIR/.env - change GRAFANA_ADMIN_PASSWORD before exposing the host."
fi

cat > /etc/systemd/system/sentinelguard.service <<EOF
[Unit]
Description=SentinelGuard fraud detection stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
Environment=COMPOSE_PROJECT_NAME=sentinelguard
ExecStart=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build --remove-orphans
ExecStop=/usr/bin/docker compose -f docker-compose.yml -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable sentinelguard.service

if [[ "$WITH_NGINX" == "1" ]]; then
  install -m 0644 "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/sentinelguard
  ln -sf /etc/nginx/sites-available/sentinelguard /etc/nginx/sites-enabled/sentinelguard
  rm -f /etc/nginx/sites-enabled/default
  nginx -t
  systemctl reload nginx
fi

echo
echo "Bootstrap complete."
echo "  Start:  sudo systemctl start sentinelguard"
echo "  Status: sudo systemctl status sentinelguard"
echo "  Logs:   docker compose -f $APP_DIR/docker-compose.yml -f $APP_DIR/docker-compose.prod.yml logs -f"
echo
echo "Open dashboard : http://<public-ip>/            (or :8501 directly)"
echo "Open API docs  : http://<public-ip>/api/docs    (or :8000/docs directly)"
echo "Open Grafana   : http://<public-ip>:3000        (admin / password from $APP_DIR/.env)"
