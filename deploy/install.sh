#!/usr/bin/env bash
# Replay Highlights — one-command installer for Ubuntu 22.04/24.04
# (Oracle Cloud Always-Free ARM VMs and ordinary x86 machines alike).
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ZeyaRabani/Replay/main/deploy/install.sh | bash
#   bash install.sh [branch]   # default branch: main
set -euo pipefail

BRANCH="${1:-main}"
REPO_URL="https://github.com/ZeyaRabani/Replay.git"
APP_DIR="$HOME/replay"

echo "== Replay Highlights installer (branch: $BRANCH)"

# --- Docker -----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
    echo "== Installing Docker"
    curl -fsSL https://get.docker.com | sh
fi

# docker compose plugin (get.docker.com installs it, but be safe on older setups)
if ! docker compose version >/dev/null 2>&1; then
    echo "== Installing docker compose plugin"
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin || sudo apt-get install -y docker-compose
fi

# let the current user run docker without sudo
if ! groups "$USER" | grep -qw docker; then
    echo "== Adding $USER to the docker group"
    sudo usermod -aG docker "$USER"
    DOCKER="sudo docker"
else
    DOCKER="docker"
fi

# --- Firewall ---------------------------------------------------------------
# Oracle Cloud Ubuntu images ship with restrictive iptables rules that drop
# inbound traffic even when the cloud security list allows it. Open 80/443.
open_port() {
    local port=$1
    # Oracle images ship a REJECT-all near the end of INPUT; ACCEPTs must be
    # inserted BEFORE it (a fixed -I INPUT 6 lands after REJECT on some images).
    local idx
    idx=$(sudo iptables -L INPUT --line-numbers -n \
        | awk '$2 == "REJECT" || $2 == "DROP" {print $1; exit}')
    idx=${idx:-6}
    if ! sudo iptables -C INPUT -m state --state NEW -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
        sudo iptables -I INPUT "$idx" -m state --state NEW -p tcp --dport "$port" -j ACCEPT
        echo "== Opened port $port in iptables (rule $idx)"
    fi
}
open_port 80
open_port 443
if command -v netfilter-persistent >/dev/null 2>&1; then
    sudo netfilter-persistent save >/dev/null 2>&1 || true
elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y iptables-persistent >/dev/null 2>&1 \
        && sudo netfilter-persistent save >/dev/null 2>&1 || true
fi

# --- Code -------------------------------------------------------------------
if [ -d "$APP_DIR/.git" ]; then
    echo "== Updating existing checkout in $APP_DIR"
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" checkout "$BRANCH"
    git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
    echo "== Cloning $REPO_URL ($BRANCH) into $APP_DIR"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$APP_DIR"
fi

# --- Run --------------------------------------------------------------------
cd "$APP_DIR"

# Public URL so the hosted frontend can send large uploads straight here.
PUBLIC_IP="$(curl -fsSL --max-time 5 https://ifconfig.me 2>/dev/null || true)"
if [ -z "${PUBLIC_IP:-}" ]; then
    PUBLIC_IP="$(curl -fsSL --max-time 5 -H 'Authorization: Bearer Oracle' \
        http://169.254.169.254/opc/v2/vnics/ 2>/dev/null \
        | grep -o '"publicIp":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
fi
if [ -n "${PUBLIC_IP:-}" ]; then
    echo "HL_PUBLIC_URL=http://$PUBLIC_IP" > deploy/.env
fi

echo "== Building and starting the container (first build takes several minutes)"
$DOCKER compose -f deploy/docker-compose.yml up -d --build

PUBLIC_IP="$(curl -fsSL --max-time 5 http://169.254.169.254/opc/v1/instance/metadata 2>/dev/null \
    | grep -o '"publicIp":"[^"]*"' | cut -d'"' -f4 || true)"
[ -z "${PUBLIC_IP:-}" ] && PUBLIC_IP="$(curl -fsSL --max-time 5 https://ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"

cat <<EOF

== Done.
   Open  http://$PUBLIC_IP  in your browser.
   First visit: type any username to sign in, then add a YouTube URL or a video file.
   If this was your first install and docker required sudo above, log out and back
   in once so '$USER' joins the docker group — updates will then work without sudo.
EOF
