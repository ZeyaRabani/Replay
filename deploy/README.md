# Replay Highlights — self-hosting in one command

A single container runs the whole app: web UI, analysis pipeline, and storage.
Works on an Oracle Cloud Always-Free Ubuntu VM (ARM64 or x86) and on a laptop
with Docker Desktop.

## (a) Oracle Cloud VM — the easy path

1. In the Oracle console, open your VM and click **Cloud Shell** (or SSH in).
2. Paste this one line:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/ZeyaRabani/Replay/main/deploy/install.sh | bash
   ```

   It installs Docker if needed, opens ports 80/443 in the VM's firewall
   (Oracle images block inbound traffic by default — also open port 80 in the
   VCN **Security List** / Network Security Group in the Oracle console),
   clones the repo into `~/replay`, builds and starts the container.
3. Open `http://<your VM's public IP>` in a browser.

To install a specific branch: download the script first, then
`bash install.sh <branch>`.

## (b) Laptop with Docker Desktop

```bash
git clone https://github.com/ZeyaRabani/Replay.git
cd Replay
docker compose -f deploy/docker-compose.yml up -d --build
```

Open `http://localhost`.

## (c) Updating to a new version

On the VM, re-run the installer one-liner — it pulls the latest code and
rebuilds. Or manually:

```bash
cd ~/replay && git pull
docker compose -f deploy/docker-compose.yml up -d --build
```

Your projects live in the `replay_data` docker volume and survive rebuilds.

## (d) YouTube cookies

YouTube sometimes blocks datacenter IPs with a "Sign in to confirm you're not a
bot" check. If a download fails with that message: install a
"Get cookies.txt" browser extension while logged into YouTube, export the file,
and paste its contents into the **Advanced: YouTube cookies** box shown when you
create a YouTube project. Nothing is logged — the file is stored only inside the
container's data volume.

## (e) How long does analysis take?

Processing runs at roughly real-time speed on a small VM: a full ~90-minute
match takes about **1–2 hours** on 4 cores (audio features + motion + scoring).
You can close the browser; the pipeline keeps running server-side. Individual
clip rendering afterwards is minutes.

## Notes

- Image is multi-arch (`linux/amd64`, `linux/arm64`) — same Dockerfile on both.
- Data persists in the `replay_data` volume (`HL_WORKDIR=/data`).
- The app listens on container port 8000, published as host port 80.
