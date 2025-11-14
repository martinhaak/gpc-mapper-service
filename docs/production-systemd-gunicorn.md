### Production-Betrieb mit systemd + gunicorn (Schritt für Schritt)

Diese Anleitung zeigt, wie du den GPC Mapping Service als produktiven Linux-Dienst unter Ubuntu (24.04 empfohlen) betreibst – inklusive API-Key, optionaler Mehrinstanzfähigkeit und Update-Workflow.

### 0) Übersicht
- App: Flask (WSGI) → gunicorn
- Dienststeuerung: systemd
- API-Schutz: per `GPC_API_KEY`
- Optional: mehrere Instanzen (verschiedene Modelle/Ports) via systemd-Template

### 1) Voraussetzungen
- Ubuntu 22.04/24.04 (empfohlen: 24.04 LTS)
- Netzwerkzugriff (Port 5002 oder eigener Port / Reverse Proxy)
- Python 3 und `python3-venv`, `git` installiert

```bash
sudo apt update
sudo apt install -y python3-venv git
```

Optional: eigenen App-User anlegen und nutzen:
```bash
sudo adduser app
sudo usermod -aG sudo app
sudo -iu app
```

### 2) Projekt klonen und Virtualenv einrichten
```bash
cd ~
git clone git@github.com:martinhaak/gpc-mapper-service.git gpc-mapping-service
cd gpc-mapping-service
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 3) Daten bereitstellen (XML, Embeddings-Verzeichnis)
```bash
mkdir -p data emb
# XML vom lokalen Rechner hochladen (Beispiel)
# scp "data/GPC as of May 2025 v20250509 DE.xml" app@<DROPLET_IP>:~/gpc-mapping-service/data/
```

### 4) Umgebungsvariablen (Beispielwerte)
- `GPC_MODEL_NAME` (Standard: `all-MiniLM-L6-v2`)
- `GPC_EMB_DIR` (z. B. `emb/all-MiniLM-L6-v2` – pro Modell separat)
- `GPC_PORT` (z. B. `5002`)
- `GPC_API_KEY` (dein geheimer Schlüssel)

Zum schnellen Testen in der Shell:
```bash
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=emb/all-MiniLM-L6-v2
export GPC_PORT=5002
export GPC_API_KEY=change_me_secret_key
```

### 5) Einmaliger Test mit gunicorn
```bash
source venv/bin/activate
gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} "mapping_service_GPC_full_xml:app"
# Test von extern: curl -H "X-API-Key: change_me_secret_key" http://<SERVER_IP>:5002/status
```
Strg+C beendet den Test.

### 6) systemd-Template-Unit anlegen (Mehrinstanz-fähig)
Wir nutzen eine Template-Unit `gpc-mapping@.service` und pro Instanz eine Env-Datei.

```bash
sudo tee /etc/systemd/system/gpc-mapping@.service >/dev/null <<'EOF'
[Unit]
Description=GPC Mapping Service (%i)
After=network.target

[Service]
User=app
WorkingDirectory=/home/app/gpc-mapping-service
EnvironmentFile=/etc/gpc-mapping/%i.env
ExecStart=/home/app/gpc-mapping-service/venv/bin/gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} mapping_service_GPC_full_xml:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
```

### 7) Pro-Instanz Environment-Dateien erstellen
Erstelle einen Ordner für die Env-Dateien und jeweils eine Datei pro Instanz.

```bash
sudo mkdir -p /etc/gpc-mapping

# Instanz: mini (MiniLM)
sudo tee /etc/gpc-mapping/mini.env >/dev/null <<'EOF'
GPC_MODEL_NAME=all-MiniLM-L6-v2
GPC_EMB_DIR=/home/app/gpc-mapping-service/emb/all-MiniLM-L6-v2
GPC_PORT=5002
GPC_API_KEY=change_me_secret_key
EOF

# Optional zweite Instanz: mpnet
sudo tee /etc/gpc-mapping/mpnet.env >/dev/null <<'EOF'
GPC_MODEL_NAME=all-mpnet-base-v2
GPC_EMB_DIR=/home/app/gpc-mapping-service/emb/all-mpnet-base-v2
GPC_PORT=5003
GPC_API_KEY=change_me_secret_key
EOF
```

Hinweis:
- Passe `User`/`WorkingDirectory`/Pfade an deinen Benutzer und Standort an (oben `app`/`/home/app/...`).
- Achte auf unterschiedliche Ports und eigene Emb-Verzeichnisse pro Modell/Instanz.

### 8) Dienste laden, aktivieren und starten
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gpc-mapping@mini
# Optional weitere Instanz:
# sudo systemctl enable --now gpc-mapping@mpnet

sudo systemctl status gpc-mapping@mini --no-pager
```

### 9) Erst-Initialisierung (Embeddings aufbauen)
```bash
curl -X POST http://127.0.0.1:5002/rebuild \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"level":"all"}'
```

### 10) Tests
```bash
curl -H "X-API-Key: change_me_secret_key" http://<SERVER_IP>:5002/status
curl -X POST http://<SERVER_IP>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"query":"apple","level":"brick","search_attributes":true}'
```

### 11) Firewall / Reverse Proxy (optional, empfohlen)
- UFW (direkter Betrieb ohne Proxy):
```bash
sudo apt install -y ufw
sudo ufw allow OpenSSH
sudo ufw allow 5002/tcp
sudo ufw --force enable
```
- Nginx + TLS (Reverse Proxy auf 127.0.0.1:5002, Let’s Encrypt via certbot):
```bash
sudo apt install -y nginx certbot python3-certbot-nginx
sudo tee /etc/nginx/sites-available/gpc-mapping >/dev/null <<'EOF'
server {
  listen 80;
  server_name YOUR_DOMAIN;

  location / {
    proxy_pass http://127.0.0.1:5002;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
EOF
sudo ln -sf /etc/nginx/sites-available/gpc-mapping /etc/nginx/sites-enabled/gpc-mapping
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d YOUR_DOMAIN --redirect --agree-tos -m you@example.com --non-interactive
```

### 12) Update-/Deploy-Workflow
```bash
cd /home/app/gpc-mapping-service
git fetch origin
git reset --hard origin/main   # oder: git pull --rebase
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
sudo systemctl restart gpc-mapping@mini
# ggf. weitere Instanzen:
# sudo systemctl restart gpc-mapping@mpnet
```

### 13) Troubleshooting
- Status/Logs:
```bash
sudo systemctl status gpc-mapping@mini --no-pager
journalctl -u gpc-mapping@mini -n 200 -f
```
- Port schon belegt? `ss -tulpn | grep 5002`
- API Key Fehler (401/403)? `GPC_API_KEY` in Env-Datei/Unit prüfen.
- Rechteprobleme auf Verzeichnissen? Besitzer/Gruppe des App-Users prüfen (`chown -R app:app ...`).

### 14) Mehrere Modelle parallel
- Für jede Instanz eigene Env-Datei (eigenes `GPC_MODEL_NAME`, `GPC_EMB_DIR`, `GPC_PORT`, gemeinsamer oder eigener `GPC_API_KEY`).
- Dienste z. B. `gpc-mapping@mini`, `gpc-mapping@mpnet` parallel betreiben und hinter Nginx getrennt exponieren.


