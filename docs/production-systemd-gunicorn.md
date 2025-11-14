### Production-Betrieb mit systemd + gunicorn (Schritt für Schritt)

Diese Anleitung zeigt, wie du den GPC Mapping Service als produktiven Linux-Dienst unter Ubuntu (24.04 empfohlen) betreibst – inklusive API-Key, optionaler Mehrinstanzfähigkeit und Update-Workflow.

### 0) Übersicht
- App: Flask (WSGI) → gunicorn
- Dienststeuerung: systemd
- API-Schutz: per `GPC_API_KEY`
- Optional: mehrere Instanzen (verschiedene Modelle/Ports) via systemd-Template

### 1) Voraussetzungen (Droplet mit root)
- Ubuntu 24.04 LTS (oder 22.04)
- root-User (du arbeitest bereits als `root`)
- Netzwerkzugriff (Port 5002 oder Reverse Proxy)
- Python 3 samt `python3-venv`, `git`

```bash
sudo apt update
sudo apt install -y python3-venv git
```

Hinweis: Du arbeitest als `root`, daher werden alle Pfade unter `/root/...` gesetzt.

### 2) Projekt klonen und Virtualenv einrichten
```bash
cd /root
git clone git@github.com:martinhaak/gpc-mapper-service.git gpc-mapping-service
cd gpc-mapping-service
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 3) Daten bereitstellen (XML, Embeddings-Verzeichnis)
```bash
mkdir -p /root/gpc-mapping-service/data /root/gpc-mapping-service/emb
# XML vom lokalen Rechner hochladen (Beispiel)
# scp "data/GPC as of May 2025 v20250509 DE.xml" root@<DROPLET_IP>:/root/gpc-mapping-service/data/
```

### 4) Umgebungsvariablen (Beispielwerte)
- `GPC_MODEL_NAME` (Standard: `all-MiniLM-L6-v2`)
- `GPC_EMB_DIR` (empfohlen absolut, z. B. `/root/gpc-mapping-service/emb/all-MiniLM-L6-v2` – pro Modell separat)
- `GPC_PORT` (z. B. `5002`)
- `GPC_API_KEY` (dein geheimer Schlüssel)
- Ressourcenschonung / Attribute:
- `GPC_BATCH_SIZE` (z. B. `8` oder `16`)
- `GPC_ATTR_LAZY` (`1` = Attribute on-the-fly, `0` = aus Persistenz laden)
- `GPC_ATTR_MAX_ROWS` (Cap im Lazy-Modus, z. B. `40000`)

Zum schnellen Testen in der Shell:
```bash
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=/root/gpc-mapping-service/emb/all-MiniLM-L6-v2
export GPC_PORT=5002
export GPC_API_KEY=change_me_secret_key
export GPC_BATCH_SIZE=8
export GPC_ATTR_LAZY=1
export GPC_ATTR_MAX_ROWS=40000
# optional: Threads begrenzen (RAM/CPU-schonend)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
```

### 5) Einmaliger Test mit gunicorn
```bash
source venv/bin/activate
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=/root/gpc-mapping-service/emb/all-MiniLM-L6-v2
export GPC_PORT=5002
export GPC_API_KEY=change_me_secret_key
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
User=root
WorkingDirectory=/root/gpc-mapping-service
EnvironmentFile=/etc/gpc-mapping/%i.env
ExecStart=/root/gpc-mapping-service/venv/bin/gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} mapping_service_GPC_full_xml:app
Restart=on-failure
# Ressourcen begrenzen (optional, je nach Bedarf auch in Env-Datei setzen)
Environment=OMP_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=TOKENIZERS_PARALLELISM=false

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
GPC_EMB_DIR=/root/gpc-mapping-service/emb/all-MiniLM-L6-v2
GPC_PORT=5002
GPC_API_KEY=change_me_secret_key
GPC_BATCH_SIZE=8
GPC_ATTR_LAZY=0
GPC_ATTR_MAX_ROWS=40000
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
EOF

# Optional zweite Instanz: mpnet
sudo tee /etc/gpc-mapping/mpnet.env >/dev/null <<'EOF'
GPC_MODEL_NAME=all-mpnet-base-v2
GPC_EMB_DIR=/root/gpc-mapping-service/emb/all-mpnet-base-v2
GPC_PORT=5003
GPC_API_KEY=change_me_secret_key
GPC_BATCH_SIZE=8
GPC_ATTR_LAZY=0
GPC_ATTR_MAX_ROWS=40000
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
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
Hinweis: Der Dienst lädt persistierte Embeddings aus `GPC_EMB_DIR`. Für Attribut-Suche ohne Lazy‑Modus sollten die drei Attribut‑Dateien einmalig persistiert werden:
- `gpc_embeddings_attr_xml.npz` (kombiniert Typ+Wert)
- `gpc_embeddings_attr_type_xml.npz`
- `gpc_embeddings_attr_value_xml.npz`

Ressourcenschonende Offline‑Persistierung (einmalig):
```bash
cd /root/gpc-mapping-service
source venv/bin/activate
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=/root/gpc-mapping-service/emb/all-MiniLM-L6-v2
export GPC_BATCH_SIZE=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python - <<'PY'
from mapping_service_GPC_full_xml import (
  ensure_attributes_loaded, ensure_attribute_types_loaded, ensure_attribute_values_loaded
)
ensure_attributes_loaded(force=True)
ensure_attribute_types_loaded(force=True)
ensure_attribute_values_loaded(force=True)
print("Attribute embeddings persisted.")
PY
```

Für Titel‑Embeddings (segment/family/class/brick) kannst du weiterhin via API erzeugen:
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
cd /root/gpc-mapping-service
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
- Persistenz wird nicht erkannt (counts=0)? Prüfen:
  - `GPC_EMB_DIR` absolut und korrekt?
  - Dateinamen exakt wie erwartet: `gpc_embeddings_{segment|family|class|brick}_xml.npz` sowie `gpc_embeddings_attr[_type|_value]_xml.npz`
  - `GPC_MODEL_NAME` identisch zum bei Persistierung verwendeten Modell.
  - Optional testweise manuell laden:
    ```bash
    python - <<'PY'
    import numpy as np, os
    base=os.environ.get("GPC_EMB_DIR")
    for n in ("segment","family","class","brick"):
        p=f"{base}/gpc_embeddings_{n}_xml.npz"
        print(p, "exists:", os.path.exists(p))
    PY
    ```

### 14) Mehrere Modelle parallel
- Für jede Instanz eigene Env-Datei (eigenes `GPC_MODEL_NAME`, `GPC_EMB_DIR`, `GPC_PORT`, gemeinsamer oder eigener `GPC_API_KEY`).
- Dienste z. B. `gpc-mapping@mini`, `gpc-mapping@mpnet` parallel betreiben und hinter Nginx getrennt exponieren.


