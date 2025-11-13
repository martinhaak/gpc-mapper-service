# GPC Mapping Service (API only)

Flask-API zum Matching von GPC (aus XML) mit optionalen Attribut-Suchen.

## Struktur
- mapping_service_GPC_full_xml.py: Flask-Service
- data/: lege hier die GPC-XML ab (Pfad in Script: data/GPC as of May 2025 v20250509 DE.xml)
- emb/: Persistenz der Embeddings (modell-/instanzspezifisch, via ENV GPC_EMB_DIR)

## Setup (Droplet)
```bash
sudo apt update && sudo apt install -y python3-venv
cd ~/gpc-mapping-service
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# XML bereitstellen
mkdir -p data
# scp die XML hierher, Dateiname wie im Script oder ENV anpassen
```

## Start (Entwicklung)
```bash
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=emb/all-MiniLM-L6-v2
export GPC_PORT=5002
source venv/bin/activate
python mapping_service_GPC_full_xml.py
```

## Embeddings aufbauen
```bash
curl -X POST http://localhost:5002/rebuild -H 'Content-Type: application/json' -d '{"level":"all"}'
```

## Production (Beispiel: gunicorn + systemd)
### gunicorn Startbefehl (einfach)
```bash
source venv/bin/activate
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=emb/all-MiniLM-L6-v2
export GPC_PORT=5002
export GPC_API_KEY=change_me_secret_key
gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} "mapping_service_GPC_full_xml:app"
```

### systemd Unit (Beispiel)
```ini
[Unit]
Description=GPC Mapping Service
After=network.target

[Service]
User=%i
WorkingDirectory=/home/%i/gpc-mapping-service
Environment=GPC_MODEL_NAME=all-MiniLM-L6-v2
Environment=GPC_EMB_DIR=/home/%i/gpc-mapping-service/emb/all-MiniLM-L6-v2
Environment=GPC_PORT=5002
Environment=GPC_API_KEY=change_me_secret_key
ExecStart=/home/%i/gpc-mapping-service/venv/bin/gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} mapping_service_GPC_full_xml:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Aktivieren (als root):
```bash
sudo cp gpc-mapping.service /etc/systemd/system/gpc-mapping@${USER}.service
sudo systemctl daemon-reload
sudo systemctl enable gpc-mapping@${USER}
sudo systemctl start gpc-mapping@${USER}
```

## API Endpunkte
- GET /status
- POST /rebuild { level: segment|family|class|brick|all }
- POST /match { query, level, Filter, search_attributes, search_attribute_type, search_attribute_value, attribute_weight }

### API Key Schutz
Wenn `GPC_API_KEY` gesetzt ist, muss ein passender API Key mitgesendet werden, sonst gibt der Service 401/403 zurück.

Unterstützte Wege:
- Query-Parameter: `?api_key=YOUR_KEY` (z. B. GET /status?api_key=YOUR_KEY)
- Header: `X-API-Key: YOUR_KEY`
- JSON-Body-Feld: `{ "api_key": "YOUR_KEY", ... }`

Beispiele:
```bash
# Status
curl "http://localhost:5002/status?api_key=YOUR_KEY"
curl -H "X-API-Key: YOUR_KEY" http://localhost:5002/status

# Rebuild
curl -X POST "http://localhost:5002/rebuild?api_key=YOUR_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"level":"all"}'

# Match
curl -X POST http://localhost:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"apple","level":"brick","search_attributes":true}'
```

