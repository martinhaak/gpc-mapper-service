### Schnelle Tests (lokal oder auf dem Droplet)

Kurze Anleitung, um den Service schnell lauffähig zu bekommen und mit ein paar Requests zu testen.

### 1) Voraussetzungen
- Python 3, `python3-venv`
- Git
- GPC-XML liegt unter `data/GPC as of May 2025 v20250509 DE.xml` (ggf. anlegen/kopieren)

```bash
sudo apt update && sudo apt install -y python3-venv git
```

### 2) Code holen und Umgebung einrichten
```bash
cd ~
git clone git@github.com:martinhaak/gpc-mapper-service.git gpc-mapping-service
cd gpc-mapping-service
python3 -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 3) XML bereitstellen (falls noch nicht vorhanden)
```bash
mkdir -p data
# Beispiel: vom lokalen Rechner auf ein Droplet kopieren
# scp "data/GPC as of May 2025 v20250509 DE.xml" app@<DROPLET_IP>:~/gpc-mapping-service/data/
```

### 4) Umgebungsvariablen setzen (Beispiel)
```bash
export GPC_MODEL_NAME=all-MiniLM-L6-v2
export GPC_EMB_DIR=emb/all-MiniLM-L6-v2
export GPC_PORT=5002
export GPC_API_KEY=change_me_secret_key
```

### 5) Schnellstart (Entwicklung)
```bash
source venv/bin/activate
python mapping_service_GPC_full_xml.py
```

Neues Terminal/Fenster für Tests verwenden.

### 6) Smoke-Tests
- Status:
```bash
curl -H "X-API-Key: change_me_secret_key" http://127.0.0.1:5002/status
```

- Embeddings selektiv aufbauen (schneller):
```bash
curl -X POST http://127.0.0.1:5002/rebuild \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"level":"segment"}'
```

- Komplettaufbau (dauert je nach Hardware):
```bash
curl -X POST http://127.0.0.1:5002/rebuild \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"level":"all"}'
```

- Einfache Suche (Titel, Ebene `brick`):
```bash
curl -X POST http://127.0.0.1:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"query":"apple","level":"brick"}'
```

- Suche inkl. Attributen (kombiniertes Scoring):
```bash
curl -X POST http://127.0.0.1:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change_me_secret_key' \
  -d '{"query":"stainless steel 18/10","level":"brick","search_attributes":true,"attribute_weight":1.0}'
```

### 7) Schneller gunicorn-Test (ohne systemd)
```bash
source venv/bin/activate
gunicorn -w 1 -b 0.0.0.0:${GPC_PORT} "mapping_service_GPC_full_xml:app"
# Test
curl -H "X-API-Key: change_me_secret_key" http://127.0.0.1:5002/status
```

### 8) Häufige Probleme (Kurz)
- 401/403: `GPC_API_KEY` nicht gesetzt oder falsch im Request.
- Datei nicht gefunden: XML liegt nicht im `data/`-Ordner unter dem erwarteten Namen.
- Langsam/Out-of-Memory: Ebene zunächst klein (segment/family/class) rebuilden; ggf. Swap anlegen.
- Port belegt: anderen `GPC_PORT` wählen.

Für Produktion: siehe `docs/production-systemd-gunicorn.md`.



