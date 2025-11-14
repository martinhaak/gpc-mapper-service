### API Referenz

Kurzübersicht aller Endpunkte, Parameter und Beispielaufrufe. Basis: Flask‑Service aus `mapping_service_GPC_full_xml.py`.

- Basis‑URL: `http://<HOST>:<PORT>` (Standard-Port via `GPC_PORT`: 5002)
- Authentifizierung (falls `GPC_API_KEY` gesetzt ist, empfohlen): 
  - Header: `X-API-Key: YOUR_KEY`
  - oder Query: `?api_key=YOUR_KEY`
  - oder JSON-Body: `{ "api_key": "YOUR_KEY" }`

Hinweis zu Shell/Zsh: Sonderzeichen (z. B. `!`) im Key maskieren oder Single-Quotes verwenden.

```bash
curl -H 'X-API-Key: gpc-TEST-2025!' http://<HOST>:5002/status
# oder:
curl "http://<HOST>:5002/status?api_key=gpc-TEST-2025%21"
```


### GET /status
- Zweck: Gesundheits-/Ladezustand, Metadaten von Ebenen und Attribut-Embeddings

Optionale Parameter:
- `api_key` (Query) – falls API‑Key aktiviert

Header-Alternative:
- `X-API-Key: YOUR_KEY`

Beispiel:
```bash
curl -H 'X-API-Key: YOUR_KEY' http://<HOST>:5002/status
```

Beispielantwort (gekürzt):
```json
{
  "xml_path": "data/GPC as of May 2025 v20250509 DE.xml",
  "levels": {
    "segment": { "model": "all-MiniLM-L6-v2", "persisted": true, "count": 42, "persisted_created_at": 1710000000 },
    "family":  { "model": "all-MiniLM-L6-v2", "persisted": true, "count": 300, "persisted_created_at": 1710000000 },
    "class":   { "model": "all-MiniLM-L6-v2", "persisted": true, "count": 1200, "persisted_created_at": 1710000000 },
    "brick":   { "model": "all-MiniLM-L6-v2", "persisted": true, "count": 4500, "persisted_created_at": 1710000000 }
  },
  "attributes": {
    "model": "all-MiniLM-L6-v2",
    "persisted": true,
    "count": 20000,
    "persisted_created_at": 1710000000
  }
}
```


### POST /rebuild
- Zweck: Embeddings neu erstellen und persistieren

JSON‑Body:
- `level` string, optional, Standard: `"all"`
  - Erlaubt: `"segment" | "family" | "class" | "brick" | "all"`

Auth (falls aktiv): Header `X-API-Key` oder `api_key` im Query/Body.

Beispiele:
```bash
# alle Ebenen
curl -X POST http://<HOST>:5002/rebuild \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"level":"all"}'

# nur eine Ebene (schneller)
curl -X POST http://<HOST>:5002/rebuild \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"level":"segment"}'
```

Antwort (Beispiel):
```json
{ "message": "Ebene 'segment' neu erstellt und gespeichert (XML).", "level": "segment" }
```
oder
```json
{ "message": "Alle Ebenen neu erstellt und gespeichert (XML).", "levels": ["segment","family","class","brick"] }
```

Fehler:
- 400 bei unbekannter Ebene
- 500 bei internen Fehlern (z. B. XML‑Parsing)


### POST /match
- Zweck: Ähnlichkeitssuche über Titel (und optional Attribute) auf gewählter Ebene

JSON‑Body (Standard‑Modus, Ebene segment/family/class/brick):
- `query` string, Pflicht: Suchtext
- `level` string, optional, Standard: `"brick"`
  - Erlaubt: `"segment" | "family" | "class" | "brick"`
- Parent‑Filter optional (engen Ergebnisraum ein): 
  - `SegmentCode` string|number
  - `FamilyCode` string|number
  - `ClassCode` string|number
- Attribute‑Suche (nur relevant für `level: "brick"`):
  - `search_attributes` boolean, optional, Standard: false
  - `attribute_weight` float, optional, Standard: 1.0
  - `search_attribute_type` boolean, optional, Standard: false
  - `search_attribute_value` boolean, optional, Standard: false
    - Wenn beide false und `search_attributes=true`, wird der kombinierte Standard verwendet (Typ+Wert zusammen).

Spezial‑Modus (nur Attribute‑Werte suchen):
- `level` string: `"attr_value"` oder `"attribute_value"`
  - Sucht ausschließlich Attribute‑Werte, gruppiert nach Brick; Antwort‑`level` bleibt `"brick"`.
  - Parent‑Filter wie oben werden berücksichtigt.

Beispiele:
```bash
# Standard: Brick-Suche nur mit Titeln
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"apple","level":"brick"}'

# Brick-Suche kombiniert mit Attributen (Typ+Wert), Gewicht 1.25
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"stainless steel 18/10","level":"brick","search_attributes":true,"attribute_weight":1.25}'

# Nur Attribute-Typen berücksichtigen
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"color","level":"brick","search_attributes":true,"search_attribute_type":true}'

# Nur Attribute-Werte berücksichtigen
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"blue","level":"brick","search_attributes":true,"search_attribute_value":true}'

# Spezial-Level: reine Attribute-Wert-Suche (gruppiert je Brick)
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"18/10","level":"attr_value"}'

# Mit Parent-Filtern (z. B. auf bestimmte Class einschränken)
curl -X POST http://<HOST>:5002/match \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_KEY' \
  -d '{"query":"knife","level":"brick","ClassCode":123456}'
```

Antwort (Beispiel, Standard‑Modus):
```json
{
  "query": "apple",
  "level": "brick",
  "matches": [
    {
      "code": 10001234,
      "title": "Apple Cutter",
      "level": "brick",
      "score": 0.9123,
      "source": "title",
      "path_titles": "Kitchen > Tools > Cutters > Apple Cutter",
      "path": {
        "segment": {"code":"10","title":"Kitchen"},
        "family":  {"code":"1010","title":"Tools"},
        "class":   {"code":"101010","title":"Cutters"},
        "brick":   {"code":"10001234","title":"Apple Cutter"}
      }
      /* optional, wenn Attribute beteiligt waren:
      "attribute": {
        "type_code": "A01", "type_text": "Material",
        "value_code": "V01", "value_text": "Stainless Steel",
        "score": 0.1234
      }
      */
    }
  ]
}
```

Antwort (Beispiel, Spezial‑Modus `attr_value`):
```json
{
  "query": "18/10",
  "level": "brick",
  "matches": [
    {
      "code": 10004567,
      "title": "Some Brick Title",
      "level": "brick",
      "score": 0.8012,
      "source": "attribute_value_only",
      "path_titles": "…",
      "path": { "segment": {...}, "family": {...}, "class": {...}, "brick": {...} },
      "attribute": {
        "type_code": "A01",
        "type_text": "Material",
        "value_code": "V18",
        "value_text": "18/10",
        "score": 0.8012
      }
    }
  ]
}
```

Fehler:
- 400 wenn `query` fehlt oder `level` unbekannt
- 401/403 bei fehlendem/falschem API‑Key (falls aktiviert)
- 500 bei internen Fehlern


### Authentifizierung zusammengefasst
- Serverseitig in der Runtime/Unit setzen:
  - `GPC_API_KEY=YOUR_KEY`
- Clientseitig mitliefern (mindestens eine der Varianten):
  - Header: `-H 'X-API-Key: YOUR_KEY'`
  - Query: `?api_key=YOUR_KEY`
  - Body: `{ "api_key": "YOUR_KEY", ... }`


### Tipps
- Bei kleinem RAM zunächst nur `segment/family/class` rebuilden, dann `brick`.
- Sonderzeichen im API‑Key korrekt quoten/escapen.
- Parent‑Filter nutzen, um Domain/Teilbäume einzugrenzen (`SegmentCode`/`FamilyCode`/`ClassCode`). 


