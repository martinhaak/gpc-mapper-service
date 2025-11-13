from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import hashlib, json, os, threading, time, re
from typing import Dict, Optional, List
import xml.etree.ElementTree as ET

# -----------------------
# Konfiguration
# -----------------------
XML_PATH   = 'data/GPC as of May 2025 v20250509 DE.xml'
# Konfigurierbar über ENV: GPC_MODEL_NAME, GPC_EMB_DIR, GPC_PORT
MODEL_NAME = os.getenv('GPC_MODEL_NAME', 'all-MiniLM-L6-v2')
BATCH_SIZE = 64
ENCODING   = 'utf-8'
EMB_DIR    = os.getenv('GPC_EMB_DIR', '.')  # pro Modell/Instanz separater Speicherpfad empfohlen
PORT       = int(os.getenv('GPC_PORT', '5002'))

# Erwartete Spalten / Hierarchie
COLS = {
    "segment": {"code": "SegmentCode", "title": "SegmentTitle"},
    "family":  {"code": "FamilyCode",  "title": "FamilyTitle",  "parents": ["SegmentCode"]},
    "class":   {"code": "ClassCode",   "title": "ClassTitle",   "parents": ["SegmentCode", "FamilyCode"]},
    "brick":   {"code": "BrickCode",   "title": "BrickTitle",   "parents": ["SegmentCode", "FamilyCode", "ClassCode"]},
}

# Dateinamen je Ebene (separat für XML-Quelle)
def emb_path(level: str) -> str:
    return os.path.join(EMB_DIR, f"gpc_embeddings_{level}_xml.npz")

# -----------------------
# XML -> DataFrame
# -----------------------
def expand_slash_ellipsis(text: str) -> List[str]:
    if not isinstance(text, str) or '/' not in text:
        return [text]
    parts = [p.strip() for p in re.split(r'\s*/\s*', text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    tokenized = [p.split() for p in parts]
    pref: List[str] = []
    for ws in zip(*tokenized):
        if len(set(ws)) == 1:
            pref.append(ws[0])
        else:
            break
    suf: List[str] = []
    rev_tok = [list(reversed(w)) for w in tokenized]
    for ws in zip(*rev_tok):
        if len(set(ws)) == 1:
            suf.append(ws[0])
        else:
            break
    suf = list(reversed(suf))

    full = parts[-1]

    def german_tail(word: str) -> str:
        m = re.search(r'([a-zäöüß]+)$', word)
        return m.group(1) if m else ''

    last_tokens = tokenized[-1]
    last_tail_word = last_tokens[-1] if last_tokens else ''
    last_tail_lowerchunk = german_tail(full)

    variants = set()
    for p in parts:
        if p.endswith('-'):
            base = p[:-1].strip()
            if last_tail_word:
                variants.add((' '.join(pref + [base + last_tail_word] + suf)).strip())
            if last_tail_lowerchunk and last_tail_lowerchunk not in last_tail_word.lower():
                variants.add((' '.join(pref + [base + last_tail_lowerchunk] + suf)).strip())
        else:
            variants.add((' '.join(pref + [p] + suf)).strip())
    variants.add(text.strip())
    return sorted({v for v in variants if v})

def load_categories_from_xml() -> pd.DataFrame:
    """Parst die GPC-XML in eine flache Tabelle mit allen benötigten Spalten.

    Erzeugt Zeilen hauptsächlich auf Brick-Ebene (vollständiger Pfad: Segment/Family/Class/Brick),
    sodass deduplizieren pro Ebene funktioniert. Falls eine Klasse/Familie keine Bricks hätte,
    würden diese Ebenen nur über Brick-Zeilen nicht erscheinen; aktuell wird davon ausgegangen,
    dass produktive Schemas Bricks enthalten.
    """
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    rows = []

    # Struktur: <schema><segment><family><class><brick ... /></class>...</family>...</segment>...
    for seg in root.findall('.//segment'):
        seg_code = seg.get('code') or ''
        seg_text = seg.get('text') or ''

        for fam in seg.findall('./family'):
            fam_code = fam.get('code') or ''
            fam_text = fam.get('text') or ''

            for cls in fam.findall('./class'):
                cls_code = cls.get('code') or ''
                cls_text = cls.get('text') or ''

                bricks = cls.findall('./brick')
                if not bricks:
                    # Fallback: falls keine Bricks, zumindest eine Klassen-Zeile erstellen
                    rows.append({
                        'SegmentCode': seg_code,
                        'SegmentTitle': seg_text,
                        'FamilyCode': fam_code,
                        'FamilyTitle': fam_text,
                        'ClassCode': cls_code,
                        'ClassTitle': cls_text,
                        'BrickCode': None,
                        'BrickTitle': None,
                    })
                    continue

                for br in bricks:
                    br_code = br.get('code') or ''
                    br_text = br.get('text') or ''
                    rows.append({
                        'SegmentCode': seg_code,
                        'SegmentTitle': seg_text,
                        'FamilyCode': fam_code,
                        'FamilyTitle': fam_text,
                        'ClassCode': cls_code,
                        'ClassTitle': cls_text,
                        'BrickCode': br_code,
                        'BrickTitle': br_text,
                    })

    df = pd.DataFrame(rows, dtype=str)
    # Sicherheit: NaN statt leere Strings für fehlende Werte
    df = df.replace({'': np.nan})
    return df

def load_attributes_from_xml() -> pd.DataFrame:
    """Parst Attribute und Werte je Brick in eine Tabelle und verknüpft sie mit Parent-Codes.

    Spalten: SegmentCode, FamilyCode, ClassCode, BrickCode, BrickTitle,
             AttTypeCode, AttTypeText, AttValueCode, AttValueText
    """
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    rows = []
    for seg in root.findall('.//segment'):
        seg_code = seg.get('code') or ''
        seg_text = seg.get('text') or ''  # ungenutzt, könnte aber ergänzt werden
        for fam in seg.findall('./family'):
            fam_code = fam.get('code') or ''
            for cls in fam.findall('./class'):
                cls_code = cls.get('code') or ''
                for br in cls.findall('./brick'):
                    br_code = br.get('code') or ''
                    br_text = br.get('text') or ''
                    for att_type in br.findall('./attType'):
                        at_code = att_type.get('code') or ''
                        at_text = att_type.get('text') or ''
                        values = att_type.findall('./attValue')
                        if not values:
                            rows.append({
                                'SegmentCode': seg_code,
                                'FamilyCode': fam_code,
                                'ClassCode': cls_code,
                                'BrickCode': br_code,
                                'BrickTitle': br_text,
                                'AttTypeCode': at_code,
                                'AttTypeText': at_text,
                                'AttValueCode': None,
                                'AttValueText': None,
                            })
                        else:
                            for att_val in values:
                                av_code = att_val.get('code') or ''
                                av_text = att_val.get('text') or ''
                                rows.append({
                                    'SegmentCode': seg_code,
                                    'FamilyCode': fam_code,
                                    'ClassCode': cls_code,
                                    'BrickCode': br_code,
                                    'BrickTitle': br_text,
                                    'AttTypeCode': at_code,
                                    'AttTypeText': at_text,
                                    'AttValueCode': av_code,
                                    'AttValueText': av_text,
                                })

    df = pd.DataFrame(rows, dtype=str)
    df = df.replace({'': np.nan})
    return df

# -----------------------
# Utils
# -----------------------
def compute_fingerprint(rows: pd.DataFrame, model_name: str, level: str) -> str:
    key_code = COLS[level]["code"]
    key_title = COLS[level]["title"]
    data_list = rows[[key_code, key_title]].astype(str).values.tolist()
    blob = json.dumps({"level": level, "model": model_name, "rows": data_list}, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.md5(blob).hexdigest()

def save_embeddings(path: str, embeddings: np.ndarray, rows: pd.DataFrame, model_name: str, fingerprint: str, level: str) -> None:
    key_code = COLS[level]["code"]
    key_title = COLS[level]["title"]
    # Zielverzeichnis sicherstellen
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        codes=rows[key_code].astype(str).values,
        titles=rows[key_title].astype(str).values,
        level=level,
        model_name=model_name,
        fingerprint=fingerprint,
        created_at=int(time.time())
    )

def try_load_embeddings(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {
            "embeddings": data['embeddings'],
            "codes": data['codes'],
            "titles": data['titles'],
            "level": str(data['level']),
            "model_name": str(data['model_name']),
            "fingerprint": str(data['fingerprint']),
            "created_at": int(data['created_at'])
        }
    except Exception:
        return None

def build_embeddings(model, titles: list) -> np.ndarray:
    return model.encode(
        titles,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    )

def normalize_df_for_level(df: pd.DataFrame, level: str, parent_filters: Optional[dict] = None) -> pd.DataFrame:
    meta = COLS[level]
    cols_needed = [meta["code"], meta["title"]]
    if "parents" in meta:
        cols_needed += meta["parents"]

    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"XML-DataFrame fehlt Spalten für Ebene '{level}': {missing}")

    tmp = df[cols_needed].dropna(subset=[meta["code"], meta["title"]]).copy()

    if parent_filters:
        for k, v in parent_filters.items():
            if k in tmp.columns and v is not None:
                tmp = tmp[tmp[k].astype(str) == str(v)]

    # Expand slash-ellipsis in titles to separate variants
    title_col = meta["title"]
    expanded_rows: List[dict] = []
    for _, r in tmp.iterrows():
        title_val = r[title_col]
        variants = expand_slash_ellipsis(str(title_val))
        for v in variants:
            nr = r.copy()
            nr[title_col] = v
            expanded_rows.append(nr)
    tmp2 = pd.DataFrame(expanded_rows)
    tmp2 = tmp2.drop_duplicates(subset=[meta["code"], meta["title"]]).reset_index(drop=True)
    return tmp2

# -----------------------
# App-State
# -----------------------
app = Flask(__name__)
lock = threading.Lock()
model = SentenceTransformer(MODEL_NAME)
df = load_categories_from_xml()
df_attr = load_attributes_from_xml()

# Cache im Speicher: je Ebene => (rows_df, embeddings)
MEM: Dict[str, Dict] = {lvl: {"rows": None, "emb": None, "fp": None} for lvl in COLS.keys()}
ATTR: Dict[str, Optional[pd.DataFrame]] = {"rows": None}
ATTR_EMB: Dict[str, Optional[np.ndarray]] = {"emb": None, "fp": None}
ATTR_TYPE: Dict[str, Optional[pd.DataFrame]] = {"rows": None}
ATTR_TYPE_EMB: Dict[str, Optional[np.ndarray]] = {"emb": None, "fp": None}
ATTR_VALUE: Dict[str, Optional[pd.DataFrame]] = {"rows": None}
ATTR_VALUE_EMB: Dict[str, Optional[np.ndarray]] = {"emb": None, "fp": None}

def attr_emb_path() -> str:
    return os.path.join(EMB_DIR, "gpc_embeddings_attr_xml.npz")

def attr_type_emb_path() -> str:
    return os.path.join(EMB_DIR, "gpc_embeddings_attr_type_xml.npz")

def attr_value_emb_path() -> str:
    return os.path.join(EMB_DIR, "gpc_embeddings_attr_value_xml.npz")

def ensure_attributes_loaded(force: bool = False):
    """Erzeugt/ladt Embeddings für Attribut-Begriffe (Type/Value Texte)."""
    with lock:
        rows = df_attr.dropna(subset=['BrickCode']).copy()
        # Textrepräsentation für Embeddings: Kombination aus Attributtyp und -wert, wo vorhanden
        combined_texts = []
        for _, r in rows.iterrows():
            if pd.notna(r.get('AttValueText')):
                combined_texts.append(f"{r.get('AttTypeText','')} {r.get('AttValueText','')}".strip())
            else:
                combined_texts.append(f"{r.get('AttTypeText','')}".strip())
        rows = rows.assign(_AttrText=pd.Series(combined_texts, index=rows.index))

        blob = json.dumps({
            'rows': rows[['BrickCode', '_AttrText']].astype(str).values.tolist(),
            'model': MODEL_NAME
        }, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        fp = hashlib.md5(blob).hexdigest()

        if (not force) and ATTR_EMB["emb"] is not None and ATTR_EMB["fp"] == fp:
            ATTR["rows"] = rows
            return

        meta = try_load_embeddings(attr_emb_path())
        if (not force) and meta is not None and str(meta.get('model_name')) == MODEL_NAME:
            # Kein Level-Feld hier, nur Fingerprint prüfen
            if str(meta.get('fingerprint')) == fp and len(meta['titles']) == len(rows):
                ATTR["rows"] = rows
                ATTR_EMB["emb"] = meta['embeddings']
                ATTR_EMB["fp"] = fp
                return

        emb = build_embeddings(model, rows['_AttrText'].tolist())
        # Zielverzeichnis sicherstellen
        d = os.path.dirname(attr_emb_path())
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            attr_emb_path(),
            embeddings=emb.astype(np.float32),
            titles=rows['_AttrText'].astype(str).values,
            model_name=MODEL_NAME,
            fingerprint=fp,
            created_at=int(time.time())
        )
        ATTR["rows"] = rows
        ATTR_EMB["emb"] = emb
        ATTR_EMB["fp"] = fp

def ensure_attribute_types_loaded(force: bool = False):
    """Erzeugt/lädt Embeddings nur für AttTypeText pro Brick."""
    with lock:
        rows = df_attr.dropna(subset=['BrickCode', 'AttTypeText']).copy()
        rows = rows.assign(_TypeText=rows['AttTypeText'].astype(str))

        blob = json.dumps({
            'rows': rows[['BrickCode', '_TypeText']].astype(str).values.tolist(),
            'model': MODEL_NAME
        }, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        fp = hashlib.md5(blob).hexdigest()

        if (not force) and ATTR_TYPE_EMB["emb"] is not None and ATTR_TYPE_EMB["fp"] == fp:
            ATTR_TYPE["rows"] = rows
            return

        meta = try_load_embeddings(attr_type_emb_path())
        if (not force) and meta is not None and str(meta.get('model_name')) == MODEL_NAME:
            if str(meta.get('fingerprint')) == fp and len(meta['titles']) == len(rows):
                ATTR_TYPE["rows"] = rows
                ATTR_TYPE_EMB["emb"] = meta['embeddings']
                ATTR_TYPE_EMB["fp"] = fp
                return

        emb = build_embeddings(model, rows['_TypeText'].tolist())
        d = os.path.dirname(attr_type_emb_path())
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            attr_type_emb_path(),
            embeddings=emb.astype(np.float32),
            titles=rows['_TypeText'].astype(str).values,
            model_name=MODEL_NAME,
            fingerprint=fp,
            created_at=int(time.time())
        )
        ATTR_TYPE["rows"] = rows
        ATTR_TYPE_EMB["emb"] = emb
        ATTR_TYPE_EMB["fp"] = fp

def ensure_attribute_values_loaded(force: bool = False):
    """Erzeugt/lädt Embeddings nur für AttValueText pro Brick."""
    with lock:
        rows = df_attr.dropna(subset=['BrickCode', 'AttValueText']).copy()
        rows = rows.assign(_ValueText=rows['AttValueText'].astype(str))

        blob = json.dumps({
            'rows': rows[['BrickCode', '_ValueText']].astype(str).values.tolist(),
            'model': MODEL_NAME
        }, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        fp = hashlib.md5(blob).hexdigest()

        if (not force) and ATTR_VALUE_EMB["emb"] is not None and ATTR_VALUE_EMB["fp"] == fp:
            ATTR_VALUE["rows"] = rows
            return

        meta = try_load_embeddings(attr_value_emb_path())
        if (not force) and meta is not None and str(meta.get('model_name')) == MODEL_NAME:
            if str(meta.get('fingerprint')) == fp and len(meta['titles']) == len(rows):
                ATTR_VALUE["rows"] = rows
                ATTR_VALUE_EMB["emb"] = meta['embeddings']
                ATTR_VALUE_EMB["fp"] = fp
                return

        emb = build_embeddings(model, rows['_ValueText'].tolist())
        d = os.path.dirname(attr_value_emb_path())
        if d:
            os.makedirs(d, exist_ok=True)
        np.savez_compressed(
            attr_value_emb_path(),
            embeddings=emb.astype(np.float32),
            titles=rows['_ValueText'].astype(str).values,
            model_name=MODEL_NAME,
            fingerprint=fp,
            created_at=int(time.time())
        )
        ATTR_VALUE["rows"] = rows
        ATTR_VALUE_EMB["emb"] = emb
        ATTR_VALUE_EMB["fp"] = fp

def ensure_level_loaded(level: str, parent_filters: Optional[dict] = None, force: bool = False):
    assert level in COLS, f"Unbekannte Ebene: {level}"
    with lock:
        rows = normalize_df_for_level(df, level)
        fp = compute_fingerprint(rows, MODEL_NAME, level)

        if (not force) and MEM[level]["emb"] is not None and MEM[level]["fp"] == fp:
            return

        meta = try_load_embeddings(emb_path(level))
        if (not force) and meta is not None and meta["model_name"] == MODEL_NAME and meta["level"] == level:
            if meta["fingerprint"] == fp and len(meta["titles"]) == len(rows):
                MEM[level] = {
                    "rows": rows,
                    "emb": meta["embeddings"],
                    "fp": fp
                }
                return

        emb = build_embeddings(model, rows[COLS[level]["title"]].tolist())
        save_embeddings(emb_path(level), emb, rows, MODEL_NAME, fp, level)
        MEM[level] = {"rows": rows, "emb": emb, "fp": fp}

def match_on_level(query: str, level: str, parent_filters: Optional[dict] = None, k: int = 10, search_attributes: bool = False, attribute_weight: float = 1.0, search_attribute_type: bool = False, search_attribute_value: bool = False):
    def build_path_fields(row: pd.Series, tgt_level: str) -> Dict[str, object]:
        order = [
            ("segment", "SegmentCode", "SegmentTitle"),
            ("family",  "FamilyCode",  "FamilyTitle"),
            ("class",   "ClassCode",   "ClassTitle"),
            ("brick",   "BrickCode",   "BrickTitle"),
        ]
        titles_path = []
        path_obj: Dict[str, Dict[str, Optional[str]]] = {}
        for lvl, code_col, title_col in order:
            code_val = row.get(code_col)
            title_val = row.get(title_col)
            if pd.notna(title_val):
                titles_path.append(str(title_val))
            path_obj[lvl] = {
                "code": (str(code_val) if pd.notna(code_val) else None),
                "title": (str(title_val) if pd.notna(title_val) else None),
            }
            if lvl == tgt_level:
                break
        return {
            "path_titles": " > ".join(titles_path),
            "path": path_obj,
        }

    ensure_level_loaded(level)

    rows = MEM[level]["rows"]
    emb = MEM[level]["emb"]

    if parent_filters:
        mask = pd.Series([True] * len(rows))
        for pcol, pval in parent_filters.items():
            if pcol in rows.columns and pval is not None:
                mask &= (rows[pcol].astype(str) == str(pval))
        candidates = rows[mask].reset_index(drop=True)
        cand_indices = np.flatnonzero(mask.values)
        if len(candidates) == 0:
            return []
        emb_subset = emb[cand_indices]
    else:
        candidates = rows
        emb_subset = emb

    q = model.encode([query], convert_to_numpy=True)
    sims = cosine_similarity(q, emb_subset)[0]
    code_col = COLS[level]["code"]
    title_col = COLS[level]["title"]

    # Deduplicate by code: keep best-scoring variant per code
    order_indices = sims.argsort()[::-1]
    best_by_code: Dict[str, Dict] = {}
    for idx in order_indices:
        row = candidates.iloc[idx]
        code_key = str(row[code_col])
        score_val = float(round(float(sims[idx]), 4))
        if code_key in best_by_code and best_by_code[code_key]['score'] >= score_val:
            continue
        code_raw = row[code_col]
        try:
            code_val = int(code_raw)
        except Exception:
            code_val = code_raw
        path_fields = build_path_fields(row, level)
        best_by_code[code_key] = {
            "code": code_val,
            "title": row[title_col],
            "level": level,
            "score": score_val,
            "source": "title",
            **path_fields,
        }
        if len(best_by_code) >= max(k * 3, k):
            # allow some buffer before final top-k cut
            pass

    out = sorted(best_by_code.values(), key=lambda x: x['score'], reverse=True)[:k]

    # Optional: Attributsuche nur für Brick-Ebene (kombiniertes Scoring)
    if level == 'brick' and search_attributes:
        ensure_attributes_loaded()
        attr_rows = ATTR["rows"]
        attr_emb = ATTR_EMB["emb"]

        use_type = bool(search_attribute_type)
        use_value = bool(search_attribute_value)
        use_combined = not (use_type or use_value)  # Standard: bisheriges Verhalten

        # Parent-Filter anwenden (auf BrickCode via df rows join)
        if parent_filters:
            mask = pd.Series([True] * len(attr_rows))
            for pcol, pval in parent_filters.items():
                if pcol in attr_rows.columns and pval is not None:
                    mask &= (attr_rows[pcol].astype(str) == str(pval))
            attr_candidates = attr_rows[mask].reset_index(drop=True)
            cand_indices = np.flatnonzero(mask.values)
            if len(attr_candidates) == 0:
                attr_candidates = None
        else:
            attr_candidates = attr_rows
            cand_indices = None

        # Map für besten Attribut-Score je Brick aufbauen (kombiniert aus gewählten Modi)
        best_attr_by_brick: Dict[str, Dict[str, object]] = {}

        # Combined-Modus
        if use_combined and attr_candidates is not None and len(attr_candidates) > 0:
            if cand_indices is not None:
                attr_emb_subset = attr_emb[cand_indices]
            else:
                attr_emb_subset = attr_emb
            sims_attr_raw = cosine_similarity(q, attr_emb_subset)[0]
            tmp = attr_candidates.assign(_sim=pd.Series(sims_attr_raw, index=attr_candidates.index))
            grouped = tmp.sort_values('_sim', ascending=False).groupby('BrickCode', as_index=False).first()
            for _, r in grouped.iterrows():
                bcode = str(r.get('BrickCode')) if pd.notna(r.get('BrickCode')) else None
                if not bcode:
                    continue
                best_attr_by_brick[bcode] = {
                    'attr_sim': float(r['_sim']),
                    'type_code': r.get('AttTypeCode'),
                    'type_text': r.get('AttTypeText'),
                    'value_code': r.get('AttValueCode'),
                    'value_text': r.get('AttValueText'),
                }

        # Type-only Modus
        if use_type:
            ensure_attribute_types_loaded()
            type_rows = ATTR_TYPE["rows"]
            type_emb = ATTR_TYPE_EMB["emb"]
            if parent_filters:
                mask = pd.Series([True] * len(type_rows))
                for pcol, pval in parent_filters.items():
                    if pcol in type_rows.columns and pval is not None:
                        mask &= (type_rows[pcol].astype(str) == str(pval))
                type_candidates = type_rows[mask].reset_index(drop=True)
                type_idx = np.flatnonzero(mask.values)
            else:
                type_candidates = type_rows
                type_idx = None
            if type_candidates is not None and len(type_candidates) > 0:
                type_emb_subset = type_emb[type_idx] if type_idx is not None else type_emb
                sims_type = cosine_similarity(q, type_emb_subset)[0]
                tmp = type_candidates.assign(_sim=pd.Series(sims_type, index=type_candidates.index))
                grouped = tmp.sort_values('_sim', ascending=False).groupby('BrickCode', as_index=False).first()
                for _, r in grouped.iterrows():
                    bcode = str(r.get('BrickCode')) if pd.notna(r.get('BrickCode')) else None
                    if not bcode:
                        continue
                    entry = best_attr_by_brick.get(bcode, {
                        'attr_sim': 0.0,
                        'type_code': None, 'type_text': None,
                        'value_code': None, 'value_text': None,
                    })
                    entry['attr_sim'] = float(entry['attr_sim']) + float(r['_sim'])
                    entry['type_code'] = r.get('AttTypeCode')
                    entry['type_text'] = r.get('AttTypeText')
                    best_attr_by_brick[bcode] = entry

        # Value-only Modus
        if use_value:
            ensure_attribute_values_loaded()
            value_rows = ATTR_VALUE["rows"]
            value_emb = ATTR_VALUE_EMB["emb"]
            if parent_filters:
                mask = pd.Series([True] * len(value_rows))
                for pcol, pval in parent_filters.items():
                    if pcol in value_rows.columns and pval is not None:
                        mask &= (value_rows[pcol].astype(str) == str(pval))
                value_candidates = value_rows[mask].reset_index(drop=True)
                value_idx = np.flatnonzero(mask.values)
            else:
                value_candidates = value_rows
                value_idx = None
            if value_candidates is not None and len(value_candidates) > 0:
                value_emb_subset = value_emb[value_idx] if value_idx is not None else value_emb
                sims_val = cosine_similarity(q, value_emb_subset)[0]
                tmp = value_candidates.assign(_sim=pd.Series(sims_val, index=value_candidates.index))
                grouped = tmp.sort_values('_sim', ascending=False).groupby('BrickCode', as_index=False).first()
                for _, r in grouped.iterrows():
                    bcode = str(r.get('BrickCode')) if pd.notna(r.get('BrickCode')) else None
                    if not bcode:
                        continue
                    entry = best_attr_by_brick.get(bcode, {
                        'attr_sim': 0.0,
                        'type_code': None, 'type_text': None,
                        'value_code': None, 'value_text': None,
                    })
                    entry['attr_sim'] = float(entry['attr_sim']) + float(r['_sim'])
                    entry['value_code'] = r.get('AttValueCode')
                    entry['value_text'] = r.get('AttValueText')
                    best_attr_by_brick[bcode] = entry

        # Kombiniere Titel-Score mit Attribut-Score: title + attribute_weight * attr
        weight = float(attribute_weight)
        # Index für schnellen Zugriff der Titelkandidaten
        out_by_code: Dict[str, int] = {str(item['code']): i for i, item in enumerate(out)}

        # Aktualisiere bestehende Titelkandidaten
        for code_str, idx_out in list(out_by_code.items()):
            attr_info = best_attr_by_brick.get(code_str)
            if not attr_info:
                continue
            combined = float(out[idx_out]['score']) + weight * float(attr_info['attr_sim'])
            out[idx_out]['score'] = float(round(combined, 4))
            out[idx_out]['source'] = 'combined'
            out[idx_out]['attribute'] = {
                'type_code': attr_info.get('type_code'),
                'type_text': attr_info.get('type_text'),
                'value_code': attr_info.get('value_code'),
                'value_text': attr_info.get('value_text'),
                'score': float(round(weight * float(attr_info['attr_sim']), 4))
            }

        # Bricks, die nur über Attribute gefunden wurden (nicht in Titel-Topk)
        # Optional hinzufügen mit reinem Attributbeitrag
        for bcode, attr_info in best_attr_by_brick.items():
            if bcode in out_by_code:
                continue
            try:
                brick_code_val = int(bcode)
            except Exception:
                brick_code_val = bcode
            try:
                titles_row = df[df['BrickCode'].astype(str) == str(bcode)].head(1)
                if len(titles_row) == 1:
                    merged_row = titles_row.iloc[0]
                else:
                    merged_row = pd.Series({'BrickCode': bcode, 'BrickTitle': None})
            except Exception:
                merged_row = pd.Series({'BrickCode': bcode, 'BrickTitle': None})
            path_fields = build_path_fields(merged_row, 'brick')
            out.append({
                'code': brick_code_val,
                'title': merged_row.get('BrickTitle'),
                'level': level,
                'score': float(round(weight * float(attr_info['attr_sim']), 4)),
                'source': 'attribute',
                **path_fields,
                'attribute': {
                    'type_code': attr_info.get('type_code'),
                    'type_text': attr_info.get('type_text'),
                    'value_code': attr_info.get('value_code'),
                    'value_text': attr_info.get('value_text'),
                    'score': float(round(weight * float(attr_info['attr_sim']), 4))
                }
            })

        # Nach Score sortieren und auf k beschränken
        out = sorted(out, key=lambda x: x['score'], reverse=True)[:k]

    return out

# -----------------------
# Routes
# -----------------------
@app.route('/status', methods=['GET'])
def status():
    with lock:
        statuses = {}
        for lvl in COLS.keys():
            meta = try_load_embeddings(emb_path(lvl))
            statuses[lvl] = {
                "model": MODEL_NAME,
                "persisted": meta is not None,
                "count": int(len(MEM[lvl]["rows"])) if MEM[lvl]["rows"] is not None else (int(len(meta["titles"])) if meta else 0),
                "persisted_created_at": meta["created_at"] if meta else None
            }
        # Attribute-Status
        attr_meta = try_load_embeddings(attr_emb_path())
        attr_count = int(len(ATTR["rows"])) if ATTR["rows"] is not None else (int(len(attr_meta['titles'])) if attr_meta else 0)
        return jsonify({
            "xml_path": XML_PATH,
            "levels": statuses,
            "attributes": {
                "model": MODEL_NAME,
                "persisted": attr_meta is not None,
                "count": attr_count,
                "persisted_created_at": (attr_meta["created_at"] if attr_meta else None)
            }
        })

@app.route('/rebuild', methods=['POST'])
def rebuild():
    data = request.get_json(silent=True) or {}
    level = (data.get("level") or "all").lower()
    try:
        if level == "all":
            for lvl in COLS.keys():
                ensure_level_loaded(lvl, force=True)
            return jsonify({"message": "Alle Ebenen neu erstellt und gespeichert (XML).", "levels": list(COLS.keys())})
        else:
            if level not in COLS:
                return jsonify({"error": f"Unbekannte Ebene: {level}"}), 400
            ensure_level_loaded(level, force=True)
            return jsonify({"message": f"Ebene '{level}' neu erstellt und gespeichert (XML).", "level": level})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/match', methods=['POST'])
def match():
    data = request.get_json(silent=True) or {}
    query = data.get("query", "")
    level = (data.get("level") or "brick").lower()
    search_attributes = bool(data.get("search_attributes", False))
    attribute_weight = float(data.get("attribute_weight", 1.0))
    search_attribute_type = bool(data.get("search_attribute_type", False))
    search_attribute_value = bool(data.get("search_attribute_value", False))

    if not query:
        return jsonify({"error": "Kein Such-String übergeben"}), 400

    # Parent-Filter früh ermitteln (für alle Pfade nutzbar)
    parent_filters = {}
    for key in ("SegmentCode", "FamilyCode", "ClassCode"):
        if key in data:
            parent_filters[key] = data[key]

    # Spezial-Level: nur Attribut-Werte suchen, Brick-Code/Path zurückgeben
    if level in ("attr_value", "attribute_value"):
        try:
            ensure_attribute_values_loaded()
            value_rows = ATTR_VALUE["rows"]
            value_emb = ATTR_VALUE_EMB["emb"]

            # Parent-Filter anwenden
            if parent_filters:
                mask = pd.Series([True] * len(value_rows))
                for pcol, pval in parent_filters.items():
                    if pcol in value_rows.columns and pval is not None:
                        mask &= (value_rows[pcol].astype(str) == str(pval))
                candidates = value_rows[mask].reset_index(drop=True)
                cand_idx = np.flatnonzero(mask.values)
                if len(candidates) == 0:
                    return jsonify({"query": query, "level": "brick", "matches": []})
                emb_subset = value_emb[cand_idx]
            else:
                candidates = value_rows
                emb_subset = value_emb

            q = model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(q, emb_subset)[0]
            tmp = candidates.assign(_sim=pd.Series(sims, index=candidates.index))
            # Bester Value-Treffer je Brick (danach erneut nach Score sortieren)
            grouped = tmp.sort_values('_sim', ascending=False).groupby('BrickCode', as_index=False).first()
            grouped = grouped.sort_values('_sim', ascending=False)

            # Ergebnisse bauen (Level im Output bleibt 'brick')
            matches = []
            for _, r in grouped.head(10).iterrows():
                bcode = r.get('BrickCode')
                try:
                    code_val = int(bcode) if pd.notna(bcode) else bcode
                except Exception:
                    code_val = bcode
                # Pfad/Titel aus Haupt-DF
                try:
                    titles_row = df[df['BrickCode'].astype(str) == str(bcode)].head(1)
                    base_row = titles_row.iloc[0] if len(titles_row) == 1 else r
                except Exception:
                    base_row = r

                # Path-Helper aus match_on_level-Kontext
                def _build_path_fields(row: pd.Series) -> Dict[str, object]:
                    order = [
                        ("segment", "SegmentCode", "SegmentTitle"),
                        ("family",  "FamilyCode",  "FamilyTitle"),
                        ("class",   "ClassCode",   "ClassTitle"),
                        ("brick",   "BrickCode",   "BrickTitle"),
                    ]
                    titles_path = []
                    path_obj: Dict[str, Dict[str, Optional[str]]] = {}
                    for lvl, code_col, title_col in order:
                        code_v = row.get(code_col)
                        title_v = row.get(title_col)
                        if pd.notna(title_v):
                            titles_path.append(str(title_v))
                        path_obj[lvl] = {
                            "code": (str(code_v) if pd.notna(code_v) else None),
                            "title": (str(title_v) if pd.notna(title_v) else None),
                        }
                        if lvl == 'brick':
                            break
                    return {
                        "path_titles": " > ".join(titles_path),
                        "path": path_obj,
                    }

                path_fields = _build_path_fields(base_row)
                matches.append({
                    "code": code_val,
                    "title": base_row.get('BrickTitle'),
                    "level": "brick",
                    "score": float(round(float(r['_sim']), 4)),
                    "source": "attribute_value_only",
                    **path_fields,
                    "attribute": {
                        "type_code": r.get('AttTypeCode'),
                        "type_text": r.get('AttTypeText'),
                        "value_code": r.get('AttValueCode'),
                        "value_text": r.get('AttValueText'),
                        "score": float(round(float(r['_sim']), 4))
                    }
                })

            return jsonify({"query": query, "level": "brick", "matches": matches})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Standard-Level-Prüfung
    if level not in COLS:
        return jsonify({"error": f"Unbekannte Ebene: {level}"}), 400

    # Standard-Level: Eltern aus COLS ableiten (optional, parent_filters ist schon gesetzt)
    meta = COLS[level]
    for p in meta.get("parents", []):
        if p in data:
            parent_filters[p] = data[p]

    try:
        results = match_on_level(query=query, level=level, parent_filters=parent_filters, k=10, search_attributes=search_attributes, attribute_weight=attribute_weight, search_attribute_type=search_attribute_type, search_attribute_value=search_attribute_value)
        return jsonify({"query": query, "level": level, "matches": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Hinweis: Für Produktion besser WSGI-Server nutzen (gunicorn/uvicorn)
    app.run(host='0.0.0.0', port=PORT)


