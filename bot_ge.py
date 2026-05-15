from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
####
# ATIVAR ARQUIVO bot_ge.py 
# python 3.10+
# Windows CMD:  python bot_ge.py predict --input "Issues LATAM.xlsx" --output "Resultado_Classificado.xlsx"
####

def normalize_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def guess_language(text: str) -> str:
    t = normalize_text(text)
    if not t:
        return "Unknown"
    pt = sum(k in t for k in [" não ", " falha", " treinamento", " bobina", " interferência", " lentidão", " pendência"])
    es = sum(k in t for k in [" no ", " falla", " entrenamiento", " interferencia", " pendiente", " licencia", " equipo"])
    en = sum(k in t for k in [" license", " missing", " crash", " freeze", " network", " worklist"])
    scores = {"PT": pt, "ES": es, "EN": en}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


def determine_severity(text: str) -> str:
    t = normalize_text(text)
    high = [
        "não funciona", "no funciona", "impossibil", "aborta", "abort", "falha", "falla",
        "parou", "parado", "cancel", "crítico", "critico", "urgente", "no arranca", "no pudo arrancar",
        "impossible", "unable", "cannot"
    ]
    low = ["intermitente", "ocasional", "minor", "menor", "ajuste", "configuração", "configuracion"]
    if any(k in t for k in high):
        return "Alta"
    if any(k in t for k in low):
        return "Baixa"
    return "Média"


def extract_component(text: str, modality: Optional[str] = None) -> str:
    t = normalize_text(text)
    pairs = [
        ("pacs", "PACS"),
        ("ris", "RIS"),
        ("worklist", "Worklist"),
        ("dicom", "DICOM"),
        ("bobina", "Bobina/Coil"),
        ("coil", "Bobina/Coil"),
        ("mesa", "Mesa"),
        ("table", "Mesa"),
        ("detector", "Detector"),
        ("grilla", "Detector/Grid"),
        ("grid", "Detector/Grid"),
        ("monitor", "Monitor"),
        ("display", "Monitor"),
        ("teclado", "Teclado"),
        ("keyboard", "Teclado"),
        ("pedal", "Pedal"),
        ("rf", "RF/Amplificador"),
        ("amplificador", "RF/Amplificador"),
        ("trigger", "Trigger"),
        ("ekg", "EKG"),
        ("ecg", "ECG"),
        ("inyector", "Injetor/Contraste"),
        ("injetor", "Injetor/Contraste"),
        ("contraste", "Contraste"),
        ("license", "Licença"),
        ("licen", "Licença"),
        ("pdf", "PDF/Relatório"),
        ("reporte", "PDF/Relatório"),
    ]
    for kw, comp in pairs:
        if kw in t:
            return comp
    if modality:
        m = str(modality).strip()
        if m:
            return m
    return "Geral"



def load_excel_with_best_header(path: str, header_row_1based: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
    """
    Auto-detect header row by finding 'Problem Description' cell in first ~80 rows.
    Returns (df, header_row_1based_used).
    """
    if header_row_1based is not None:
        hdr0 = max(0, header_row_1based - 1)
        df = pd.read_excel(path, header=hdr0, engine="openpyxl")
        return df, header_row_1based

    preview = pd.read_excel(path, header=None, nrows=80, engine="openpyxl")
    target = "problem description"

    best_idx = None
    for i in range(min(len(preview), 80)):
        row = preview.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any(cell == target for cell in row):
            best_idx = i
            break

    if best_idx is None:
        df = pd.read_excel(path, header=0, engine="openpyxl")
        return df, 1

    df = pd.read_excel(path, header=best_idx, engine="openpyxl")
    return df, best_idx + 1


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.astype(str)
    mask_unnamed = cols.str.match(r"^Unnamed")
    return df.loc[:, ~mask_unnamed].copy()


def find_problem_description_column(df: pd.DataFrame) -> str:
    if "Problem Description" in df.columns:
        return "Problem Description"
    # lenient match
    for c in df.columns:
        cl = str(c).lower()
        if "problem" in cl and "description" in cl:
            return c
    raise ValueError("Não encontrei a coluna 'Problem Description' (nem similar) no Excel.")



def normalize_country(country_raw: Any) -> str:
    """Padroniza nomes de países e siglas para o formato completo.

    Retorna 'Unknown' para valores nulos/NaN. Aceita siglas (ex: 'BR', 'BRA')
    e variantes em maiúsculas/minúsculas, normalizando para um nome padrão.
    """
    if country_raw is None or (isinstance(country_raw, float) and math.isnan(country_raw)):
        return "Unknown"

    c = str(country_raw).strip().upper()

    # Mapeamento de siglas e variações para Nome Padrão
    mapping = {
        # Brasil
        "BR": "Brazil",
        "BRA": "Brazil",
        "BRASIL": "Brazil",
        # México
        "MX": "Mexico",
        "MEX": "Mexico",
        "MÉXICO": "Mexico",
        # Argentina
        "AR": "Argentina",
        "ARG": "Argentina",
        # Colômbia
        "CO": "Colombia",
        "COL": "Colombia",
        "COLÔMBIA": "Colombia",
        # Chile
        "CL": "Chile",
        "CHL": "Chile",
        # Peru
        "PE": "Peru",
        "PER": "Peru",
        # Equador
        "EC": "Ecuador",
        "EQU": "Ecuador",
        "EQUADOR": "Ecuador",
        # Outros comuns na LATAM
        "UY": "Uruguay",
        "URU": "Uruguay",
        "PY": "Paraguay",
        "PAR": "Paraguay",
        "PA": "Panama",
        "PAN": "Panama",
        "CR": "Costa Rica",
        "CRI": "Costa Rica",
        "DO": "Dominican Republic",
        "DOM": "Dominican Republic",
        "REP. DOMINICANA": "Dominican Republic",
        "GT": "Guatemala",
        "GTM": "Guatemala",
        "PR": "Puerto Rico",
        "PRI": "Puerto Rico",
    }

    # Retorna mapeamento quando houver
    if c in mapping:
        return mapping[c]

    # Se o usuário já passou um nome de país (ex: 'brazil'), normaliza com Title Case
    return c.title()

# ---------------------------
# Rule Engine (weighted, explainable)
# ---------------------------

@dataclass(frozen=True)
class Rule:
    label: str  # "Group::Category"
    weight: float
    patterns: List[str]          # regex patterns (case-insensitive applied on normalized text)
    require_all: List[str] = None  # regex patterns; if any missing -> rule doesn't apply
    negative: List[str] = None     # if any present -> subtract or block
    block_if_negative: bool = True

    def compile(self):
        # Pre-compile regex patterns
        def comp_list(lst):
            return [re.compile(p, flags=re.IGNORECASE) for p in (lst or [])]
        return {
            "label": self.label,
            "weight": self.weight,
            "patterns": comp_list(self.patterns),
            "require_all": comp_list(self.require_all),
            "negative": comp_list(self.negative),
            "block_if_negative": self.block_if_negative,
        }


def build_rules() -> List[Rule]:
    """
    Rules are intentionally redundant but weighted.
    You should iterate based on Needs_Review samples.
    """
    return [
        # -------- Connectivity / PACS / RIS --------
        Rule(
            label="Conectividade::PACS",
            weight=4.0,
            patterns=[r"\bpacs\b", r"\bdicom\b.*\bsend\b", r"\benv[ií]o\b.*\bpacs\b"],
            negative=[r"\bsem pacs\b"],  # example negative
        ),
        Rule(
            label="Conectividade::RIS/Worklist",
            weight=4.0,
            patterns=[r"\bworklist\b", r"\bris\b", r"\bwl-\d+", r"\bmodality worklist\b"],
        ),
        Rule(
            label="Conectividade::Rede/Network",
            weight=3.0,
            patterns=[
                r"\bnetwork\b", r"\brede\b", r"\bred\b", r"\bconexion\b", r"\bconex[aã]o\b",
                r"\bperdida\b.*\bcomunic", r"\bperda\b.*\bcomunic"
            ],
        ),
        Rule(
            label="Conectividade::Impressora",
            weight=3.0,
            patterns=[r"\bprinter\b", r"\bimpresora\b", r"\bimpressora\b"],
        ),

        # -------- Software / License / Crash --------
        Rule(
            label="Software::Licença",
            weight=5.0,
            patterns=[r"\blicen", r"\blicen[cs]a\b", r"\blicencia\b", r"\blicense\b", r"\bexpired\b", r"\bexpir"],
            require_all=[r"\blicen|license|licencia\b"],
            negative=[r"\blicen.*ok\b", r"\blicen.*valid"],  # reduce false positives
        ),
        Rule(
            label="Software::Travamento/Crash",
            weight=4.5,
            patterns=[r"\bcrash\b", r"\bfreeze\b", r"\btrav", r"\blentid", r"\bhang\b", r"\bcongel", r"\babort"],
        ),
        Rule(
            label="Software::Aplicativo/App",
            weight=3.5,
            patterns=[
                r"\baplicativo\b", r"\bapp\b", r"\bsoftware\b", r"\bsmart score\b", r"\bcortex\b",
                r"\bbrainwave\b", r"\bproview\b", r"\baw\b", r"\baws\b"
            ],
        ),
        Rule(
            label="Software::Configuração",
            weight=3.0,
            patterns=[r"\bconfig", r"\bsetup\b", r"\bidioma\b", r"\blanguage\b"],
        ),
        Rule(
            label="Software::Manual/Documentação",
            weight=3.0,
            patterns=[r"\bmanual\b", r"\bdocumenta"],
        ),
        Rule(
            label="Software::PDF/Relatório",
            weight=3.5,
            patterns=[r"\bpdf\b", r"\breporte\b", r"\brelat[óo]rio\b", r"\breport\b"],
            require_all=[r"\bpdf\b|\breport\b|\breporte\b|\brelat"],
        ),

        # -------- Hardware --------
        Rule(
            label="Hardware::Bobina/Coil",
            weight=4.5,
            patterns=[r"\bbobina\b", r"\bcoil\b", r"\bspine\b", r"\bnvarray\b", r"\bantenna\b|\bantena\b"],
        ),
        Rule(
            label="Hardware::Mesa/Table",
            weight=4.0,
            patterns=[r"\bmesa\b", r"\btable\b", r"\bsubir\b.*\bmesa\b", r"\bbajar\b.*\bmesa\b"],
        ),
        Rule(
            label="Hardware::Detector/Grid",
            weight=4.0,
            patterns=[r"\bdetector\b", r"\bgrilla\b", r"\bgrid\b", r"\bparrilla\b"],
        ),
        Rule(
            label="Hardware::Amplificador/RF",
            weight=4.0,
            patterns=[r"\brf\b", r"\bamplificador\b", r"\binterferen", r"\bnoise\b|\bru[ií]do\b"],
            negative=[r"\bnoize\b"],  # example typo control
        ),
        Rule(
            label="Hardware::Trigger/Sincronização",
            weight=3.8,
            patterns=[r"\btrigger\b", r"\bekg\b", r"\becg\b", r"\bsincron"],
        ),
        Rule(
            label="Hardware::Injetor/Contraste",
            weight=3.8,
            patterns=[r"\binyector\b", r"\binjetor\b", r"\bcontraste\b|\bcontrast\b"],
        ),
        Rule(
            label="Hardware::Teclado/Controles",
            weight=3.0,
            patterns=[r"\bteclado\b", r"\bkeyboard\b", r"\bpedal\b", r"\bboton\b|\bbot[aã]o\b"],
        ),

        # -------- Image Quality --------
        Rule(
            label="Qualidade de Imagem::Artefato",
            weight=4.0,
            patterns=[r"\bartef", r"\bartifact", r"\blineas\b|\blinhas\b", r"\bgranulos"],
        ),
        Rule(
            label="Qualidade de Imagem::Exposição",
            weight=3.5,
            patterns=[r"\bsubexpu", r"\bsobreexpu", r"\bexposi", r"\bblanca\b|\bbranca\b"],
        ),
        Rule(
            label="Qualidade de Imagem::Difusão",
            weight=3.5,
            patterns=[r"\bdifusi", r"\bdiffusion\b"],
        ),

        # -------- Installation / Missing --------
        Rule(
            label="Instalação::Peças Faltantes",
            weight=3.5,
            patterns=[r"\bfaltante\b|\bfaltando\b", r"\bmissing\b", r"\bpendiente\b|\bpend[êe]ncia\b"],
        ),
        Rule(
            label="Instalação::Instalação Incompleta",
            weight=3.2,
            patterns=[r"\bn[aã]o instalad", r"\bno instalad", r"\bnot installed\b", r"\bincomplet"],
        ),

        # -------- Operational --------
        Rule(
            label="Operacional::Treinamento",
            weight=2.5,
            patterns=[r"\btreinamento\b|\btraining\b|\bentrenamiento\b", r"\baplica[cç][aã]o\b|\baplicacion\b"],
        ),
        Rule(
            label="Operacional::Suporte FE/Engenharia",
            weight=2.2,
            patterns=[r"\bfe\b", r"\bingeniero\b|\bengenheiro\b", r"\bcase\b|\bchamado\b|\bticket\b"],
        ),
        Rule(
            label="Operacional::Disponibilidade/Agenda",
            weight=2.0,
            patterns=[r"\bcancel", r"\bdisponibil", r"\btiempo\b|\btempo\b", r"\breagend"],
        ),
        # ---- Shortage (peças/acessórios faltantes, não entregues ou não encontrados) ----
        Rule(
            label="Shortage::Acessório/Peça Faltante",
            weight=5.5,
            patterns=[
                r"\bprotetor\b.*\bauricular\b",
                r"\bgrades?\b.*\bfenestrad", r"\brejillas?\b.*\branurad",
                r"\bkit\b.*\b(i?mcompleto|incompleto|faltant)",
                r"\bgrades?\b.*\blaterais\b",
                r"\bmamotomia\b.*\b(i?mcompleto|incompleto|faltant)",
                r"\bllave\b.*\bquench\b",
                r"\bshortage\b",
                r"\bn[aã]o\b.*\b(encontrad|chegara?m)\b.*\b(grade|rejilla|acess[oó]rio|bobina|kit|llave|cable|sensor)\b",
                r"\bfalta\b.*\b(insumo|protetor|grade|rejilla|llave|sensor|cable)\b",
                r"\bfaltant\w*\b.*\b(acess[oó]rio|bobina|kit|pe[cç]a|cable|sensor)\b",
                r"\bacess[oó]rios?\b.*\bfaltant",
                r"\bbobinas?\b.*\b(shortage|n[aã]o\b.*\bchegar)",
                r"\bn[aã]o\b.*\b(conto|contou|cont[aá])\b.*\b(sensor|cable|inyector|insumo)\b",
                r"\bno\b.*\b(se\b.*\b)?(encontr|cont[oó])\b.*\b(cable|sensor|llave|accesorio)\b",
                # --- NOVOS padrões que faltam ---
                r"\bfaltando\b.*\b(bobina|flex|mama|cable|sensor|pdm)\b",
                r"\bfalta\b.*\b(de\s+)?(insumo|protetor|inyector)\b",
                r"\bno\b.*\b(se\s+)?(cuenta|conto)\b.*\b(con\s+)?(inyector|sensor|cable)\b",
                r"\bno\b.*\bdisponible\b.*\b(cable|sensor|accesorio)\b",
                r"\bcable\b.*\b(tierra|ecg)\b.*\bno\b.*\b(encontr|disponib)",
                r"\bbase\b.*\bpdm\b.*\b(cable|faltant)",
            ],
            negative=[r"\blicen[cç]", r"\bsoftware\b", r"\bmanual\b", r"\bworklist\b"],
        ),

        # ---- Problema Infraestrutura/Sistema/Conexão do Cliente ----
        Rule(
            label="Problema Infraestrutura/Sistema/Conexão do Cliente::Climatização/Chiller",
            weight=5.5,
            patterns=[
                r"\bchiller\b",
                r"\baire\s+acondicionado\b", r"\bar\s+condicionado\b",
                r"\btemperatura\b.*\b(cuarto|sala|m[aá]quina|gabinete|refrigerante)\b",
                r"\baquecimento\b.*\bgabinete\b",
                r"\btemperatura\b.*\b(alta|elev|subi|lev[ao]rse)\b",
                r"\banticongelante\b.*\bfuga\b",
                r"\bfuga\b.*\banticongelante\b",
            ],
        ),
        Rule(
            label="Problema Infraestrutura/Sistema/Conexão do Cliente::Elétrica/Disjuntor",
            weight=5.5,
            patterns=[
                r"\bdisjuntor\b",
                r"\bfalla\s+el[eé]ctrica\b",
                r"\bvolta[gj]e\b.*\b(varia|inestab|baj)",
                r"\bvaria[cç][oõ]es\b.*\bvolta",
                r"\bdesarme\b",
                r"\bups\b.*\b(no|n[aã]o)\b.*\bactiv",
            ],
        ),
        Rule(
            label="Problema Infraestrutura/Sistema/Conexão do Cliente::Infraestrutura Sala",
            weight=5.0,
            patterns=[
                r"\bobras?\b.*\bexternas?\b",
                r"\bru[ií]dos?\b.*\bobras?\b",
                r"\bporta\b.*\babre\b.*\bsozinha\b",
                r"\bsuelo\b.*\bdesnivel",
                r"\bhumedad\b",
                r"\bpared\b.*\binflad",
                r"\bsala\b.*\brepar[oa]s?\b",
                r"\bgaiola\b.*\b(problema|noise|ru[ií]do)\b",
            ],
        ),
        Rule(
            label="Problema Infraestrutura/Sistema/Conexão do Cliente::Hélio/Pressão",
            weight=5.5,
            patterns=[
                r"\bhelium\b.*\bpressure\b",
                r"\bh[eé]lio\b.*\bpress[aã]o\b",
                r"\bn[ií]vel\b.*\bh[eé]lio\b",
                r"\bhelium\b.*\bpsi\b",
                r"\bpsi\b.*\bsafety\b",
                r"\bpress[ií][oó]n\b.*\bhelio\b",
                r"\bhelio\b.*\bpsi\b",
            ],
        ),
        Rule(
            label="Problema Infraestrutura/Sistema/Conexão do Cliente::Conectividade/Chave Cliente",
            weight=4.5,
            patterns=[
                r"\bconectividad\b.*\binestabil",
                r"\bfallas?\b.*\b(en\s+la\s+)?conectividad\b",
                r"\bllave\b.*\b(del\s+equipo\b.*\b)?(no\b.*\bencontr|perdid)",
                r"\bchave\b.*\b(perdid|n[aã]o\b.*\bencontr)",
            ],
            negative=[r"\bworklist\b", r"\bpacs\b", r"\bdicom\b", r"\baw\b"],
        ),
    ]


def rule_score(text: str, compiled_rules: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Returns:
      scores[label] = float
      hits[label] = list of short explanations (matched pattern strings)
    """
    t = normalize_text(text)
    scores: Dict[str, float] = {}
    hits: Dict[str, List[str]] = {}

    for r in compiled_rules:
        label = r["label"]
        weight = r["weight"]
        pats = r["patterns"]
        reqs = r["require_all"]
        negs = r["negative"]
        block = r["block_if_negative"]

        # require_all: all must match at least once
        if reqs:
            ok = True
            for rp in reqs:
                if not rp.search(t):
                    ok = False
                    break
            if not ok:
                continue

        # negative handling
        neg_hit = False
        if negs:
            for np in negs:
                if np.search(t):
                    neg_hit = True
                    break
        if neg_hit and block:
            continue

        # count matches from patterns
        local_hits = []
        match_count = 0
        for p in pats:
            m = p.search(t)
            if m:
                match_count += 1
                local_hits.append(p.pattern)

        if match_count == 0:
            continue

        # scoring: weight * (1 + 0.35*(match_count-1)) to reward multiple signals
        s = weight * (1.0 + 0.35 * (match_count - 1))
        scores[label] = scores.get(label, 0.0) + s
        hits.setdefault(label, []).extend(local_hits)

    return scores, hits


def softmax_confidence(best: float, second: float) -> float:
    """
    Convert margin into a smooth 0-1 confidence.
    """
    margin = max(0.0, best - second)
    # tune slope as needed
    return float(1.0 / (1.0 + math.exp(-0.9 * (margin - 1.5))))


def pick_label_from_scores(scores: Dict[str, float]) -> Tuple[str, float, List[Tuple[str, float]]]:
    if not scores:
        return "Outros::Não Categorizado", 0.0, []
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    conf = softmax_confidence(best_score, second_score)
    top_alts = sorted_items[:5]
    return best_label, conf, top_alts


def split_group_category(label: str) -> Tuple[str, str]:
    if "::" not in label:
        return "Outros", "Não Categorizado"
    g, c = label.split("::", 1)
    return g, c


# ---------------------------
# ML Model (optional, improves with labels / bootstrapping)
# ---------------------------

@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    clf: CalibratedClassifierCV
    labels: List[str]           # class labels in clf order
    meta: Dict[str, Any]        # training metadata


def train_text_model(texts: List[str], y: List[str], random_state: int = 42) -> ModelBundle:
    """
    Trains TF-IDF + LinearSVC (calibrated) for probability/confidence.
    """
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )

    X = vec.fit_transform(texts)

    # LinearSVC has no proba; we calibrate it.
    base = LinearSVC()
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    clf.fit(X, y)

    labels = list(getattr(clf, "classes_", []))
    meta = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(texts),
        "n_classes": len(labels),
    }
    return ModelBundle(vectorizer=vec, clf=clf, labels=labels, meta=meta)


def predict_text_model(bundle: ModelBundle, texts: List[str]) -> Tuple[List[str], List[float], List[List[Tuple[str, float]]]]:
    X = bundle.vectorizer.transform(texts)
    proba = bundle.clf.predict_proba(X)  # shape: (n, n_classes)

    preds = []
    confs = []
    top_alts = []

    for i in range(proba.shape[0]):
        row = proba[i]
        idx_sorted = row.argsort()[::-1]
        best_idx = int(idx_sorted[0])
        best_label = bundle.labels[best_idx]
        best_conf = float(row[best_idx])

        alts = []
        for j in idx_sorted[:5]:
            alts.append((bundle.labels[int(j)], float(row[int(j)])))

        preds.append(best_label)
        confs.append(best_conf)
        top_alts.append(alts)

    return preds, confs, top_alts


# ---------------------------
# Ensemble + decision policy
# ---------------------------

def ensemble_decision(
    rule_label: str,
    rule_conf: float,
    rule_alts: List[Tuple[str, float]],
    model_label: Optional[str],
    model_conf: Optional[float],
    model_alts: Optional[List[Tuple[str, float]]],
    alpha_rules: float = 0.55,
) -> Tuple[str, float, str, List[Tuple[str, float]]]:
    """
    Combine rule + model.
    alpha_rules closer to 1 => trust rules more.
    If model missing => rules only.
    """
    if model_label is None or model_conf is None or model_alts is None:
        return rule_label, rule_conf, "rules", rule_alts

    # convert rule_alts scores into pseudo probabilities by softmax-like normalization
    # (only among top alts we have)
    rule_scores = {lbl: sc for lbl, sc in rule_alts}
    if rule_scores:
        max_sc = max(rule_scores.values())
        exp = {k: math.exp((v - max_sc) / 2.0) for k, v in rule_scores.items()}  # temperature
        z = sum(exp.values()) or 1.0
        rule_prob = {k: v / z for k, v in exp.items()}
    else:
        rule_prob = {}

    model_prob = {lbl: p for lbl, p in model_alts}

    # union of candidates
    candidates = set(rule_prob.keys()) | set(model_prob.keys())
    if not candidates:
        return "Outros::Não Categorizado", 0.0, "ensemble", []

    combined = {}
    for lbl in candidates:
        rp = rule_prob.get(lbl, 0.0)
        mp = model_prob.get(lbl, 0.0)
        combined[lbl] = alpha_rules * rp + (1.0 - alpha_rules) * mp

    sorted_comb = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_lbl, best_p = sorted_comb[0]
    second_p = sorted_comb[1][1] if len(sorted_comb) > 1 else 0.0
    # confidence: combine with agreement boost
    agree_boost = 0.10 if best_lbl == rule_label == model_label else 0.0
    conf = float(min(1.0, max(0.0, (best_p - second_p) * 1.6 + agree_boost)))

    return best_lbl, conf, "ensemble", sorted_comb[:5]


def needs_review(confidence: float, text: str) -> bool:
    """
    Policy: if confidence low OR text empty/too short -> review.
    Tune thresholds as you learn.
    """
    t = normalize_text(text)
    if len(t) < 12:
        return True
    return confidence < 0.62


# ---------------------------
# Bootstrap pseudo-labels (weak supervision)
# ---------------------------

def bootstrap_pseudo_labels(
    df: pd.DataFrame,
    text_col: str,
    compiled_rules: List[Dict[str, Any]],
    min_conf: float = 0.72,
    min_score: float = 3.2,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Use rules to pseudo-label high-confidence rows.
    Returns (texts, labels, df_with_bootstrap_cols).
    """
    texts: List[str] = []
    labels: List[str] = []

    rule_labels = []
    rule_confs = []
    rule_best_scores = []

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        scores, _hits = rule_score(text, compiled_rules)
        label, conf, top_alts = pick_label_from_scores(scores)
        best_score = top_alts[0][1] if top_alts else 0.0

        rule_labels.append(label)
        rule_confs.append(conf)
        rule_best_scores.append(best_score)

        if conf >= min_conf and best_score >= min_score and label != "Outros::Não Categorizado":
            texts.append(normalize_text(text))
            labels.append(label)

    out = df.copy()
    out["Bootstrap_Rule_Label"] = rule_labels
    out["Bootstrap_Rule_Conf"] = rule_confs
    out["Bootstrap_Rule_Score"] = rule_best_scores

    return texts, labels, out


# ---------------------------
# Main classification pipeline
# ---------------------------

def classify_dataframe(
    df: pd.DataFrame,
    model_bundle: Optional[ModelBundle] = None,
    alpha_rules: float = 0.55,
) -> pd.DataFrame:
    df = df.copy()

    # Padronização de Países 
    if "Customer Country" in df.columns: df["Customer Country"] = df["Customer Country"].apply(normalize_country) 

    text_col = find_problem_description_column(df)
    modality_col = "Modality" if "Modality" in df.columns else None

    rules = build_rules()
    compiled_rules = [r.compile() for r in rules]

    texts_norm = [normalize_text(x) for x in df[text_col].tolist()]

    # Rule predictions (always)
    rule_labels = []
    rule_confs = []
    rule_hits = []
    rule_topalts = []

    for raw_text in df[text_col].tolist():
        scores, hits = rule_score(raw_text, compiled_rules)
        label, conf, top_alts = pick_label_from_scores(scores)
        rule_labels.append(label)
        rule_confs.append(conf)
        # store concise hits for winning label
        h = hits.get(label, [])
        # keep unique patterns, up to 8
        h_uniq = []
        for p in h:
            if p not in h_uniq:
                h_uniq.append(p)
        rule_hits.append("; ".join(h_uniq[:8]))
        rule_topalts.append(top_alts)

    # Model predictions (optional)
    model_labels = None
    model_confs = None
    model_topalts = None
    if model_bundle is not None:
        model_labels, model_confs, topalts = predict_text_model(model_bundle, texts_norm)
        # convert list of list tuples into list of list tuples (already)
        model_topalts = topalts

    # Ensemble
    final_labels = []
    final_confs = []
    final_sources = []
    final_topalts = []

    for i in range(len(df)):
        rl = rule_labels[i]
        rc = rule_confs[i]
        ralts = rule_topalts[i]

        ml = model_labels[i] if model_labels is not None else None
        mc = model_confs[i] if model_confs is not None else None
        malts = model_topalts[i] if model_topalts is not None else None

        lbl, conf, source, topalts = ensemble_decision(
            rule_label=rl,
            rule_conf=rc,
            rule_alts=ralts,
            model_label=ml,
            model_conf=mc,
            model_alts=malts,
            alpha_rules=alpha_rules,
        )
        final_labels.append(lbl)
        final_confs.append(conf)
        final_sources.append(source)
        final_topalts.append(topalts)

    # Build output columns
    groups, cats = zip(*(split_group_category(l) for l in final_labels))
    df["Problem_Group"] = list(groups)
    df["Problem_Category"] = list(cats)

    df["Confidence"] = [round(float(c), 4) for c in final_confs]
    df["Source"] = final_sources
    df["Needs_Review"] = ["Yes" if needs_review(final_confs[i], df[text_col].iloc[i]) else "No" for i in range(len(df))]

    # Explainability columns
    df["Keyword_Hits"] = rule_hits  # (from rules) even if ensemble chose model
    df["Top_Alternatives"] = [
        " | ".join([f"{lbl}:{round(score, 3)}" for lbl, score in alts])
        for alts in final_topalts
    ]

    # Extra helpful columns
    df["Language"] = [guess_language(x) for x in df[text_col].tolist()]
    df["Severity"] = [determine_severity(x) for x in df[text_col].tolist()]
    df["Component"] = [
        extract_component(df[text_col].iloc[i], df[modality_col].iloc[i] if modality_col else None)
        for i in range(len(df))
    ]

    return df


# ---------------------------
# Persistence
# ---------------------------

def save_model(path: str, bundle: ModelBundle, extra: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "vectorizer": bundle.vectorizer,
        "clf": bundle.clf,
        "labels": bundle.labels,
        "meta": bundle.meta,
        "extra": extra or {},
    }
    dump(payload, path)


def load_model(path: str) -> ModelBundle:
    payload = load(path)
    return ModelBundle(
        vectorizer=payload["vectorizer"],
        clf=payload["clf"],
        labels=payload["labels"],
        meta=payload.get("meta", {}),
    )


# ---------------------------
# Commands
# ---------------------------

def cmd_bootstrap(args: argparse.Namespace) -> None:
    df, hdr = load_excel_with_best_header(args.input, args.header_row)
    df = df.dropna(how="all")
    if not args.keep_unnamed:
        df = drop_unnamed_columns(df)

    text_col = find_problem_description_column(df)

    compiled_rules = [r.compile() for r in build_rules()]
    texts, y, df_boot = bootstrap_pseudo_labels(
        df=df,
        text_col=text_col,
        compiled_rules=compiled_rules,
        min_conf=args.min_conf,
        min_score=args.min_score,
    )

    if len(texts) < args.min_samples:
        raise RuntimeError(
            f"Bootstrap gerou poucos pseudo-labels: {len(texts)} (< {args.min_samples}). "
            "Reduza --min-conf/--min-score ou adicione mais regras."
        )

    # Train
    bundle = train_text_model(texts, y)
    bundle.meta.update({
        "mode": "bootstrap",
        "input_file": os.path.basename(args.input),
        "header_row_used": hdr,
        "bootstrap_min_conf": args.min_conf,
        "bootstrap_min_score": args.min_score,
        "min_samples": args.min_samples,
    })

    save_model(args.model_out, bundle, extra={"bootstrap_preview": df_boot.head(5).to_dict(orient="records")})
    print(f"✅ Model bootstrapped and saved to: {args.model_out}")
    print(f"Pseudo-labeled samples: {len(texts)} | Classes: {len(bundle.labels)}")


def cmd_train(args: argparse.Namespace) -> None:
    df = pd.read_excel(args.labeled, engine="openpyxl").dropna(how="all")
    if not args.keep_unnamed:
        df = drop_unnamed_columns(df)

    text_col = find_problem_description_column(df)

    if "Problem_Group" not in df.columns or "Problem_Category" not in df.columns:
        raise ValueError("Arquivo rotulado precisa ter colunas 'Problem_Group' e 'Problem_Category'.")

    # Use only rows that have labels
    dfl = df.dropna(subset=["Problem_Group", "Problem_Category"]).copy()
    dfl["__label"] = dfl["Problem_Group"].astype(str).str.strip() + "::" + dfl["Problem_Category"].astype(str).str.strip()

    texts = [normalize_text(x) for x in dfl[text_col].tolist()]
    y = dfl["__label"].tolist()

    if len(texts) < args.min_samples:
        raise RuntimeError(f"Poucos exemplos rotulados: {len(texts)} (< {args.min_samples}).")

    bundle = train_text_model(texts, y)
    bundle.meta.update({
        "mode": "human_labels",
        "labeled_file": os.path.basename(args.labeled),
    })

    save_model(args.model_out, bundle)
    print(f"✅ Model trained and saved to: {args.model_out}")
    print(f"Labeled samples: {len(texts)} | Classes: {len(bundle.labels)}")


def cmd_predict(args: argparse.Namespace) -> None:
    df, hdr = load_excel_with_best_header(args.input, args.header_row)
    df = df.dropna(how="all")
    if not args.keep_unnamed:
        df = drop_unnamed_columns(df)

    model_bundle = load_model(args.model) if args.model else None
    out = classify_dataframe(df, model_bundle=model_bundle, alpha_rules=args.alpha_rules)

    out.to_excel(args.output, index=False, engine="openpyxl")

    # quick summary
    print(f"✅ Output saved: {args.output}")
    print(f"Header used (Excel row): {hdr}")
    print(f"Rows: {len(out)}")
    print("\nTop Problem_Group:")
    print(out["Problem_Group"].value_counts().head(10).to_string())
    print("\nNeeds_Review:")
    print(out["Needs_Review"].value_counts().to_string())


# ---------------------------
# CLI
# ---------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid intelligent classifier for issue Excel files")
    sub = p.add_subparsers(dest="cmd", required=True)

    # bootstrap
    b = sub.add_parser("bootstrap", help="Bootstrap model using rule-based pseudo-labels")
    b.add_argument("--input", required=True, help="Input Excel (.xlsx)")
    b.add_argument("--model-out", required=True, help="Output model file (.joblib)")
    b.add_argument("--header-row", type=int, default=None, help="Header row (1-based). If omitted, auto-detect.")
    b.add_argument("--keep-unnamed", action="store_true", help="Keep 'Unnamed' columns")
    b.add_argument("--min-conf", type=float, default=0.72, help="Min rule confidence to accept pseudo-label")
    b.add_argument("--min-score", type=float, default=3.2, help="Min rule raw score to accept pseudo-label")
    b.add_argument("--min-samples", type=int, default=120, help="Minimum pseudo-labeled samples required")
    b.set_defaults(func=cmd_bootstrap)

    # train
    t = sub.add_parser("train", help="Train model from human-labeled Excel")
    t.add_argument("--labeled", required=True, help="Labeled Excel containing Problem_Group and Problem_Category")
    t.add_argument("--model-out", required=True, help="Output model file (.joblib)")
    t.add_argument("--keep-unnamed", action="store_true", help="Keep 'Unnamed' columns")
    t.add_argument("--min-samples", type=int, default=120, help="Minimum labeled samples required")
    t.set_defaults(func=cmd_train)

    # predict
    r = sub.add_parser("predict", help="Classify an Excel and generate a new Excel with extra columns")
    r.add_argument("--input", required=True, help="Input Excel (.xlsx)")
    r.add_argument("--output", required=True, help="Output Excel (.xlsx)")
    r.add_argument("--model", default=None, help="Optional model (.joblib). If omitted => rules only")
    r.add_argument("--header-row", type=int, default=None, help="Header row (1-based). If omitted, auto-detect.")
    r.add_argument("--keep-unnamed", action="store_true", help="Keep 'Unnamed' columns")
    r.add_argument("--alpha-rules", type=float, default=0.55, help="Ensemble weight for rules (0-1)")
    r.set_defaults(func=cmd_predict)

    return p


def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()