"""Correspondance automatique des colonnes d'une base inconnue vers le schéma interne.

Usage type, le jour 1 :

    raw = pd.read_csv(...)  ou  résultat d'une requête SQL
    proposal = propose_schema_map(raw)
    print(proposal["report"])           # à relire : colonne choisie, score, alternatives
    write_schema_map(proposal, "config/schema_map.yaml")

Deux indices sont combinés pour chaque colonne :
- son **nom** (synonymes français et anglais, après normalisation) ;
- son **contenu** (une notation se reconnaît, une date se lit, un spread est un
  nombre positif de l'ordre de la centaine…).

Le résultat est une proposition à relire, jamais une vérité : le rapport donne
pour chaque champ la colonne retenue, la confiance et les autres candidates.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bondalorians.data.schema import to_notch

# Champ interne -> synonymes (après normalisation : minuscules, sans accents ni ponctuation).
SYNONYMS: dict[str, list[str]] = {
    "isin": ["isin", "isincode", "codeisin", "id", "bondid", "securityid", "cusip", "idbb", "figi"],
    # Le ticker Bloomberg identifie l'émetteur ; une colonne « nom émetteur » contient parfois le nom de
    # l'obligation (« RZR 1 ⅜ 11/02/36 ») : le contenu tranche (voir content_score).
    "issuer": ["issuer", "issuername", "emetteur", "nomemetteur", "issuerid", "company", "obligor", "name", "entity",
               "ticker", "issuerticker", "tickeremetteur"],
    "sector": ["sector", "secteur", "nomsecteur", "gicssector", "bicssector", "industrysector", "bicslevel1"],
    "industry": ["industry", "industrie", "industrieemetteur", "industrygroup", "nomgroupeindustrie", "bicslevel2",
                 "subindustry", "sousindustrie"],
    "sector_family": ["classificationsecteuremetteurniv1", "bclass1", "bclasslevel1", "assetclass", "famille",
                      "classe"],
    "structure": ["structureaffaire", "structure", "holdcoopco", "capitalstructure", "businessstructure"],
    "pricing_date": ["pricingdate", "datepricing", "asofdate", "datevalo", "datevalorisation", "date", "dateprix",
                     "valuationdate"],
    "country": ["country", "pays", "countryofrisk", "countryofdomicile", "domicile", "risk_country"],
    "currency": ["currency", "devise", "ccy", "crncy", "monnaie"],
    "rating": ["rating", "notation", "compositerating", "creditrating", "bbgcomposite", "ratingsp", "sprating",
               "moodysrating", "fitchrating", "note", "avgrating", "rtg"],
    "seniority": ["seniority", "seniorite", "rank", "paymentrank", "ranking", "rang", "debttype", "capitalrank"],
    "coupon_type": ["coupontype", "typecoupon", "cpntype", "coupon_type", "rateType", "fixedfloat", "typetaux"],
    "callable": ["callable", "iscallable", "remboursable", "callflag", "call", "optionality", "calltype",
                 "typematurit", "maturitytype", "mtytyp"],
    "green": ["greenbond", "green", "vert", "obligationverte", "greenflag"],
    "social": ["socialbond", "social"],
    "bail_in": ["bondbailin", "bailin", "bailinbond"],
    # Échéance de calcul (« workout », date de call si l'obligation est cotée au call) : c'est celle du Z-Spread.
    "maturity_date": ["maturity", "maturitydate", "echeance", "dateecheance", "dateofmaturity", "maturitydt",
                      "redemptiondate", "datematuritpricingobligation", "workoutdate", "maturitepricing"],
    "final_maturity_date": ["datematurit", "finalmaturity", "finalmaturitydate", "echeancefinale", "legalmaturity"],
    "call_date": ["prochainedatecall", "nextcalldate", "nextcall", "calldate", "dateprochaincall", "firstcalldate"],
    "maturity_years": ["maturityyears", "yearstomaturity", "ttm", "timetomaturity", "residualmaturity",
                       "maturiteresiduelle", "dureeresiduelle", "maturityinyears"],
    "amount_outstanding": ["amountoutstanding", "outstanding", "amtoutstanding", "encours", "notional",
                           "issuesize", "size", "taille", "montant", "facevalue", "amountissued"],
    "z_spread": ["zspread", "z_spread", "zsprd", "spreadz", "zspreadbp", "zspd", "spread", "oas"],
    # Colonnes supplémentaires du jour 1.
    "esg": ["esg", "esgscore", "esgrating", "noteesg", "msciesg", "esgrisk", "sustainalytics", "scoreesg"],
    "issue_date": ["issuedate", "dateemission", "dateofissue", "issue", "emission", "firstsettle", "issuedt",
                   "settlementdate"],
    "duration": ["duration", "durationbid", "modifiedduration", "modduration", "durationmodifiee", "effduration",
                 "effectiveduration", "spreadduration", "dur"],
    "price": ["price", "prix", "cleanprice", "dirtyprice", "px", "pxlast", "midprice", "cours", "prixbid", "pxbid"],
    "ytm": ["ytm", "yield", "yieldtomaturity", "rendement", "taux", "yld", "yieldtoworst", "ytw"],
    "discount_margin": ["discountmargenextcall", "discountmargin", "dm", "margeactualisee"],
}
# Format d'export Bloomberg connu (base du hackathon) : correspondance fixe, plus sûre que la devinette.
PRESETS: dict[str, dict[str, str]] = {
    "bloomberg_fr": {
        "pricing_date": "pricing_date", "isin": "id", "issuer": "ticker", "sector": "nomsecteur",
        "industry": "industrieemetteur", "sector_family": "classificationsecteuremetteurniv1",
        "structure": "structureaffaire", "country": "paysemetteur", "currency": "devise", "rating": "ratingmoyen",
        "seniority": "rangseniorit_obligation", "coupon_type": "typecoupon", "callable": "typematurit",
        "maturity_date": "datematurit_pricingobligation", "final_maturity_date": "datematurit",
        "call_date": "prochainedatecall", "issue_date": "dateemission", "z_spread": "niveauzspread",
        "green": "greenbond", "social": "socialbond", "bail_in": "bondbailin", "duration": "durationbid",
        "price": "prixbid", "ytm": "rendement", "discount_margin": "discountmargenextcall",
    },
}
REQUIRED = ["issuer", "rating", "seniority", "currency", "z_spread"]
# Le prix et le rendement sont liés mécaniquement au spread : contrôles seulement, jamais variables.
CONTROL_ONLY = ["price", "ytm", "discount_margin"]

SENIORITY_RULES = [  # (motif, valeur interne) — testés dans l'ordre
    (r"\bat\s*1\b|additional\s*tier|addtl|\bat1\b|coco|contingent", "AT1"),
    (r"tier\s*2|\bt2\b|lower\s*tier|\blt2\b", "TIER2"),
    (r"non[\s-]*pref|\bsnp\b|senior\s*non|holdco", "SENIOR_NON_PREFERRED"),
    (r"hybrid|hybride|perp|corporate\s*hybrid", "HYBRID"),
    (r"sub|subord|junior|\bjr\b|\bsub\b", "SUBORDINATED"),
    # « unsecured » contient « secured » : la dette non garantie doit être testée avant la garantie.
    (r"unsecured|non[\s-]*garanti|chirograph", "SENIOR_UNSECURED"),
    (r"preferred|pref|\bsp\b|opco", "SENIOR_PREFERRED"),
    (r"secured|garanti|covered|\bsec\b|1st\s*lien|first\s*lien|collat", "SECURED"),
    (r"senior|\bsr\b", "SENIOR_UNSECURED"),
]
COUPON_RULES = [
    (r"float|frn|variable|var|flottant|libor|euribor|sofr|estr", "FLOATING"),
    (r"zero|zéro|\bzc\b|discount", "ZERO"),
    (r"fix|fixe|fixed|step|bullet", "FIXED"),
]
BOND_NAME = r"[\d⅛¼⅜½⅝¾⅞]\S*\s+\d{1,2}/\d{1,2}/\d{2,4}|\bPERP\b"
MSCI_ESG = {"AAA": 7, "AA": 6, "A": 5, "BBB": 4, "BB": 3, "B": 2, "CCC": 1}


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


# ----------------------------------------------------------- indices de contenu
def _numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return pd.Series(pd.NaT, index=series.index)
    from bondalorians.data.repository import _to_dates

    return _to_dates(series)


def content_score(field: str, series: pd.Series) -> float:
    """Entre 0 et 1 : à quel point le contenu ressemble à ce champ."""
    values = series.dropna()
    if values.empty:
        return 0.0
    sample = values.sample(min(500, len(values)), random_state=0)
    numeric = _numeric(sample)
    share_numeric = numeric.notna().mean()
    distinct = sample.astype(str).nunique() / len(sample)
    if field == "rating":
        return float(sample.map(to_notch).notna().mean())
    if field == "esg":
        letters = sample.astype(str).str.upper().str.strip().isin(MSCI_ESG).mean()
        scores = numeric.between(0, 100).mean() if share_numeric > 0.9 else 0.0
        return float(max(letters, scores * 0.8))
    if field in ("green", "social"):
        text = sample.astype(str).str.strip().str.lower()
        return float(text.isin(["true", "false", "1", "0", "y", "n", "yes", "no", "oui", "non"]).mean())
    if field == "pricing_date":
        parsed = _dates(sample.astype(str))
        return float(parsed.notna().mean() * (series.nunique() <= 30))
    if field in ("maturity_date", "issue_date", "final_maturity_date", "call_date"):
        parsed = _dates(sample)
        return float(parsed.notna().mean())
    if field == "z_spread":
        return float(numeric.between(-50, 5000).mean() * (numeric.median() > 5) * (share_numeric > 0.9))
    if field == "maturity_years":
        return float(numeric.between(0, 100).mean() * (share_numeric > 0.9) * (distinct > 0.05))
    if field == "duration":
        return float(numeric.between(0, 40).mean() * (share_numeric > 0.9) * (distinct > 0.05))
    if field == "price":
        return float(numeric.between(20, 200).mean() * (share_numeric > 0.9) * (70 < numeric.median() < 130))
    if field == "ytm":
        in_pct = numeric.between(-2, 30).mean()
        return float(in_pct * (share_numeric > 0.9))
    if field == "amount_outstanding":
        return float((numeric > 0).mean() * (share_numeric > 0.9) * (numeric.median() > 1))
    if field == "currency":
        codes = sample.astype(str).str.strip().str.upper().str.fullmatch(r"[A-Z]{3}").mean()
        return float(codes * (series.nunique() < 40))
    if field == "callable":
        text = sample.astype(str).str.strip().str.lower()
        booleans = text.isin(["true", "false", "1", "0", "y", "n", "yes", "no", "oui", "non", "o"]).mean()
        types = text.str.contains(r"call|at maturity|bullet|perp|sink|put|normal").mean()
        return float(max(booleans, types))
    if field == "seniority":
        text = sample.astype(str).str.lower()
        hits = text.apply(lambda v: any(re.search(p, v) for p, _ in SENIORITY_RULES)).mean()
        return float(hits * (series.nunique() < 50))
    if field == "coupon_type":
        text = sample.astype(str).str.lower()
        hits = text.apply(lambda v: any(re.search(p, v) for p, _ in COUPON_RULES)).mean()
        return float(hits * (series.nunique() < 20))
    if field == "isin":
        isin_like = sample.astype(str).str.strip().str.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}\d").mean()
        return float(max(isin_like, 0.5 * (distinct > 0.98)))
    if field in ("industry", "sector_family", "structure", "bail_in"):
        text_like = 1 - share_numeric
        return float(text_like * (series.nunique() <= 200))
    if field == "discount_margin":
        return float(numeric.notna().mean() > 0) * 0.5
    if field in ("issuer", "sector", "country"):
        text_like = 1 - share_numeric
        cardinality = series.nunique() / max(len(series), 1)
        if field == "issuer":
            # « RZR 1 ⅜ 11/02/36 » : un nom d'obligation (coupon + échéance), pas un émetteur.
            bond_like = sample.astype(str).str.contains(BOND_NAME, regex=True).mean()
            plausible = cardinality < 0.9 or len(series) < 50  # petit extrait : un émetteur par ligne possible
            return float(text_like * (0.001 < cardinality) * plausible * (1 - bond_like))
        if field == "sector":
            return float(text_like * (series.nunique() <= 60))
        return float(text_like * (series.nunique() <= 120))
    return 0.0


def name_score(field: str, column: str) -> float:
    norm = normalize(column)
    synonyms = [normalize(s) for s in SYNONYMS[field]]
    if norm in synonyms:
        return 1.0
    if any(s and len(s) >= 4 and (s in norm) for s in synonyms):
        # « paysemetteur », « industrieemetteur » : le début du nom désigne un autre champ (pays de l'émetteur).
        others = [normalize(o) for f, syn in SYNONYMS.items() if f != field for o in syn]
        if any(len(o) >= 4 and norm.startswith(o) and o not in synonyms for o in others):
            return 0.3
        return 0.7
    if any(norm and len(norm) >= 4 and norm in s for s in synonyms):
        return 0.5
    return 0.0


def detect_preset(raw: pd.DataFrame, share: float = 0.8) -> str | None:
    """Nom du format connu si au moins `share` de ses colonnes sont présentes."""
    present = {str(c).strip().lower() for c in raw.columns}
    for name, mapping in PRESETS.items():
        if sum(c in present for c in mapping.values()) >= share * len(mapping):
            return name
    return None


def propose_schema_map(raw: pd.DataFrame, as_of: str | None = None) -> dict:
    """Propose une correspondance champ interne -> colonne réelle, avec un rapport."""
    candidates = []
    for field in SYNONYMS:
        for column in raw.columns:
            n = name_score(field, column)
            if n == 0:
                continue
            c = content_score(field, raw[column])
            candidates.append({"champ": field, "colonne": column, "nom": n, "contenu": round(c, 2),
                               "score": round(0.55 * n + 0.45 * c, 3)})
    table = pd.DataFrame(candidates, columns=["champ", "colonne", "nom", "contenu", "score"])

    # Format connu : correspondance fixe ; le reste est deviné comme d'habitude.
    chosen: dict[str, str] = {}
    preset = detect_preset(raw)
    if preset:
        chosen = {f: c for f, c in PRESETS[preset].items() if c in raw.columns}
    used: set[str] = set(chosen.values())
    # Affectation gloutonne : meilleurs couples d'abord, une colonne par champ.
    for row in table.sort_values("score", ascending=False).itertuples():
        if row.champ in chosen or row.colonne in used or row.contenu < 0.3:
            continue
        # Une colonne « ESG » ne doit pas être prise pour la notation de crédit, ni l'inverse.
        if row.champ == "rating" and "esg" in normalize(row.colonne):
            continue
        chosen[row.champ] = row.colonne
        used.add(row.colonne)

    if "maturity_years" in chosen and "maturity_date" in chosen:
        chosen.pop("maturity_date")  # la maturité en années suffit

    report = []
    for field in SYNONYMS:
        alternatives = table[(table["champ"] == field) & (table["colonne"] != chosen.get(field))]
        alternatives = alternatives.sort_values("score", ascending=False).head(3)
        best = table[(table["champ"] == field) & (table["colonne"] == chosen.get(field))]
        confidence = float(best["score"].iloc[0]) if len(best) else 0.0
        if preset and field in chosen and chosen[field] == PRESETS[preset].get(field):
            confidence = 1.0
        report.append({
            "champ": field, "colonne": chosen.get(field),
            "confiance": "élevée" if confidence >= 0.8 else "moyenne" if confidence >= 0.55 else
            ("faible" if field in chosen else "—"),
            "score": round(confidence, 2),
            "autres_candidates": ", ".join(f"{a.colonne} ({a.score:.2f})" for a in alternatives.itertuples()),
            "role": ("contrôle seulement (lié mécaniquement au spread)" if field in CONTROL_ONLY
                     else "fixé par le format " + preset if preset and field in PRESETS[preset]
                     else "obligatoire" if field in REQUIRED else "facultatif"),
        })
    report = pd.DataFrame(report)

    values: dict[str, dict] = {}
    if "seniority" in chosen:
        values["seniority"] = _value_map(raw[chosen["seniority"]], SENIORITY_RULES, "SENIOR_UNSECURED")
    if "coupon_type" in chosen:
        values["coupon_type"] = _value_map(raw[chosen["coupon_type"]], COUPON_RULES, "FIXED")

    unmapped = [c for c in raw.columns if c not in used]
    problems = []
    for field in REQUIRED:
        if field not in chosen:
            problems.append(f"Champ obligatoire introuvable : {field}. Renseignez-le à la main dans le fichier.")
    if "maturity_years" not in chosen and "maturity_date" not in chosen:
        problems.append("Ni maturité en années ni date d'échéance trouvée : renseignez l'une des deux.")
    if "rating" in chosen:
        recognized = raw[chosen["rating"]].dropna().map(to_notch).notna().mean()
        if recognized < 0.95:
            unknown = raw[chosen["rating"]].dropna().astype(str)
            unknown = unknown[unknown.map(to_notch).isna()].value_counts().head(8)
            problems.append(f"Notations reconnues : {recognized:.0%}. Non reconnues : "
                            + ", ".join(f"{k} ({v})" for k, v in unknown.items())
                            + ". Ces lignes seront écartées si rien n'est fait.")
    if "issuer" in chosen:
        bond_like = raw[chosen["issuer"]].dropna().astype(str).str.contains(BOND_NAME, regex=True).mean()
        if bond_like > 0.5:
            problems.append(f"La colonne émetteur « {chosen['issuer']} » ressemble à des noms d'obligations "
                            f"({bond_like:.0%}) : chaque obligation deviendrait son propre émetteur. Prenez le ticker.")
    if "pricing_date" in chosen:
        from bondalorians.data.repository import _to_dates

        counts = _to_dates(raw[chosen["pricing_date"]].astype(str)).value_counts().sort_index()
        if len(counts) > 1:
            problems.append("Plusieurs dates de prix : " + ", ".join(f"{d.date()} ({n} lignes)" for d, n in counts.items())
                            + ". Une seule date sert à estimer (la plus récente par défaut) ; l'autre sert à mesurer "
                            "le mouvement du marché.")
    schema_map = {
        "source": {"query": "SELECT * FROM <table>", "as_of": as_of, "format": preset},
        "columns": {field: chosen.get(field) for field in SYNONYMS},
        "values": values,
    }
    return {"schema_map": schema_map, "report": report, "candidates": table, "unmapped": unmapped,
            "problems": problems}


def _value_map(series: pd.Series, rules, default: str) -> dict:
    mapping = {}
    for value in series.dropna().astype(str).unique():
        text = value.lower()
        mapping[value] = next((target for pattern, target in rules if re.search(pattern, text)), default)
    return mapping


def write_schema_map(proposal: dict, path: str | Path, query: str | None = None) -> Path:
    """Écrit la proposition dans `path` (l'ancien fichier est gardé en .bak)."""
    path = Path(path)
    schema_map = proposal["schema_map"]
    if query:
        schema_map["source"]["query"] = query
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    header = (
        "# Correspondance proposée automatiquement (scripts/jour1.py) : À RELIRE.\n"
        "# columns : champ interne -> colonne réelle (null = absent).\n"
        "# values : traduction des valeurs réelles vers les valeurs internes.\n"
        "# price et ytm servent aux contrôles seulement : jamais de variables du modèle.\n\n"
    )
    path.write_text(header + yaml.safe_dump(schema_map, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
