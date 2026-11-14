"""Clustering : regrouper les obligations qui se ressemblent, pour lire l'échantillon.

Le clustering ne sert pas à estimer le spread. Il répond à deux questions :
- quelles « familles » d'obligations l'échantillon contient-il ?
- ces familles expliquent-elles les spreads mieux que la notation seule ?

Quatre méthodes, sur les mêmes variables normalisées :
- k-means : des familles rondes, de tailles comparables ;
- Ward (hiérarchique) : regroupe pas à pas les familles les plus proches ;
- mélange gaussien : familles de formes variées, appartenance probabiliste ;
- HDBSCAN : familles de densité, et repère les obligations atypiques.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bondalorians.analysis.common import fmt, notch_label, pct, r2, rating_bucket, real_bonds

METHODS = {
    "kmeans": "k-means",
    "ward": "Hiérarchique (Ward)",
    "gmm": "Mélange gaussien",
    "hdbscan": "HDBSCAN (densité)",
}
METHOD_HELP = {
    "kmeans": "Cherche k centres et rattache chaque obligation au plus proche. Familles compactes, de tailles voisines.",
    "ward": "Part de chaque obligation seule et fusionne pas à pas les deux familles dont la réunion "
            "disperse le moins. Donne un arbre de familles.",
    "gmm": "Suppose que chaque famille est une « cloche » de forme libre ; une obligation peut appartenir "
           "un peu à plusieurs familles.",
    "hdbscan": "Cherche les zones où les obligations sont denses ; celles qui ne tombent dans aucune zone "
               "sont classées atypiques. Choisit seule le nombre de familles.",
}
NUMERIC = {"notch": 1.0, "log_t": 1.0, "sen_rank": 1.0, "log_amt": 0.5}
CATEGORICAL = ["sector", "currency"]
ATYPICAL = "Atypique"


class ClusterEncoder:
    """Variables centrées-réduites ; catégories en indicatrices pondérées.

    Une catégorie différente compte comme un écart-type sur une variable
    numérique (poids 1/racine(2) par indicatrice).
    """

    def __init__(self, include_spread: bool = False):
        self.include_spread = include_spread

    def fit(self, df: pd.DataFrame) -> "ClusterEncoder":
        self.numeric_ = dict(NUMERIC)
        if self.include_spread:
            self.numeric_["log_spread"] = 1.5
        frame = self._numeric_frame(df)
        self.mean_, self.std_ = frame.mean(), frame.std().replace(0, 1.0)
        self.levels_ = {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}
        self.groups_ = list(self.numeric_)
        for column in CATEGORICAL:
            self.groups_ += [column] * len(self.levels_[column])
        return self

    def _numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df[[c for c in NUMERIC]].astype(float).copy()
        if self.include_spread:
            frame["log_spread"] = np.log(df["z_spread"].astype(float))
        return frame

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frame = (self._numeric_frame(df) - self.mean_) / self.std_
        blocks = [frame[c].to_numpy()[:, None] * w for c, w in self.numeric_.items()]
        for column in CATEGORICAL:
            values = df[column].astype(str).to_numpy()
            blocks.append(np.column_stack([(values == v) / np.sqrt(2) for v in self.levels_[column]]))
        return np.hstack(blocks).astype(float)


K_RANGE = range(2, 11)
READING_K = 4


def choose_k(x: np.ndarray, k_values=K_RANGE, seed: int = 7) -> pd.DataFrame:
    """Silhouette du k-means pour chaque k : plus elle est haute, mieux les familles sont séparées."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = []
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        score = silhouette_score(x, labels, sample_size=min(2000, len(x)), random_state=seed)
        rows.append({"k": k, "silhouette": round(float(score), 3)})
    return pd.DataFrame(rows)


def structure_verdict(silhouettes: pd.DataFrame, reading_k: int = READING_K) -> dict:
    """Le nombre de familles vient-il des données, ou est-ce un choix de lecture ?

    Structure nette si la meilleure silhouette dépasse 0,5, ou si elle dépasse 0,25
    et forme un vrai sommet (ni au bord de la plage testée, ni sur une courbe qui
    baisse sans cesse). Sinon, le nombre de familles est un choix de lecture.
    """
    scores = silhouettes.set_index("k")["silhouette"]
    best_k, best = int(scores.idxmax()), float(scores.max())
    edge = best_k in (scores.index.min(), scores.index.max())
    decreasing = bool((scores.diff().dropna() <= 0.005).all())
    clear = best >= 0.5 or (best >= 0.25 and not edge and not decreasing)
    if clear:
        reason = f"Structure en groupes : la séparation est la plus nette avec {best_k} familles (silhouette {fmt(best, 2)})."
    else:
        why = []
        if best < 0.25:
            why.append(f"la meilleure silhouette ({fmt(best, 2)}) est sous 0,25")
        if decreasing:
            why.append("la silhouette baisse à chaque famille ajoutée")
        elif edge:
            why.append(f"le meilleur score est au bord de la plage testée ({best_k} familles)")
        reason = (
            "Pas de structure en groupes nette : " + " et ".join(why) + ". Les obligations forment plutôt un "
            f"continuum ; le nombre de familles est donc un choix de lecture ({reading_k} par défaut), pas une "
            "découverte. La séparation la plus nette est en "
            f"{best_k} familles (silhouette {fmt(best, 2)})."
        )
    return {"clear": clear, "best_k": best_k, "best_silhouette": round(best, 3),
            "chosen_k": best_k if clear else reading_k, "reason": reason}


def _fit(method: str, x: np.ndarray, k: int, seed: int):
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        return model.labels_, model
    if method == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x), None
    if method == "gmm":
        model = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3,
                                n_init=3, random_state=seed).fit(x)
        return model.predict(x), model
    if method == "hdbscan":
        size = max(30, len(x) // 100)
        return HDBSCAN(min_cluster_size=size, min_samples=10, copy=True).fit_predict(x), None
    raise ValueError(f"Méthode inconnue : {method}")


def _family_name(group: pd.DataFrame) -> str:
    letter = rating_bucket(group["notch"].median())
    maturity = group["maturity_years"].median()
    sector_share = group["sector"].value_counts(normalize=True)
    sector = sector_share.index[0] if sector_share.iloc[0] >= 0.4 else "multi-secteurs"
    sub = " · subordonnée" if group["sen_rank"].median() >= 3 else ""
    return f"{letter} · {fmt(maturity)} ans · {sector}{sub}"


def _profiles(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for label in sorted(set(labels)):
        group = df[labels == label]
        top_sector = group["sector"].value_counts(normalize=True)
        top_ccy = group["currency"].value_counts(normalize=True)
        rows.append({
            "famille": ATYPICAL if label == -1 else f"F{label + 1}",
            "nom": "Obligations atypiques" if label == -1 else _family_name(group),
            "obligations": len(group), "emetteurs": group["issuer"].nunique(),
            "notation_mediane": notch_label(group["notch"].median()),
            "maturite_mediane": round(float(group["maturity_years"].median()), 1),
            "secteur_principal": f"{top_sector.index[0]} ({pct(top_sector.iloc[0])})",
            "devise_principale": f"{top_ccy.index[0]} ({pct(top_ccy.iloc[0])})",
            "part_subordonnee": round(float((group["sen_rank"] >= 3).mean()), 3),
            "spread_median": round(float(group["z_spread"].median()), 1),
            "spread_p25": round(float(group["z_spread"].quantile(.25)), 1),
            "spread_p75": round(float(group["z_spread"].quantile(.75)), 1),
        })
    return pd.DataFrame(rows)


def _projection(x: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[float], list[str]]:
    """Deux axes d'analyse en composantes principales, et ce qui les porte."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0).fit(x)
    names = {"notch": "notation", "log_t": "maturité", "sen_rank": "séniorité", "log_amt": "taille",
             "log_spread": "spread", "sector": "secteur", "currency": "devise"}
    axes = []
    for component in pca.components_:
        weight = pd.Series(component**2).groupby(np.array(groups)).sum().sort_values(ascending=False)
        axes.append(" et ".join(names[g] for g in weight.index[:2]))
    return pca, [round(float(v), 3) for v in pca.explained_variance_ratio_], axes


def cluster_universe(estimator, method: str = "kmeans", k: int | None = None,
                     include_spread: bool = False, target: dict | None = None, seed: int = 7,
                     reading_k: int | None = None) -> dict:
    """Familles d'obligations selon une méthode ; place éventuellement une obligation cible.

    `target` : résultat d'estimation (dict renvoyé par `estimate`).
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import NearestNeighbors

    reading_k = reading_k or estimator.settings.get("analysis", {}).get("clustering_reading_k", READING_K)
    df = real_bonds(estimator)
    encoder = ClusterEncoder(include_spread).fit(df)
    x = encoder.transform(df)
    silhouettes = choose_k(x, seed=seed)
    verdict = structure_verdict(silhouettes, reading_k)
    chosen_k = k or verdict["chosen_k"]

    all_labels, models = {}, {}
    for name in METHODS:
        all_labels[name], models[name] = _fit(name, x, chosen_k, seed)
    labels = all_labels[method]
    names = list(METHODS)
    agreement = pd.DataFrame(
        [[round(float(adjusted_rand_score(all_labels[a], all_labels[b])), 2) for b in names] for a in names],
        index=[METHODS[n] for n in names], columns=[METHODS[n] for n in names],
    )

    profiles = _profiles(df, labels)
    family = np.where(labels == -1, ATYPICAL, np.char.add("F", (labels + 1).astype(str)))
    y = np.log(df["z_spread"].to_numpy())
    explained_cluster = r2(y, pd.Series(y).groupby(family).transform("mean"))
    letters = df["notch"].map(rating_bucket).to_numpy()
    explained_letter = r2(y, pd.Series(y).groupby(letters).transform("mean"))
    combo = pd.Series(family).astype(str) + "|" + letters
    explained_both = r2(y, pd.Series(y).groupby(combo.to_numpy()).transform("mean"))

    pca, variance, axes = _projection(x, encoder.groups_)
    coords = pca.transform(x)
    points = pd.DataFrame({
        "axe1": coords[:, 0], "axe2": coords[:, 1], "famille": family,
        "issuer": df["issuer"], "rating": df["rating"], "maturity_years": df["maturity_years"].round(1),
        "sector": df["sector"], "z_spread": df["z_spread"],
    })

    findings = []
    n_families = int((profiles["famille"] != ATYPICAL).sum())
    if method == "hdbscan":
        findings.append(
            f"HDBSCAN trouve {n_families} zones denses"
            + (f" et {int((labels == -1).sum())} obligations atypiques" if (labels == -1).any() else "") + "."
        )
    else:
        findings.append(f"{METHODS[method]} découpe l'échantillon en {n_families} familles.")
    if method != "hdbscan":
        findings.append(verdict["reason"])
        if k is not None:
            findings.append(f"Nombre de familles fixé à la main : {k}.")
    if method == "hdbscan":
        findings.append("HDBSCAN choisit seul son nombre de familles ; les obligations hors des zones denses "
                        "sont classées atypiques, ce qui ne veut pas dire mal valorisées.")
    findings.append(
        f"Les familles expliquent {pct(explained_cluster)} de la dispersion des spreads, contre "
        f"{pct(explained_letter)} pour la lettre de notation seule, et {pct(explained_both)} pour les deux combinées."
        + (" Attention : le spread a servi à former les familles, ce chiffre est donc flatteur." if include_spread else "")
    )
    findings.append(
        f"Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : "
        + ", ".join(f"{METHODS[n].lower()} {fmt(agreement.loc[METHODS[method], METHODS[n]], 2)}"
                    for n in names if n != method) + "."
    )
    findings.append(
        f"La carte montre {pct(sum(variance))} de l'information : l'axe horizontal résume surtout "
        f"{axes[0]}, l'axe vertical {axes[1]}. Deux points proches sur la carte peuvent être éloignés en réalité."
    )

    placement = None
    if target is not None:
        isin = target["bond"].get("isin")
        if isin and isin in set(estimator.df_["isin"]):
            prepared = estimator.resolve(isin)[0]
        else:
            prepared = estimator.resolve({key: v for key, v in target["bond"].items() if v is not None})[0]
        frame = pd.DataFrame([{**prepared, "z_spread": target["z_spread_bp"]}])
        xt = encoder.transform(frame)
        neighbours = NearestNeighbors(n_neighbors=2).fit(x)
        typical = np.percentile(neighbours.kneighbors(x)[0][:, 1], 95)
        if method in ("kmeans", "gmm"):
            label = int(models[method].predict(xt)[0])
        else:
            valid = labels != -1
            nn = NearestNeighbors(n_neighbors=1).fit(x[valid])
            distance, index = nn.kneighbors(xt)
            label = int(labels[valid][index[0, 0]]) if distance[0, 0] <= typical else -1
        own = ATYPICAL if label == -1 else f"F{label + 1}"
        members = df[family == own]
        nearest = float(neighbours.kneighbors(xt, n_neighbors=1)[0][0, 0])
        tc = pca.transform(xt)[0]
        placement = {"famille": own, "axe1": float(tc[0]), "axe2": float(tc[1]),
                     "isolee": nearest > typical}
        if own == ATYPICAL or members.empty:
            findings.append("Votre obligation ne ressemble à aucune famille : elle est atypique pour cet échantillon.")
        else:
            row = profiles.set_index("famille").loc[own]
            share_below = float((members["z_spread"] < target["z_spread_bp"]).mean())
            placement.update({"nom": row["nom"], "percentile": round(share_below, 3)})
            findings.append(
                f"Votre obligation appartient à la famille {own} ({row['nom']}, {int(row['obligations'])} obligations, "
                f"spread médian {fmt(row['spread_median'])} pb). Son estimation de {fmt(target['z_spread_bp'])} pb "
                f"est au-dessus de {pct(share_below)} de sa famille."
            )
        if placement["isolee"]:
            findings.append("Elle est loin de toutes les obligations de l'échantillon : prudence sur l'estimation.")

    return {
        "title": f"Clustering · {METHODS[method]}",
        "method": method, "k": chosen_k, "include_spread": include_spread,
        "findings": findings,
        "tables": {"familles": profiles, "silhouette": silhouettes, "structure": verdict, "accord": agreement.reset_index(names="methode"),
                   "variance_carte": variance, "axes": axes,
                   "part_expliquee": {"familles": round(explained_cluster, 3), "lettre": round(explained_letter, 3),
                                      "familles_et_lettre": round(explained_both, 3)},
                   "placement": placement},
        "points": points,
    }
