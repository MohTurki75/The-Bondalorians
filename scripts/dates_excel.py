"""Dates lisibles par Excel, maturités résiduelles et date de maturité de pricing.

    python scripts/dates_excel.py data/base_bonds2.csv
    python scripts/dates_excel.py data/base_bonds2.csv --sortie data/base_dates.csv --sep ","

Ce que fait le script, sur l'export Bloomberg (pricing_date, datematurit, prochainedatecall…) :

1. toutes les colonnes de date passent au format AAAA/MM/JJ (« Sun Nov 02 2036 00:00:00 GMT+0000 (…) »
   -> 2036/11/02 ; « 20260316 » -> 2026/03/16 ; « #N/A » -> vide) ;
2. ajoute `maturite_pricing` = min(prochaine date de call, date de maturité) si l'obligation est
   remboursable (typematurit contient « CALL »), sinon la date de maturité. Un call déjà passé à la
   date de pricing est ignoré. Une perpétuelle sans maturité prend son call ;
3. ajoute les durées en années depuis la date de pricing de chaque ligne : `ttm_maturite`,
   `ttm_call`, `ttm_maturite_pricing` ;
4. compare `maturite_pricing` à la colonne Bloomberg `datematurit_pricingobligation`
   (colonne `ecart_bloomberg_jours`) et affiche le résumé.

Sorties : un CSV (séparateur « ; » et virgule décimale par défaut, pour Excel en français) et un
.xlsx avec de vraies dates Excel si openpyxl est installé.
Autonome : n'utilise que pandas (et openpyxl pour le .xlsx).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMNS = ["pricing_date", "datematurit_pricingobligation", "datematurit", "dateemission",
                "premieredatesettlement", "prochainedatecall"]
JS_DATE = r"^[A-Za-z]{3}\s+([A-Za-z]{3}\s+\d{1,2}\s+\d{4})"  # « Sun Nov 02 2036 … » -> « Nov 02 2036 »
MISSING = ["#N/A", "#N/A N/A", "#N/A Field Not Applicable", "N/A", ""]


def to_date(series: pd.Series) -> pd.Series:
    """Date sans heure ni fuseau, quel que soit le format de l'export."""
    text = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    js = text.str.extract(JS_DATE)[0]
    out = out.fillna(pd.to_datetime(js, format="%b %d %Y", errors="coerce"))
    compact = text.where(text.str.fullmatch(r"\d{8}").fillna(False))
    out = out.fillna(pd.to_datetime(compact, format="%Y%m%d", errors="coerce"))
    rest = text.where(out.isna() & text.notna())
    out = out.fillna(pd.to_datetime(rest, errors="coerce", dayfirst=False, format="mixed"))
    return out.dt.normalize()


def years(target: pd.Series, origin: pd.Series) -> pd.Series:
    return ((target - origin).dt.days / 365.25).round(4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dates Excel et maturité de pricing.")
    parser.add_argument("csv")
    parser.add_argument("--sortie", help="défaut : <fichier>_dates.csv")
    parser.add_argument("--sep", default=";", help="séparateur du CSV de sortie (défaut « ; »)")
    parser.add_argument("--decimal", default=None, help="séparateur décimal (défaut « , » si sep = « ; »)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, na_values=MISSING, encoding="utf-8-sig")
    dates = {c: to_date(df[c]) for c in DATE_COLUMNS if c in df.columns}
    pricing = dates["pricing_date"]
    maturity = dates.get("datematurit", pd.Series(pd.NaT, index=df.index))
    call = dates.get("prochainedatecall", pd.Series(pd.NaT, index=df.index))

    callable_ = df.get("typematurit", pd.Series("", index=df.index)).fillna("").str.upper().str.contains("CALL")
    useful_call = call.where(callable_ & (call >= pricing))  # un call déjà passé ne compte pas
    pricing_maturity = maturity.where(useful_call.isna() | (maturity <= useful_call), useful_call)
    pricing_maturity = pricing_maturity.fillna(useful_call)  # perpétuelle : pas de maturité, le call

    out = df.copy()
    for column, values in dates.items():
        out[column] = values.dt.strftime("%Y/%m/%d")
    position = out.columns.get_loc("datematurit") + 1 if "datematurit" in out.columns else len(out.columns)
    new = {
        "remboursable": np.where(callable_, "oui", "non"),
        "maturite_pricing": pricing_maturity.dt.strftime("%Y/%m/%d"),
        "ttm_maturite": years(maturity, pricing),
        "ttm_call": years(call.where(callable_), pricing),
        "ttm_maturite_pricing": years(pricing_maturity, pricing),
    }
    if "datematurit_pricingobligation" in dates:
        new["ecart_bloomberg_jours"] = (pricing_maturity - dates["datematurit_pricingobligation"]).dt.days
    for offset, (name, values) in enumerate(new.items()):
        out.insert(position + offset, name, values)

    # Nombres stockés en texte dans l'export : convertis pour qu'Excel puisse calculer.
    for column in ("niveaumid", "niveauzspread", "rendement", "prixbid", "durationbid", "discountmargenextcall"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    path = Path(args.sortie) if args.sortie else Path(args.csv).with_name(Path(args.csv).stem + "_dates.csv")
    decimal = args.decimal or ("," if args.sep == ";" else ".")
    out.to_csv(path, index=False, sep=args.sep, decimal=decimal, encoding="utf-8-sig")
    print(f"Écrit : {path} ({len(out)} lignes, séparateur « {args.sep} », décimale « {decimal} »)")
    try:
        xlsx = path.with_suffix(".xlsx")
        excel = out.copy()
        for column in list(dates) + ["maturite_pricing"]:
            excel[column] = pd.to_datetime(excel[column], format="%Y/%m/%d", errors="coerce")
        with pd.ExcelWriter(xlsx, engine="openpyxl", datetime_format="YYYY/MM/DD") as writer:
            excel.to_excel(writer, index=False, sheet_name="obligations")
        print(f"Écrit : {xlsx} (vraies dates Excel)")
    except ImportError:
        print("openpyxl absent : pas de .xlsx (pip install openpyxl).")

    # Contrôles.
    print("\nDates illisibles (renseignées mais non converties) :")
    for column in dates:
        raw = df[column].notna()
        bad = int((raw & dates[column].isna()).sum())
        print(f"  {column:32} {bad}" + (f"  ex. {df.loc[raw & dates[column].isna(), column].iloc[0]!r}" if bad else ""))
    print(f"\nRemboursables : {int(callable_.sum())} ; maturité de pricing = call pour "
          f"{int((callable_ & pricing_maturity.eq(useful_call) & useful_call.notna()).sum())} ; "
          f"call déjà passé ignoré : {int((callable_ & call.notna() & (call < pricing)).sum())} ; "
          f"sans maturité ni call : {int(pricing_maturity.isna().sum())}")
    if "ecart_bloomberg_jours" in new:
        gap = pd.Series(new["ecart_bloomberg_jours"]).dropna()
        print(f"Comparaison avec datematurit_pricingobligation (Bloomberg) : identique pour {int((gap == 0).sum())} "
              f"lignes sur {len(gap)} ; différente pour {int((gap != 0).sum())}.")
        differ = out.loc[out["ecart_bloomberg_jours"].fillna(0) != 0,
                         ["id", "typematurit", "datematurit", "prochainedatecall", "datematurit_pricingobligation",
                          "maturite_pricing", "ecart_bloomberg_jours"]]
        if len(differ):
            print(differ.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
