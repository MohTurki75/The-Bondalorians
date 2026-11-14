# Générateur synthétique d'obligations

Ce programme génère un fichier CSV d'obligations fictives pour tester un système de recherche de comparables et d'estimation du Z-Spread.

> **Important :** les obligations et les Z-Spreads générés sont synthétiques. Ils ne représentent pas des données de marché et ne doivent pas être utilisés pour prendre des décisions financières.

## Prérequis

- Python 3 installé sur votre Mac.
- Aucune bibliothèque Python supplémentaire n'est nécessaire : le script utilise uniquement la bibliothèque standard de Python.

Vérifier que Python est installé :

```bash
python3 --version
```

## Télécharger les fichiers

Placez `generate_bonds.py` dans le dossier où vous souhaitez créer le CSV. Vous pouvez aussi ouvrir ce dossier dans Terminal.

## Lancer le générateur

Dans Terminal, accédez au dossier contenant le script, puis lancez :

```bash
python3 generate_bonds.py
```

Par défaut, le programme génère 4 000 obligations dans le fichier `bonds_synthetic.csv`, dans le dossier courant.

## Choisir le nombre de lignes, la graine et le fichier de sortie

```bash
python3 generate_bonds.py --rows 4000 --seed 42 --output bonds_synthetic.csv
```

Options disponibles :

| Option | Valeur par défaut | Description |
|---|---:|---|
| `--rows` | `4000` | Nombre d'obligations à générer. Minimum : 1. |
| `--seed` | `42` | Graine aléatoire. Gardez la même valeur pour reproduire le même fichier. |
| `--output` | `bonds_synthetic.csv` | Chemin et nom du fichier CSV à créer. |

Par exemple, pour générer 100 lignes dans un sous-dossier :

```bash
python3 generate_bonds.py --rows 100 --seed 7 --output data/bonds_test.csv
```

Le sous-dossier de sortie est créé automatiquement s'il n'existe pas.

## Colonnes générées

| Colonne | Description |
|---|---|
| `isin` | Identifiant synthétique au format `XS` suivi de 10 chiffres. |
| `issuer` | Identifiant fictif de l'émetteur. Plusieurs obligations peuvent partager un émetteur. |
| `sector` | Secteur d'activité fictif. |
| `country` | Code pays à deux lettres. |
| `currency` | Devise associée au pays généré. |
| `rating` | Rating de crédit synthétique, de `AAA` à `B`. |
| `seniority` | Rang de séniorité de l'obligation. |
| `coupon_type` | Type de coupon : fixe, flottant ou indexé sur l'inflation. |
| `callable` | Indique si l'obligation est callable (`True` ou `False`). |
| `maturity_years` | Maturité résiduelle synthétique en années. |
| `amount_outstanding` | Montant en circulation synthétique. |
| `z_spread` | Z-Spread synthétique en points de base (bps). |

## Logique synthétique du Z-Spread

Le spread est généré à partir d'une combinaison de facteurs :

- rating : les ratings plus faibles tendent à augmenter le spread ;
- maturité : effet croissant et non linéaire ;
- séniorité : la dette subordonnée reçoit généralement un spread plus élevé ;
- devise ;
- type de coupon ;
- caractère callable ;
- secteur et émetteur ;
- bruit aléatoire et rares valeurs extrêmes.

Un effet de régime de marché commun est également ajouté à chaque fichier généré. Cela crée une certaine co-mouvance entre les spreads.

Ces règles servent à produire un jeu d'essai avec des relations apprenables. Elles ne constituent pas un modèle de marché calibré. En particulier, les spreads des obligations callable sont des valeurs synthétiques et ne sont pas des OAS calculés à partir de modèles d'options.

## Importer le CSV dans PostgreSQL

Le fichier peut être importé depuis `psql` avec `\\copy`, par exemple :

```sql
CREATE TABLE bonds (
    isin TEXT PRIMARY KEY,
    issuer TEXT,
    sector TEXT,
    country CHAR(2),
    currency CHAR(3),
    rating TEXT,
    seniority TEXT,
    coupon_type TEXT,
    callable BOOLEAN,
    maturity_years NUMERIC(6, 2),
    amount_outstanding NUMERIC(18, 1),
    z_spread NUMERIC(8, 1)
);
```

Puis, depuis le terminal `psql` :

```text
\\copy bonds FROM 'bonds_synthetic.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');
```

Adaptez le chemin si le CSV se trouve dans un autre dossier. La commande `\\copy` est une commande du client `psql`, elle ne se lance pas dans un éditeur SQL générique.

## Fichiers

```text
generate_bonds.py   # générateur Python
README.md           # ce guide
bonds_synthetic.csv # fichier produit après exécution
```
