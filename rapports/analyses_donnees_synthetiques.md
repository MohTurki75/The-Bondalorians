# Analyses de l'échantillon et des modèles

- Généré le : 2026-11-14 17:31 UTC
- Source : données synthétiques
- Obligations : 4 000 ; version du modèle : `0.1.0-4000-a4c3cfe522`
- CatBoost : actif ; forêt aléatoire : incluse

## Consigne

Tu es analyste crédit et data scientist. Tu expliques ces résultats à un gérant de portefeuille
qui n'est pas spécialiste des données. Le système estime le Z-Spread juste d'une obligation à partir d'un
univers d'obligations comparables ; ce document décrit l'échantillon, chaque modèle et un clustering.

Règles :
1. N'utilise que les chiffres de ce document. N'en invente aucun ; si un calcul te semble utile, dis-le
   sans le faire.
2. Distingue ce qui est mesuré ici (fait), ce que tu en déduis (interprétation) et ce que tu supposes
   (hypothèse).
3. Toutes les erreurs citées sont hors échantillon : chaque obligation a été estimée par un modèle qui ne
   l'avait pas vue.
4. Si la source est « données synthétiques », rappelle que ces chiffres valident la mécanique et ne disent
   rien du marché réel.
5. Sois critique : signale ce qui est fragile, contradictoire ou insuffisamment démontré.

Réponds, dans cet ordre :
1. Que dit l'échantillon, et qu'est-ce qui détermine le niveau des spreads ?
2. Quel modèle est le plus fiable, dans quels cas, et l'apprentissage automatique apporte-t-il quelque chose ?
3. Le score de confiance et l'intervalle à 90 % sont-ils dignes de confiance ?
4. Le clustering révèle-t-il une vraie structure, ou seulement un découpage de lecture ?
5. Si une obligation est décrite en fin de document : comment se situe-t-elle, et avec quelle confiance ?
6. Les trois faiblesses principales du système, et ce qu'il faudrait vérifier sur les données réelles.

## Glossaire

| Terme | Sens |
|---|---|
| pb | Point de base : 0,01 %. |
| Médiane | Valeur du milieu : la moitié des cas est au-dessous. |
| Hors échantillon | Estimé par un modèle qui n'a pas vu l'obligation. |
| Part de la dispersion expliquée | De 0 à 100 % : à quel point une information permet de deviner le spread (en logarithme). |
| Courbe des pairs | Régression du log du spread sur notation, maturité, séniorité, secteur, devise. |
| Modèle A | Courbe des pairs + écart propre à l'émetteur (rétréci s'il a peu d'obligations) + écart des comparables. |
| Modèle B | CatBoost (arbres de décision en série) sans l'émetteur comme variable. |
| Modèle C | Modèle A corrigé par un CatBoost qui apprend ses erreurs. |
| Forêt aléatoire | Moyenne de centaines d'arbres de décision ; modèle de comparaison, hors estimation. |
| k | Dans le modèle A, l'écart d'un émetteur à n obligations est retenu à n / (n + k). |
| SHAP | Contribution de chaque variable à une prédiction. |
| Couverture | Part des cas où le vrai spread tombe dans l'intervalle annoncé. |
| Silhouette | De −1 à 1 : séparation des familles d'un clustering ; sous 0,25, elles se chevauchent. |
| Indice de Rand ajusté | De 0 (hasard) à 1 (identique) : accord entre deux clusterings. |
| Palier | Simple, intermédiaire, complexe ou émetteur inconnu, selon l'information disponible sur l'émetteur. |

## 1. Échantillon

- 4 000 obligations de 539 émetteurs ; un émetteur a en moyenne 7,4 obligations, et 74 émetteurs (14 %) n'en ont qu'une.
- Le spread médian est de 183 pb ; la moitié des obligations sont entre 101 et 359 pb.
- Notation (cran) est la caractéristique qui explique le plus la dispersion des spreads, à elle seule (84 %).
- La lettre de notation seule explique 79 % de la dispersion. Une fois la notation connue, l'émetteur explique encore 37 % de ce qui reste : c'est pourquoi l'estimation suit la courbe propre à chaque émetteur quand elle existe.
- La catégorie la plus représentée est BBB (36 % de l'échantillon) : les estimations y reposent sur plus de comparables que dans les catégories rares.

### Spread par grande notation (pb)

| Notation | Obligations | 10e centile | 25e centile | Médiane | 75e centile | 90e centile |
|---|---|---|---|---|---|---|
| AAA | 44,0 | 19,5 | 25,8 | 34,2 | 43,3 | 52,0 |
| AA | 333 | 30,5 | 40,0 | 51,0 | 67,5 | 82,2 |
| A | 1 056 | 59,0 | 76,8 | 100 | 130 | 168 |
| BBB | 1 441 | 114 | 148 | 197 | 265 | 358 |
| BB | 835 | 269 | 333 | 438 | 601 | 909 |
| B | 248 | 689 | 884 | 1 199 | 1 868 | 2 711 |
| CCC et moins | 43,0 | 2 347 | 2 898 | 3 979 | 6 806 | 10 504 |

### Part de la dispersion des spreads expliquée par chaque caractéristique seule

| Caractéristique | Part expliquée (%) |
|---|---|
| Notation (cran) | 84,2 |
| Notation (lettre) | 79,0 |
| Émetteur | 73,7 |
| Séniorité | 29,0 |
| Secteur | 6,9 |
| Maturité (5 tranches) | 6,0 |
| Pays | 2,6 |
| Devise | 0,0 |

## 2. Les modèles un par un

### Courbe des pairs

- La courbe des pairs explique 97 % de la dispersion des spreads (en log) à elle seule, sans rien savoir de l'émetteur.
- À 5 ans, pour une obligation SENIOR_UNSECURED en EUR du secteur Assurance : 97 pb en A, 199 pb en BBB, 441 pb en BB.
- Pour un BBB, passer de 2 à 10 ans fait passer le spread de 150 à 247 pb.
- Les courbes sont tracées pour une obligation typique de l'échantillon : changer la séniorité, le secteur ou la devise les décale vers le haut ou le bas.

### Écart propre à l'émetteur

- k vaut 0,28 : avec 1 obligation, l'écart de l'émetteur est retenu à 78 % ; avec 3, à 92 % ; avec 10, à 97 %.
- Deux obligations d'un même émetteur s'écartent typiquement de ±8 % entre elles ; un émetteur s'écarte typiquement de ±15 % de la courbe des pairs. Plus le second chiffre domine, plus on fait confiance à l'émetteur, même avec peu d'obligations.
- L'écart retenu le plus fort est celui de Émetteur 0592 : 46 % par rapport à ses pairs, sur 22 obligation(s).

### Choix des comparables

- 200 combinaisons essayées sur 800 obligations ; sur 800 autres obligations, l'erreur typique passe de 11,1 % à 10,5 %.
- Réglage adopté.
- Critère le plus pesant : maturité (36 %). 60 comparables au plus, largeur de noyau 0,42.
- Ces poids varient beaucoup d'un jeu de données à l'autre : ils mesurent ce qui aide à prédire ici, pas une vérité de marché.

| Critère | Poids |
|---|---|
| notch | 0,097 |
| maturity | 0,364 |
| sector | 0,279 |
| country | 0,102 |
| size | 0,158 |

Nombre maximal de comparables : 60 ; largeur du noyau : 0,42.

### CatBoost (modèle B)

- Effet moyen = de combien, en moyenne, une variable déplace le spread prédit d'une obligation par rapport à l'obligation moyenne.
- Les trois variables les plus utilisées : notation (64 %), maturité (18 %), rang de séniorité (17 %).
- Effet de la notation seule : -81 % en AAA, 1 070 % en CCC-. La contrainte de monotonie garantit que, pour une obligation donnée, dégrader la notation ne baisse jamais le spread prédit ; les moyennes par notation peuvent légèrement fluctuer là où les obligations sont rares.

| Variable | Effet moyen sur le spread (%) |
|---|---|
| Notation | 64,1 |
| Maturité | 18,4 |
| Rang de séniorité | 16,7 |
| Secteur | 7,0 |
| Séniorité (catégorie) | 3,5 |
| Remboursable | 2,1 |
| Pays | 2,0 |
| Devise | 1,8 |
| Taille | 0,2 |
| Type de coupon | 0,0 |

### Correcteur (modèle C)

- Le correcteur modifie l'estimation du modèle A de 1,8 % en médiane ; 90 % des corrections sont entre -7,5 % et 8,7 %.
- Hors échantillon, il améliore le modèle A dans 49 % des cas.
- Erreur médiane : 8,3 % pour A, 8,4 % pour C.

### Forêt aléatoire

- Importance = de combien l'erreur de la forêt augmente quand on brouille une variable (mesuré sur des émetteurs qu'elle n'a jamais vus).
- Variables dont la forêt dépend le plus : notation (+64,4 points d'erreur), maturité (+11,3 points d'erreur), rang de séniorité (+3,3 points d'erreur).
- Variables quasi inutiles pour elle : type de coupon, devise, taille, pays.

| Variable | Hausse de l'erreur si brouillée (points de %) |
|---|---|
| Notation | 64,4 |
| Maturité | 11,3 |
| Rang de séniorité | 3,3 |
| Séniorité (catégorie) | 0,6 |
| Secteur | 0,2 |
| Remboursable par anticipation | 0,1 |
| Type de coupon | -0,1 |
| Devise | -0,1 |
| Taille | -0,2 |
| Pays | -0,4 |

### Effets comparés

- Chaque courbe montre le spread moyen prédit si l'on change une seule caractéristique de toutes les obligations, le reste restant tel quel.
- Courbes proches : les modèles ont appris la même chose. Une forêt aléatoire produit des paliers, un modèle de régression des courbes lisses.

## 3. Comparaison des modèles et confiance

### Erreur médiane hors échantillon (pb), par palier et par modèle

| Palier | Obligations testées | A · Comparables | B · CatBoost | C · Comparables corrigés | Courbe des pairs | Forêt aléatoire |
|---|---|---|---|---|---|---|
| Complexe | 406 | 40,8 | 86,0 | 41,6 | 74,9 | 89,2 |
| Intermédiaire | 1 843 | 9,1 | 17,2 | 9,4 | 17,5 | 18,5 |
| Simple | 1 247 | 8,4 | 14,4 | 8,3 | 16,4 | 13,6 |
| Émetteur inconnu | 3 600 | 19,7 | 21,4 | 20,8 | 19,6 | 22,3 |

- Erreur médiane = la moitié des obligations sont estimées avec une erreur plus faible. Chaque obligation est estimée sans avoir servi à l'apprentissage.
- Complexe : meilleur modèle A · Comparables (40,8 pb).
- Intermédiaire : meilleur modèle A · Comparables (9,1 pb).
- Simple : meilleur modèle C · Comparables corrigés (8,3 pb).
- Émetteur inconnu : meilleur modèle Courbe des pairs (19,6 pb).
- La forêt aléatoire bat le modèle A dans 0 palier(s) sur 4. Elle n'utilise pas la courbe de l'émetteur : elle ne peut gagner que là où l'émetteur apporte peu.
- Modèle retenu par palier dans l'estimation : Complexe → A, Émetteur inconnu → A, Intermédiaire → A, Simple → A.

### Score de confiance : erreur par tranche de difficulté prévue

| Tranche (1 = plus facile) | Erreur prévue (%) | Erreur réalisée (%) |
|---|---|---|
| 1 | 4,4 | 5,7 |
| 2 | 4,6 | 5,8 |
| 3 | 4,8 | 5,7 |
| 4 | 5,1 | 5,5 |
| 5 | 6,5 | 8,1 |
| 6 | 8,7 | 11,8 |
| 7 | 9,0 | 11,5 |
| 8 | 9,2 | 11,0 |
| 9 | 9,4 | 11,8 |
| 10 | 9,9 | 13,2 |

### Intervalle à 90 % : couverture par palier

| Palier | Obligations | Couverture (%) | Largeur médiane (%) |
|---|---|---|---|
| Complexe | 406 | 90,4 | 32,0 |
| Émetteur inconnu | 3 600 | 90,0 | 60,7 |
| Intermédiaire | 1 843 | 90,1 | 30,6 |
| Simple | 1 247 | 90,1 | 27,1 |

- Les 10 % de cas jugés les plus faciles ont une erreur médiane de 5,7 % ; les 10 % les plus difficiles, 13,2 %.
- Le classement par difficulté est utile (corrélation de rang 0,32 entre difficulté prévue et erreur réalisée).
- Couverture visée : 90 %. Obtenue : complexe 90 %, émetteur inconnu 90 %, intermédiaire 90 %, simple 90 %.

## 4. Clustering

Le clustering regroupe les obligations par leurs caractéristiques (notation, maturité, séniorité, taille, secteur, devise), sans le spread. Il n'intervient pas dans l'estimation.

### Combien de familles ? (silhouette du k-means)

| Familles | Silhouette |
|---|---|
| 2 | 0,350 |
| 3 | 0,210 |
| 4 | 0,204 |
| 5 | 0,175 |
| 6 | 0,156 |
| 7 | 0,147 |
| 8 | 0,139 |
| 9 | 0,128 |
| 10 | 0,126 |

- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).

### Accord entre méthodes (indice de Rand ajusté)

| Méthode | k-means | Hiérarchique (Ward) | Mélange gaussien | HDBSCAN (densité) |
|---|---|---|---|---|
| k-means | 1,00 | 0,36 | 0,05 | -0,00 |
| Hiérarchique (Ward) | 0,36 | 1,00 | 0,07 | 0,03 |
| Mélange gaussien | 0,05 | 0,07 | 1,00 | 0,05 |
| HDBSCAN (densité) | -0,00 | 0,03 | 0,05 | 1,00 |

### Clustering · k-means

- k-means découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 57 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 89 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : hiérarchique (ward) 0,36, mélange gaussien 0,05, hdbscan (densité) -0,00.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout séniorité et notation, l'axe vertical maturité et notation. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F3 (A · 7 ans · multi-secteurs, 1374 obligations, spread médian 105 pb). Son estimation de 100 pb est au-dessus de 46 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | BB · 6 ans · multi-secteurs · subordonnée | 464 | 237 | BB+ | 6,1 | Assurance (25 %) | 356 | 686 | 1 439 |
| F2 | BB · 7 ans · multi-secteurs | 1 273 | 261 | BB+ | 7,0 | Assurance (17 %) | 246 | 348 | 514 |
| F3 | A · 7 ans · multi-secteurs | 1 374 | 273 | A | 7,1 | Consommation (13 %) | 70,3 | 105 | 145 |
| F4 | BBB · 2 ans · multi-secteurs | 889 | 338 | BBB+ | 2,1 | Assurance (13 %) | 77,1 | 122 | 186 |

### Clustering · Hiérarchique (Ward)

- Hiérarchique (Ward) découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 51 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 89 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means 0,36, mélange gaussien 0,07, hdbscan (densité) 0,03.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout séniorité et notation, l'axe vertical maturité et notation. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F3 (BBB · 9 ans · multi-secteurs, 1292 obligations, spread médian 153 pb). Son estimation de 100 pb est au-dessus de 22 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | A · 3 ans · multi-secteurs | 1 322 | 379 | A- | 2,9 | Services publics (13 %) | 61,9 | 99,8 | 158 |
| F2 | BB · 6 ans · multi-secteurs · subordonnée | 688 | 260 | BB+ | 6,1 | Assurance (37 %) | 288 | 548 | 1 095 |
| F3 | BBB · 9 ans · multi-secteurs | 1 292 | 371 | BBB+ | 8,6 | Énergie (15 %) | 103 | 153 | 218 |
| F4 | BB · 6 ans · multi-secteurs | 698 | 168 | BB | 6,4 | Consommation (22 %) | 304 | 422 | 626 |

### Clustering · Mélange gaussien

- Mélange gaussien découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 6 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 81 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means 0,05, hiérarchique (ward) 0,07, hdbscan (densité) 0,05.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout séniorité et notation, l'axe vertical maturité et notation. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F2 (BBB · 6 ans · multi-secteurs, 1582 obligations, spread médian 145 pb). Son estimation de 100 pb est au-dessus de 31 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | BBB · 6 ans · Banques | 584 | 172 | BBB | 5,6 | Banques (67 %) | 142 | 295 | 611 |
| F2 | BBB · 6 ans · multi-secteurs | 1 582 | 256 | BBB+ | 5,7 | Technologie (24 %) | 85,6 | 145 | 269 |
| F3 | BBB · 5 ans · multi-secteurs | 91 | 71 | BBB+ | 5,1 | Consommation (32 %) | 104 | 165 | 317 |
| F4 | BBB · 6 ans · multi-secteurs | 1 743 | 230 | BBB | 5,7 | Assurance (30 %) | 112 | 202 | 379 |

### Clustering · HDBSCAN (densité)

- HDBSCAN trouve 20 zones denses et 1334 obligations atypiques.
- HDBSCAN choisit seul son nombre de familles ; les obligations hors des zones denses sont classées atypiques, ce qui ne veut pas dire mal valorisées.
- Les familles expliquent 16 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 83 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means -0,00, hiérarchique (ward) 0,03, mélange gaussien 0,05.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout séniorité et notation, l'axe vertical maturité et notation. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation ne ressemble à aucune famille : elle est atypique pour cet échantillon.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| Atypique | Obligations atypiques | 1 334 | 403 | BBB | 4,8 | Assurance (19 %) | 97,8 | 217 | 502 |
| F1 | BB · 7 ans · multi-secteurs · subordonnée | 58 | 49 | BB | 6,6 | Assurance (19 %) | 645 | 974 | 1 714 |
| F2 | A · 5 ans · multi-secteurs | 132 | 100 | A- | 5,2 | Assurance (18 %) | 56,0 | 93,0 | 138 |
| F3 | BB · 7 ans · multi-secteurs · subordonnée | 96 | 69 | BB+ | 7,0 | Assurance (34 %) | 453 | 651 | 1 148 |
| F4 | BBB · 6 ans · Banques | 59 | 24 | BBB- | 6,2 | Banques (100 %) | 195 | 343 | 427 |
| F5 | BB · 6 ans · Assurance | 81 | 38 | BB+ | 5,8 | Assurance (100 %) | 251 | 392 | 741 |
| F6 | BBB · 6 ans · Santé | 49 | 23 | BBB | 6,5 | Santé (100 %) | 118 | 149 | 198 |
| F7 | BBB · 8 ans · Énergie | 49 | 25 | BBB+ | 8,2 | Énergie (100 %) | 140 | 206 | 257 |
| F8 | BBB · 6 ans · Consommation | 63 | 31 | BBB+ | 5,7 | Consommation (100 %) | 101 | 159 | 259 |
| F9 | A · 7 ans · Services publics | 59 | 27 | A- | 6,8 | Services publics (100 %) | 90,4 | 116 | 204 |
| F10 | BBB · 6 ans · Industrie | 42 | 28 | BBB+ | 6,2 | Industrie (100 %) | 103 | 158 | 289 |
| F11 | BBB · 7 ans · Télécoms | 86 | 30 | BBB+ | 6,7 | Télécoms (100 %) | 106 | 149 | 234 |
| F12 | BBB · 6 ans · Industrie | 231 | 54 | BBB | 6,4 | Industrie (100 %) | 89,5 | 171 | 312 |
| F13 | BBB · 6 ans · Banques | 112 | 35 | BBB+ | 5,7 | Banques (100 %) | 98,6 | 170 | 283 |
| F14 | BBB · 7 ans · Immobilier | 191 | 39 | BBB+ | 6,5 | Immobilier (100 %) | 131 | 191 | 422 |
| F15 | BBB · 6 ans · Énergie | 206 | 47 | BBB | 5,5 | Énergie (100 %) | 110 | 175 | 256 |
| F16 | BBB · 6 ans · Assurance | 148 | 43 | BBB | 5,7 | Assurance (100 %) | 149 | 228 | 381 |
| F17 | BBB · 6 ans · Santé | 204 | 48 | BBB | 6,1 | Santé (100 %) | 96,9 | 148 | 217 |
| F18 | BBB · 5 ans · Technologie | 257 | 53 | BBB+ | 5,1 | Technologie (100 %) | 79,7 | 139 | 281 |
| F19 | BBB · 5 ans · Services publics | 245 | 46 | BBB+ | 5,4 | Services publics (100 %) | 90,3 | 125 | 191 |
| F20 | BBB · 6 ans · Consommation | 298 | 55 | BBB | 5,7 | Consommation (100 %) | 94,1 | 178 | 337 |

## 5. Obligation estimée

| Émetteur | Notation | Séniorité | Maturité (ans) | Devise | Secteur |
|---|---|---|---|---|---|
| Émetteur 0345 | BBB+ | SECURED | 5,7 | EUR | Services publics |

- Z-Spread juste estimé : 100 pb ; intervalle à 90 % : 88 à 113 pb.
- Confiance : 98/100 (note A) ; erreur typique attendue 4 pb.
- Spread observé dans la base pour cette obligation : 95 pb. Il n'a pas servi à l'estimation (l'obligation est exclue de ses propres comparables).
- Palier : Intermédiaire. Émetteur connu, mais seulement 0 obligation(s) de même devise et séniorité.
- Modèle retenu : A (comparables). Estimations des modèles (pb) : courbe des pairs 103, A 100, B (CatBoost) 94, C (comparables corrigés) 99.

### Comparables

- 60 comparables retenus ; les 5 plus proches pèsent 20 % du total.
- Une fois ramenés aux caractéristiques de votre obligation, 80 % d'entre eux donnent entre 85 et 132 pb.
- L'émetteur a 13 autre(s) obligation(s) : sa propre courbe pèse 98 %, les comparables 2 %.

### Contrôles

| Contrôle | Résultat | Détail |
|---|---|---|
| Spread positif | ok | 99.6 pb |
| Plage du segment de notation | ok | Estimation 100 pb ; plage 1 %-99 % des notations voisines : 61 à 535 pb. |
| Accord champion-challenger | ok | Écart relatif entre les deux modèles : 6 % (en log). |
| Comparables suffisants | ok | Nombre effectif de comparables hors émetteur : 48.6 ; poids de la courbe émetteur : 0.98. |

### Cascade (de l'obligation moyenne à l'estimation)

| Étape | Effet (pb) | Cumul (pb) |
|---|---|---|
| Obligation moyenne de l'univers | — | 201 |
| Notation | -41,3 | 160 |
| Maturité | 7,1 | 167 |
| Séniorité | -47,0 | 120 |
| Secteur | -14,7 | 106 |
| Devise | -1,6 | 104 |
| Type de coupon | 0,0 | 104 |
| Option de remboursement anticipé | -0,8 | 103 |
| Écart émetteur (13 obligation(s), retenu à 98 %) | -3,5 | 99,6 |
| Comparables hors émetteur | -0,0 | 99,6 |

### Famille de l'obligation selon chaque méthode

- k-means : famille F3 (A · 7 ans · multi-secteurs), estimation au-dessus de 46 % de la famille.
- Hiérarchique (Ward) : famille F3 (BBB · 9 ans · multi-secteurs), estimation au-dessus de 22 % de la famille.
- Mélange gaussien : famille F2 (BBB · 6 ans · multi-secteurs), estimation au-dessus de 31 % de la famille.
- HDBSCAN (densité) : famille Atypique.

### Réserves de l'estimation

- L'estimation repose uniquement sur les données de la base : liquidité réelle, perspective de notation et actualité de l'émetteur ne sont pas observées.
- Aucune prime de nouvelle émission n'est ajoutée.

## Limites connues du système

- Les comparables et les modèles n'observent que la base : liquidité réelle, perspective de notation et actualité de l'émetteur sont invisibles.
- Aucune prime de nouvelle émission n'est ajoutée.
- Le code suppose une photographie à une date ; aucune dynamique temporelle n'est modélisée.
- Les poids des comparables appris varient d'un jeu de données à l'autre ; ils ne s'interprètent pas un par un.
