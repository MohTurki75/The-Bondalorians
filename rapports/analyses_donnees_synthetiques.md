# Analyses de l'échantillon et des modèles

- Généré le : 2026-11-16 20:49 UTC
- Source : données synthétiques
- Obligations : 4 000 ; version du modèle : `0.1.0-4000-3f1029c103`
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

- 4 000 obligations de 549 émetteurs ; un émetteur a en moyenne 7,3 obligations, et 76 émetteurs (14 %) n'en ont qu'une.
- Le spread médian est de 196 pb ; la moitié des obligations sont entre 104 et 383 pb.
- Notation (cran) est la caractéristique qui explique le plus la dispersion des spreads, à elle seule (84 %).
- La lettre de notation seule explique 79 % de la dispersion. Une fois la notation connue, l'émetteur explique encore 39 % de ce qui reste : c'est pourquoi l'estimation suit la courbe propre à chaque émetteur quand elle existe.
- La catégorie la plus représentée est BBB (37 % de l'échantillon) : les estimations y reposent sur plus de comparables que dans les catégories rares.

### Spread par grande notation (pb)

| Notation | Obligations | 10e centile | 25e centile | Médiane | 75e centile | 90e centile |
|---|---|---|---|---|---|---|
| AAA | 59,0 | 21,3 | 24,0 | 27,4 | 33,4 | 37,3 |
| AA | 249 | 28,6 | 35,9 | 44,7 | 55,7 | 65,4 |
| A | 1 012 | 63,0 | 76,9 | 95,6 | 122 | 149 |
| BBB | 1 481 | 120 | 150 | 202 | 282 | 364 |
| BB | 930 | 268 | 342 | 461 | 624 | 953 |
| B | 248 | 550 | 674 | 1 029 | 1 715 | 2 655 |
| CCC et moins | 21,0 | 2 568 | 3 059 | 4 364 | 7 452 | 9 304 |

### Part de la dispersion des spreads expliquée par chaque caractéristique seule

| Caractéristique | Part expliquée (%) |
|---|---|
| Notation (cran) | 84,5 |
| Notation (lettre) | 79,0 |
| Émetteur | 73,8 |
| Séniorité | 26,5 |
| Secteur | 4,9 |
| Maturité (5 tranches) | 2,6 |
| Pays | 0,6 |
| Devise | 0,2 |

## 2. Les modèles un par un

### Courbe des pairs

- La courbe des pairs explique 95 % de la dispersion des spreads (en log) à elle seule, sans rien savoir de l'émetteur.
- À 5 ans, pour une obligation SENIOR_UNSECURED en EUR du secteur Consommation : 86 pb en A, 176 pb en BBB, 368 pb en BB.
- Pour un BBB, passer de 2 à 10 ans fait passer le spread de 137 à 203 pb.
- Les courbes sont tracées pour une obligation typique de l'échantillon : changer la séniorité, le secteur ou la devise les décale vers le haut ou le bas.

### Écart propre à l'émetteur

- k vaut 1,05 : avec 1 obligation, l'écart de l'émetteur est retenu à 49 % ; avec 3, à 74 % ; avec 10, à 91 %.
- Deux obligations d'un même émetteur s'écartent typiquement de ±15 % entre elles ; un émetteur s'écarte typiquement de ±15 % de la courbe des pairs. Plus le second chiffre domine, plus on fait confiance à l'émetteur, même avec peu d'obligations.
- L'écart retenu le plus fort est celui de Émetteur 0094 : 41 % par rapport à ses pairs, sur 2 obligation(s).

### Choix des comparables

- 200 combinaisons essayées sur 800 obligations ; sur 800 autres obligations, l'erreur typique passe de 15,4 % à 14,9 %.
- Réglage adopté.
- Critère le plus pesant : maturité (36 %). 40 comparables au plus, largeur de noyau 0,74.
- Ces poids varient beaucoup d'un jeu de données à l'autre : ils mesurent ce qui aide à prédire ici, pas une vérité de marché.

| Critère | Poids |
|---|---|
| notch | 0,140 |
| maturity | 0,359 |
| sector | 0,188 |
| country | 0,172 |
| size | 0,142 |

Nombre maximal de comparables : 40 ; largeur du noyau : 0,74.

### CatBoost (modèle B)

- Effet moyen = de combien, en moyenne, une variable déplace le spread prédit d'une obligation par rapport à l'obligation moyenne.
- Les trois variables les plus utilisées : notation (67 %), rang de séniorité (15 %), maturité (14 %).
- Effet de la notation seule : -85 % en AAA, 1 015 % en CCC-. La contrainte de monotonie garantit que, pour une obligation donnée, dégrader la notation ne baisse jamais le spread prédit ; les moyennes par notation peuvent légèrement fluctuer là où les obligations sont rares.

| Variable | Effet moyen sur le spread (%) |
|---|---|
| Notation | 66,9 |
| Rang de séniorité | 14,8 |
| Maturité | 13,9 |
| Secteur | 6,5 |
| Taille | 3,0 |
| Pays | 2,4 |
| Séniorité (catégorie) | 2,3 |
| Remboursable | 2,2 |
| Devise | 2,1 |
| Type de coupon | 0,1 |

### Correcteur (modèle C)

- Le correcteur modifie l'estimation du modèle A de 2,7 % en médiane ; 90 % des corrections sont entre -7,6 % et 8,4 %.
- Hors échantillon, il améliore le modèle A dans 51 % des cas.
- Erreur médiane : 11,2 % pour A, 11,3 % pour C.

### Forêt aléatoire

- Importance = de combien l'erreur de la forêt augmente quand on brouille une variable (mesuré sur des émetteurs qu'elle n'a jamais vus).
- Variables dont la forêt dépend le plus : notation (+70,7 points d'erreur), maturité (+6,8 points d'erreur), rang de séniorité (+4,9 points d'erreur).
- Variables quasi inutiles pour elle : type de coupon, secteur.

| Variable | Hausse de l'erreur si brouillée (points de %) |
|---|---|
| Notation | 70,7 |
| Maturité | 6,8 |
| Rang de séniorité | 4,9 |
| Séniorité (catégorie) | 0,3 |
| Taille | 0,3 |
| Pays | 0,1 |
| Devise | 0,1 |
| Remboursable par anticipation | 0,1 |
| Type de coupon | 0,0 |
| Secteur | -0,4 |

### Effets comparés

- Chaque courbe montre le spread moyen prédit si l'on change une seule caractéristique de toutes les obligations, le reste restant tel quel.
- Courbes proches : les modèles ont appris la même chose. Une forêt aléatoire produit des paliers, un modèle de régression des courbes lisses.

## 3. Comparaison des modèles et confiance

### Erreur médiane hors échantillon (pb), par palier et par modèle

| Palier | Obligations testées | A · Comparables | B · CatBoost | C · Comparables corrigés | Courbe des pairs | Forêt aléatoire |
|---|---|---|---|---|---|---|
| Complexe | 410 | 116 | 134 | 118 | 122 | 134 |
| Intermédiaire | 1 818 | 14,2 | 19,3 | 14,3 | 19,4 | 19,9 |
| Simple | 1 262 | 11,1 | 18,4 | 10,2 | 21,3 | 15,8 |
| Émetteur inconnu | 3 600 | 24,8 | 24,6 | 25,0 | 24,1 | 28,0 |

- Erreur médiane = la moitié des obligations sont estimées avec une erreur plus faible. Chaque obligation est estimée sans avoir servi à l'apprentissage.
- Complexe : meilleur modèle C · Comparables corrigés (117,8 pb).
- Intermédiaire : meilleur modèle A · Comparables (14,2 pb).
- Simple : meilleur modèle C · Comparables corrigés (10,2 pb).
- Émetteur inconnu : meilleur modèle Courbe des pairs (24,1 pb).
- La forêt aléatoire bat le modèle A dans 0 palier(s) sur 4. Elle n'utilise pas la courbe de l'émetteur : elle ne peut gagner que là où l'émetteur apporte peu.
- Modèle retenu par palier dans l'estimation : Complexe → A, Émetteur inconnu → A, Intermédiaire → A, Simple → A.

### Score de confiance : erreur par tranche de difficulté prévue

| Tranche (1 = plus facile) | Erreur prévue (%) | Erreur réalisée (%) |
|---|---|---|
| 1 | 5,2 | 5,5 |
| 2 | 6,4 | 7,6 |
| 3 | 7,4 | 9,1 |
| 4 | 8,7 | 9,3 |
| 5 | 9,8 | 11,9 |
| 6 | 10,5 | 13,2 |
| 7 | 11,1 | 13,6 |
| 8 | 11,8 | 14,4 |
| 9 | 12,8 | 17,2 |
| 10 | 14,9 | 20,6 |

### Intervalle à 90 % : couverture par palier

| Palier | Obligations | Couverture (%) | Largeur médiane (%) |
|---|---|---|---|
| Complexe | 410 | 90,2 | 93,3 |
| Émetteur inconnu | 3 600 | 90,0 | 70,3 |
| Intermédiaire | 1 818 | 90,1 | 47,7 |
| Simple | 1 262 | 90,1 | 39,2 |

- Les 10 % de cas jugés les plus faciles ont une erreur médiane de 5,5 % ; les 10 % les plus difficiles, 20,6 %.
- Le classement par difficulté est utile (corrélation de rang 0,34 entre difficulté prévue et erreur réalisée).
- Couverture visée : 90 %. Obtenue : complexe 90 %, émetteur inconnu 90 %, intermédiaire 90 %, simple 90 %.

## 4. Clustering

Le clustering regroupe les obligations par leurs caractéristiques (notation, maturité, séniorité, taille, secteur, devise), sans le spread. Il n'intervient pas dans l'estimation.

### Combien de familles ? (silhouette du k-means)

| Familles | Silhouette |
|---|---|
| 2 | 0,353 |
| 3 | 0,214 |
| 4 | 0,204 |
| 5 | 0,162 |
| 6 | 0,158 |
| 7 | 0,150 |
| 8 | 0,143 |
| 9 | 0,130 |
| 10 | 0,131 |

- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).

### Accord entre méthodes (indice de Rand ajusté)

| Méthode | k-means | Hiérarchique (Ward) | Mélange gaussien | HDBSCAN (densité) |
|---|---|---|---|---|
| k-means | 1,00 | 0,47 | 0,01 | 0,00 |
| Hiérarchique (Ward) | 0,47 | 1,00 | 0,02 | 0,01 |
| Mélange gaussien | 0,01 | 0,02 | 1,00 | 0,08 |
| HDBSCAN (densité) | 0,00 | 0,01 | 0,08 | 1,00 |

### Clustering · k-means

- k-means découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 55 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 88 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : hiérarchique (ward) 0,47, mélange gaussien 0,01, hdbscan (densité) 0,00.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout notation et séniorité, l'axe vertical maturité et séniorité. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F3 (A · 7 ans · multi-secteurs, 1310 obligations, spread médian 101 pb). Son estimation de 93 pb est au-dessus de 43 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | BBB · 2 ans · multi-secteurs | 904 | 344 | BBB | 2,2 | Consommation (13 %) | 85,7 | 148 | 249 |
| F2 | BB · 7 ans · multi-secteurs | 1 315 | 260 | BB+ | 7,1 | Technologie (15 %) | 255 | 352 | 510 |
| F3 | A · 7 ans · multi-secteurs | 1 310 | 274 | A | 7,1 | Consommation (14 %) | 71,7 | 101 | 138 |
| F4 | BB · 6 ans · multi-secteurs · subordonnée | 471 | 256 | BB+ | 5,7 | Assurance (19 %) | 360 | 724 | 1 356 |

### Clustering · Hiérarchique (Ward)

- Hiérarchique (Ward) découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 49 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 86 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means 0,47, mélange gaussien 0,02, hdbscan (densité) 0,01.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout notation et séniorité, l'axe vertical maturité et séniorité. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F1 (A · 6 ans · multi-secteurs, 1670 obligations, spread médian 103 pb). Son estimation de 93 pb est au-dessus de 42 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | A · 6 ans · multi-secteurs | 1 670 | 352 | A- | 6,4 | Consommation (16 %) | 71,5 | 103 | 151 |
| F2 | BB · 6 ans · multi-secteurs · subordonnée | 588 | 276 | BB+ | 6,0 | Assurance (28 %) | 330 | 617 | 1 202 |
| F3 | BB · 6 ans · multi-secteurs | 1 322 | 267 | BB+ | 6,0 | Technologie (18 %) | 219 | 323 | 474 |
| F4 | BBB · 1 ans · multi-secteurs | 420 | 216 | BBB- | 1,5 | Assurance (15 %) | 97,4 | 182 | 308 |

### Clustering · Mélange gaussien

- Mélange gaussien découpe l'échantillon en 4 familles.
- Pas de structure en groupes nette : la silhouette baisse à chaque famille ajoutée. Les obligations forment plutôt un continuum ; le nombre de familles est donc un choix de lecture (4 par défaut), pas une découverte. La séparation la plus nette est en 2 familles (silhouette 0,35).
- Les familles expliquent 0 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 79 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means 0,01, hiérarchique (ward) 0,02, hdbscan (densité) 0,08.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout notation et séniorité, l'axe vertical maturité et séniorité. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F4 (BBB · 6 ans · multi-secteurs, 555 obligations, spread médian 220 pb). Son estimation de 93 pb est au-dessus de 20 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| F1 | BBB · 6 ans · multi-secteurs | 1 646 | 292 | BBB | 5,6 | Santé (21 %) | 112 | 199 | 376 |
| F2 | BBB · 6 ans · multi-secteurs | 495 | 191 | BBB | 5,6 | Technologie (24 %) | 118 | 210 | 420 |
| F3 | BBB · 6 ans · multi-secteurs | 1 304 | 237 | BBB+ | 5,5 | Services publics (23 %) | 94,4 | 172 | 364 |
| F4 | BBB · 6 ans · multi-secteurs | 555 | 198 | BBB | 5,8 | Consommation (29 %) | 104 | 220 | 428 |

### Clustering · HDBSCAN (densité)

- HDBSCAN trouve 18 zones denses et 1304 obligations atypiques.
- HDBSCAN choisit seul son nombre de familles ; les obligations hors des zones denses sont classées atypiques, ce qui ne veut pas dire mal valorisées.
- Les familles expliquent 13 % de la dispersion des spreads, contre 79 % pour la lettre de notation seule, et 83 % pour les deux combinées.
- Accord avec les autres méthodes (indice de Rand ajusté, 1 = identiques, 0 = hasard) : k-means 0,00, hiérarchique (ward) 0,01, mélange gaussien 0,08.
- La carte montre 59 % de l'information : l'axe horizontal résume surtout notation et séniorité, l'axe vertical maturité et séniorité. Deux points proches sur la carte peuvent être éloignés en réalité.
- Votre obligation appartient à la famille F15 (BBB · 6 ans · Télécoms, 186 obligations, spread médian 150 pb). Son estimation de 93 pb est au-dessus de 32 % de sa famille.

| Famille | Description | Obligations | Émetteurs | Notation médiane | Maturité médiane (ans) | Secteur principal | Spread 25e centile | Spread médian | Spread 75e centile |
|---|---|---|---|---|---|---|---|---|---|
| Atypique | Obligations atypiques | 1 304 | 398 | BBB | 4,9 | Assurance (14 %) | 89,5 | 237 | 536 |
| F1 | BB · 6 ans · multi-secteurs · subordonnée | 127 | 84 | BB+ | 5,5 | Assurance (39 %) | 334 | 629 | 910 |
| F2 | BB · 6 ans · multi-secteurs · subordonnée | 57 | 46 | BB | 5,8 | Santé (26 %) | 731 | 1 048 | 1 706 |
| F3 | BBB · 5 ans · Assurance | 127 | 63 | BBB- | 5,1 | Assurance (54 %) | 257 | 421 | 544 |
| F4 | BBB · 6 ans · multi-secteurs | 97 | 82 | BBB+ | 5,8 | Services publics (14 %) | 78,6 | 120 | 166 |
| F5 | BBB · 6 ans · Technologie | 43 | 27 | BBB | 5,9 | Technologie (100 %) | 133 | 191 | 400 |
| F6 | BBB · 5 ans · Énergie | 54 | 32 | BBB+ | 5,1 | Énergie (59 %) | 104 | 147 | 203 |
| F7 | BBB · 6 ans · Assurance | 114 | 49 | BBB | 6,1 | Assurance (100 %) | 115 | 226 | 300 |
| F8 | BBB · 6 ans · Banques | 106 | 40 | BBB+ | 5,7 | Banques (100 %) | 109 | 187 | 316 |
| F9 | BBB · 5 ans · Santé | 266 | 46 | BBB | 5,4 | Santé (100 %) | 120 | 162 | 231 |
| F10 | BBB · 6 ans · Immobilier | 192 | 41 | BBB+ | 5,8 | Immobilier (100 %) | 120 | 177 | 430 |
| F11 | BBB · 7 ans · Santé | 52 | 25 | BBB | 7,0 | Santé (100 %) | 142 | 207 | 255 |
| F12 | BBB · 6 ans · Consommation | 93 | 32 | BBB+ | 6,2 | Consommation (100 %) | 114 | 158 | 304 |
| F13 | BBB · 6 ans · Technologie | 259 | 52 | BBB | 6,1 | Technologie (100 %) | 143 | 221 | 365 |
| F14 | BBB · 6 ans · Énergie | 185 | 40 | BBB+ | 6,2 | Énergie (100 %) | 105 | 160 | 283 |
| F15 | BBB · 6 ans · Télécoms | 186 | 40 | BBB | 5,7 | Télécoms (100 %) | 86,2 | 150 | 270 |
| F16 | BBB · 6 ans · Consommation | 269 | 51 | BBB+ | 6,1 | Consommation (100 %) | 96,7 | 152 | 302 |
| F17 | A · 6 ans · Services publics | 228 | 47 | A- | 6,1 | Services publics (100 %) | 82,8 | 124 | 209 |
| F18 | BBB · 6 ans · Industrie | 241 | 55 | BBB | 6,0 | Industrie (100 %) | 64,2 | 214 | 358 |

## 5. Obligation estimée

| Émetteur | Notation | Séniorité | Maturité (ans) | Devise | Secteur |
|---|---|---|---|---|---|
| Émetteur 0138 | A+ | SENIOR_UNSECURED | 6,9 | USD | Télécoms |

- Z-Spread juste estimé : 93 pb ; intervalle à 90 % : 80 à 108 pb.
- Confiance : 96/100 (note A) ; erreur typique attendue 5 pb.
- Spread observé dans la base pour cette obligation : 100 pb. Il n'a pas servi à l'estimation (l'obligation est exclue de ses propres comparables).
- Palier : Simple. 5 obligations de l'émetteur, même devise et séniorité, encadrant la maturité (5.0 à 8.8 ans).
- Modèle retenu : A (comparables). Estimations des modèles (pb) : courbe des pairs 83, A 93, B (CatBoost) 80, C (comparables corrigés) 91, lecture descendante 92.

### Lecture descendante (contrôle)

- 92 pb ; niveau : même émetteur, même devise et séniorité ; moyenne des 2 points les plus proches de chaque côté ; écart de -1,1 % avec l'estimation.

| Émetteur | Notation | Séniorité | Devise | Maturité (ans) | Spread observé (pb) | Spread corrigé (pb) |
|---|---|---|---|---|---|---|
| Émetteur 0138 | A+ | SENIOR_UNSECURED | USD | 6,4 | 92,2 | 87,4 |
| Émetteur 0138 | A+ | SENIOR_UNSECURED | USD | 5,1 | 87,4 | 89,7 |
| Émetteur 0138 | A+ | SENIOR_UNSECURED | USD | 7,8 | 101 | 95,0 |
| Émetteur 0138 | A+ | SENIOR_UNSECURED | USD | 8,8 | 94,1 | 93,8 |

### Comparables

- 40 comparables retenus ; les 5 plus proches pèsent 13 % du total.
- Une fois ramenés aux caractéristiques de votre obligation, 80 % d'entre eux donnent entre 74 et 103 pb.
- L'émetteur a 5 autre(s) obligation(s) : sa propre courbe pèse 83 %, les comparables 17 %.

### Contrôles

| Contrôle | Résultat | Détail |
|---|---|---|
| Spread positif | ok | 92.8 pb |
| Plage du segment de notation | ok | Estimation 93 pb ; plage 1 %-99 % des notations voisines : 30 à 155 pb. |
| Accord champion-challenger | ok | Écart relatif entre les deux modèles : 15 % (en log). |
| Comparables suffisants | ok | Nombre effectif de comparables hors émetteur : 39.9 ; poids de la courbe émetteur : 0.83. |
| Accord avec la lecture descendante | ok | Lecture descendante 92 pb (même émetteur, même devise et séniorité) ; écart -1 % avec l'estimation. |

### Cascade (de l'obligation moyenne à l'estimation)

| Étape | Effet (pb) | Cumul (pb) |
|---|---|---|
| Obligation moyenne de l'univers | — | 206 |
| Notation | -124 | 82,5 |
| Maturité | 13,9 | 96,4 |
| Séniorité | -12,8 | 83,6 |
| Secteur | -3,8 | 79,8 |
| Devise | 4,0 | 83,8 |
| Pays | 3,0 | 86,8 |
| Type de coupon | 0,1 | 86,9 |
| Option de remboursement anticipé | -0,7 | 86,2 |
| Taille de la souche | -3,3 | 82,9 |
| Écart émetteur (5 obligation(s), retenu à 83 %, pente de maturité comprise) | 9,0 | 91,8 |
| Comparables hors émetteur | 1,0 | 92,8 |

### Famille de l'obligation selon chaque méthode

- k-means : famille F3 (A · 7 ans · multi-secteurs), estimation au-dessus de 43 % de la famille.
- Hiérarchique (Ward) : famille F1 (A · 6 ans · multi-secteurs), estimation au-dessus de 42 % de la famille.
- Mélange gaussien : famille F4 (BBB · 6 ans · multi-secteurs), estimation au-dessus de 20 % de la famille.
- HDBSCAN (densité) : famille F15 (BBB · 6 ans · Télécoms), estimation au-dessus de 32 % de la famille.

### Réserves de l'estimation

- L'estimation repose uniquement sur les données de la base : liquidité réelle, perspective de notation et actualité de l'émetteur ne sont pas observées.
- Aucune prime de nouvelle émission n'est ajoutée.

## Limites connues du système

- Les comparables et les modèles n'observent que la base : liquidité réelle, perspective de notation et actualité de l'émetteur sont invisibles.
- Aucune prime de nouvelle émission n'est ajoutée.
- Le code suppose une photographie à une date ; aucune dynamique temporelle n'est modélisée.
- Les poids des comparables appris varient d'un jeu de données à l'autre ; ils ne s'interprètent pas un par un.
