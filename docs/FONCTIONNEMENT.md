# Fonctionnement pas à pas

Ce document décrit ce que fait le programme, dans l'ordre, et ce qui est
appris sur les données plutôt que fixé à la main.

## Vocabulaire

- **Univers** : les obligations de la base, avec leur Z-Spread observé.
- **Cible** : l'obligation dont on veut le Z-Spread juste.
- **Log du spread** : tous les calculs se font sur le logarithme du spread.
  Un écart de 0,10 en log vaut environ 10 % de spread.
- **Résidu** : écart, en log, entre le spread observé d'une obligation et
  celui que donne la courbe des pairs pour son profil.

## Phase 1 : préparation, au démarrage

Elle dure environ 5 secondes sans CatBoost et 50 secondes avec, pour
4 000 obligations.

### 1. Lecture et traduction

L'univers est lu dans PostgreSQL. `config/schema_map.yaml` traduit les noms de
colonnes et les valeurs réelles vers le schéma interne. Si la base donne une
date de maturité, la maturité résiduelle en années est calculée.

### 2. Nettoyage

- La notation (S&P, Fitch ou Moody's) devient un cran numérique : AAA = 1.
- La séniorité devient un rang : 0 pour la dette garantie, 4 pour l'AT1.
- Les lignes inexploitables sont écartées et comptées : notation inconnue,
  maturité ou spread non positif.

### 3. Courbe des pairs

Une régression explique le log du spread par la notation, la maturité, leur
interaction, la séniorité, le secteur, la devise, le type de coupon et
l'option de remboursement anticipé. Elle donne le spread « normal » d'une
obligation d'un profil donné. Ses coefficients sont appris sur les données.

La version enrichie (activée par défaut, `peer_curve.enriched`) ajoute des
termes qui restent lisibles dans la cascade :

- une courbure de maturité : les courbes s'aplatissent sur les longues
  maturités ;
- une sensibilité à la notation propre à chaque secteur : une dégradation ne
  coûte pas la même chose à une foncière et à un service public ;
- une prime de séniorité qui varie avec la notation : la dette subordonnée
  d'un émetteur faible coûte plus cher ;
- le pays de l'émetteur et la taille de la souche (liquidité).

### 4. Résidus

Pour chaque obligation, le programme calcule ce que la courbe n'explique pas.
Ces résidus portent l'information propre à chaque émetteur et à chaque
obligation.

### 5. Poids de la courbe émetteur (appris)

L'écart d'un émetteur à la courbe est retenu à hauteur de
`lambda = n / (n + k)`, où `n` est son nombre d'obligations. Le coefficient
`k` est le rapport entre deux variances mesurées sur les résidus :

    k = variance du bruit entre obligations d'un même émetteur
        / variance des écarts entre émetteurs

Si les émetteurs diffèrent beaucoup entre eux et que leurs obligations sont
cohérentes, `k` est petit : une seule obligation suffit à faire confiance à
l'émetteur. C'est une formule statistique (analyse de variance), pas du ML.

La **pente de maturité propre à l'émetteur** (activée par défaut,
`comparables.issuer_slope`) suit la même logique. Pour chaque émetteur, le
programme mesure si ses résidus montent ou descendent avec la maturité. Cette
pente est retenue à hauteur de `Sxx / (Sxx + k_b)`, où `Sxx` mesure l'étendue
des maturités de l'émetteur : des obligations toutes proches en maturité ne
disent rien de la pente. `k_b` est estimé par les moments, en comparant la
dispersion des pentes observées à leur bruit d'estimation. Sur des données
sans pente propre, `k_b` devient très grand et la pente est neutralisée.

### 6. Réglage des comparables (appris)

Trois paramètres déterminent quelles obligations servent de comparables :
les poids des cinq critères de distance, la largeur du noyau et le nombre de
comparables retenus. Le programme essaie 200 combinaisons et garde la
meilleure, avec quatre garde-fous :

1. le réglage se fait sur 10 % des obligations ;
2. le résultat n'est adopté que s'il bat les valeurs par défaut sur 10 %
   d'autres obligations, jamais utilisées pour le réglage ;
3. chaque critère garde un poids d'au moins 5 % ;
4. les obligations de réglage sont exclues du calcul des intervalles.

### 7. Entraînement des modèles CatBoost (appris, optionnel)

- **Modèle B** : prédit le log du spread à partir des caractéristiques, sans
  l'émetteur, avec une contrainte de monotonie sur la notation et la séniorité.
  Il est indépendant du modèle A et sert de contrôle.
- **Modèle C** : prédit l'erreur du modèle A, à partir des caractéristiques et
  de la qualité des comparables. Pour l'entraîner, le modèle A estime chaque
  obligation de l'univers sans elle-même, dans deux situations : émetteur
  connu et émetteur retiré. Le modèle C est donc le modèle A corrigé.

### 8. Backtest interne

Chaque obligation est estimée sans avoir servi à l'apprentissage, sous deux
schémas :

- **émetteur connu** : des obligations sont retirées au hasard ;
- **émetteur inconnu** : des émetteurs entiers sont retirés.

Le backtest produit trois choses :

1. **Le champion par palier.** Le modèle A reste champion, sauf si un modèle
   CatBoost réduit l'erreur médiane d'au moins 5 %.
2. **Le modèle de difficulté (appris).** Il prédit l'erreur attendue d'une
   estimation à partir de quinze indicateurs : nombre effectif de comparables,
   distance moyenne, poids de l'émetteur, désaccord entre modèles, densité du
   segment, notation, maturité, etc., et (`confidence.dispersion_features`)
   le désaccord des comparables entre eux et la dispersion des obligations de
   l'émetteur autour de sa courbe.
3. **La largeur de l'intervalle à 90 %**, par prédiction conforme.

## Phase 2 : estimation, à chaque demande

### 1. Identification de la cible

La cible est donnée par son ISIN, si elle est dans l'univers, ou par ses
caractéristiques. Une obligation de l'univers est retirée de ses propres
comparables.

### 2. Spread de base

La courbe des pairs est appliquée au profil de la cible.

### 3. Écart émetteur

Le programme calcule la moyenne des résidus des autres obligations de
l'émetteur, dans la même devise si possible, et la retient à hauteur de
`lambda`. S'y ajoute la pente de maturité de l'émetteur, rétrécie, appliquée à
l'écart entre la maturité de la cible et la maturité moyenne de l'émetteur.
L'extrapolation est limitée à 0,3 en log au-delà des maturités de l'émetteur.

### 4. Écart des comparables

1. **Filtres durs** : même devise, même séniorité, même type de coupon, hors
   émetteur. S'il reste moins de 5 candidats, les filtres sont relâchés un à
   un, et le relâchement est signalé.
2. **Distance** : somme pondérée de cinq écarts normalisés (notation,
   maturité, secteur, pays, taille).
3. **Sélection** : les plus proches sont retenus, pondérés par leur proximité.
4. **Écart** : moyenne pondérée de leurs résidus.

Utiliser les résidus revient à translater chaque comparable au profil de la
cible : un comparable à 3 ans n'est pas comparé tel quel à une cible à 10 ans.

### 5. Modèle A

    log du spread = spread de base
                  + lambda × écart émetteur
                  + (1 − lambda) × écart des comparables

### 6. Modèles B et C

CatBoost donne une estimation indépendante (B) et une correction du modèle A (C).

### 7. Choix du chiffre final

Le champion du palier donne l'estimation. L'écart entre A et B alimente un
contrôle : au-delà de 25 %, le cas est signalé.

### 8. Intervalle et confiance

- Le modèle de difficulté prédit l'erreur typique de ce cas.
- L'intervalle à 90 % est proportionnel à cette difficulté : étroit pour un
  cas facile, large pour un cas difficile.
- Le score sur 100 est la part des cas du backtest jugés plus difficiles.
  Un score de 80 signifie : plus facile que 80 % des cas testés.

### 9. Contrôles

Spread positif, plage des notations voisines, cohérence avec la dette plus
senior du même émetteur, accord entre modèles, nombre de comparables.

### 10. Explication et audit

- Cascade en points de base, de l'obligation moyenne à l'estimation.
- Tableau des comparables, contributions SHAP des modèles CatBoost.
- Récit en français, construit sans LLM.
- Enregistrement complet dans le journal, rejouable par son identifiant.

## Phase 3 : retour utilisateur

1. Le gérant saisit le spread auquel il a traité (`trade`, poids 1) ou son
   avis (`opinion`, poids 0,3).
2. Le retour devient une observation pondérée de l'univers.
3. La courbe des pairs, les résidus et `k` sont recalculés aussitôt.
4. Les modèles CatBoost, les poids de comparables, le champion et les
   intervalles ne changent qu'au prochain démarrage complet.

## Appris ou fixé

| Paramètre | Statut | Comment |
|---|---|---|
| Coefficients de la courbe des pairs | Appris | Régression |
| `k` du poids émetteur | Appris | Analyse de variance |
| `k_b` de la pente émetteur | Appris | Méthode des moments sur les pentes par émetteur |
| Poids de distance, noyau, nombre de comparables | Appris | Recherche sur le backtest, avec validation |
| Modèles B et C | Appris | CatBoost |
| Champion par palier | Appris | Backtest, avec marge de 5 % |
| Score de confiance et largeur d'intervalle | Appris | Modèle de difficulté et prédiction conforme |
| Échelles de normalisation des distances | Fixé | 6 crans, 1,5 en log de maturité |
| Poids minimal d'un critère, marge du champion | Fixé | Garde-fous |
| Seuils des paliers et des contrôles | Fixé | Règles métier, à valider par le BA |
| Poids des retours (1 et 0,3) | Fixé | Jugement ; trop peu de retours pour apprendre |

## Effet mesuré de chaque amélioration

Mesures sur obligations mises de côté, données synthétiques
(`python scripts/ablation.py --ml`). Elles valident la mécanique, pas la
précision sur données réelles.

### Deux profils de données synthétiques

- **Simple** (`make_synthetic.py --simple`) : modèle log-linéaire proche de
  la courbe des pairs d'origine. Il favorise le modèle A par construction.
- **Réaliste** (par défaut) : il ajoute des courbes qui s'aplatissent en haut
  rendement, une sensibilité à la notation propre au secteur, une prime de
  subordination plus forte pour les émetteurs faibles, une pente de maturité
  propre à chaque émetteur, un effet pays, une prime d'illiquidité des petites
  souches, un bruit plus fort pour les notations faibles et la dette
  subordonnée, et 4 % de notations en retard sur le marché.

### Dernières améliorations (profil réaliste)

| Étape | Erreur médiane, émetteur connu | Erreur médiane, émetteur inconnu | Largeur de l'intervalle (émetteur connu) |
|---|---|---|---|
| Version précédente | 18,3 pb (9,8 %) | 32,4 pb (16,5 %) | 56 % |
| + courbe des pairs enrichie | 15,8 pb (9,0 %) | 28,2 pb (15,2 %) | 53 % |
| + pente de maturité par émetteur | 15,5 pb (8,4 %) | 28,2 pb (15,2 %) | 50 % |
| + dispersion dans la confiance | 15,5 pb (8,4 %) | 28,2 pb (15,2 %) | 49 % |
| + CatBoost (B et C) | Aucun gain | Aucun gain | — |

- La courbe enrichie fait le gros du gain : −14 % d'erreur en pb pour un
  émetteur connu, −13 % pour un émetteur inconnu, −20 % sur la dette
  subordonnée.
- La pente par émetteur gagne environ 3 % d'erreur relative (6 % dans le
  palier simple), de façon régulière sur quatre tirages différents.
- La dispersion ne change pas l'estimation, seulement l'intervalle : 3 % plus
  étroit à couverture égale, et un score un peu mieux classé (corrélation de
  rang −0,28 à −0,31).
- Sur le profil simple, ces ajouts ne gagnent rien (variations de ±4 %, dans
  le bruit) et la pente émetteur est neutralisée : ils ne dégradent pas un
  marché qui n'a pas ces effets.

### Premières améliorations

| Amélioration | Effet avec bruit uniforme | Effet avec bruit croissant selon la notation |
|---|---|---|
| `k` estimé | Émetteur connu : erreur médiane de 11,9 à 9,8 pb, intervalle 14 % plus étroit | Aucun : le `k` estimé (2,8) est proche de l'ancien (2) |
| Poids de distance réglés | Émetteur inconnu : erreur médiane de 23,5 à 21,4 pb | Émetteur inconnu : de 34,2 à 31,0 pb |
| Intervalle adaptatif | Aucun | Émetteur inconnu : couverture de 0,885 à 0,905 ; couverture rééquilibrée entre spreads serrés et larges |
| CatBoost (B et C) | Aucun gain | Aucun gain ; jamais retenu avec la marge de 5 % |

Lecture honnête :

- Le gain de `k` et celui des poids de distance sont nets mais modestes.
- Les poids réglés changent beaucoup d'un jeu de données à l'autre. Le gain
  est réel, mais les poids eux-mêmes ne s'interprètent pas.
- Le modèle de difficulté n'apporte rien quand toutes les obligations sont
  également prévisibles, ce qui est le cas des données synthétiques par
  défaut. Son intérêt dépend des données réelles.
- CatBoost n'apporte rien ici, parce que les données synthétiques sont
  générées par un modèle que la courbe des pairs reproduit déjà. C'est
  attendu et ne préjuge pas du réel.
