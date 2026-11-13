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
   estimation à partir de douze indicateurs : nombre effectif de comparables,
   distance moyenne, poids de l'émetteur, désaccord entre modèles, densité du
   segment, notation, maturité, etc.
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
`lambda`.

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
