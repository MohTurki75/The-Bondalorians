# Fiche pour les agents

- Généré le : 2026-11-23 22:20 UTC ; source : base_jour1.csv
- Mode d'emploi : Pour estimer une obligation : 1) chercher sa case dans `aiguillage` et utiliser la méthode recommandée, en multipliant la demi-largeur de l'intervalle par `facteur_intervalle` ; 2) vérifier les `poches` qui la concernent et mentionner l'ajustement suggéré ; 3) ne jamais citer un chiffre absent de cette fiche ou des outils. Toutes les erreurs sont hors échantillon (obligations estimées sans avoir servi à l'apprentissage). Les poches sont des tendances moyennes d'un segment, pas des certitudes pour une obligation.

## Règles

- **R01 (structure)** Les obligations d'un même émetteur sont cohérentes : s'appuyer fortement sur la courbe de l'émetteur. _k = 0.773 ; poids de l'émetteur avec 1, 3 et 10 obligations : [0.56, 0.8, 0.93]._
- **R02 (structure)** Variables qui déterminent le plus le spread : Notation (cran), Émetteur, Séniorité. _Parts de la dispersion expliquées seules : Notation (cran) 85 %, Émetteur 76 %, Séniorité 22 %._
- **R03 (aiguillage)** Obligation investissement (BBB- et mieux), dette subordonnée, émetteur 1 à 2 obligations : utiliser S · ancrage senior. _Erreur médiane 12.57 % contre 13.71 % (52 cas), écart confirmé sur deux moitiés d'émetteurs._
- **R04 (poche)** Émetteur inconnu, ESG ESG élevé : pour ce segment, abaisser l'estimation d'environ 10 %. _Écart médian -9.6 % (754 obligations, 116 émetteurs) ; -6.8 % puis -12.1 % sur deux moitiés d'émetteurs._
- **R05 (poche)** Émetteur inconnu, ESG ESG faible : pour ce segment, relever l'estimation d'environ 8 %. _Écart médian +8.2 % (1241 obligations, 172 émetteurs) ; +7.0 % puis +9.6 % sur deux moitiés d'émetteurs._
- **R06 (poche)** Émetteur inconnu, Pays DE · ESG ESG faible : pour ce segment, relever l'estimation d'environ 20 %. _Écart médian +20.5 % (118 obligations, 23 émetteurs) ; +18.5 % puis +22.6 % sur deux moitiés d'émetteurs._
- **R07 (poche)** Émetteur inconnu, Pays US · ESG ESG faible : pour ce segment, relever l'estimation d'environ 19 %. _Écart médian +19.0 % (84 obligations, 21 émetteurs) ; +15.7 % puis +23.4 % sur deux moitiés d'émetteurs._
- **R08 (poche)** Émetteur inconnu, Maturité > 15 ans · ESG ESG élevé : pour ce segment, abaisser l'estimation d'environ 15 %. _Écart médian -15.3 % (41 obligations, 33 émetteurs) ; -13.6 % puis -16.4 % sur deux moitiés d'émetteurs._
- **R09 (poche)** Émetteur inconnu, Séniorité SENIOR_NON_PREFERRED · ESG ESG élevé : pour ce segment, abaisser l'estimation d'environ 15 %. _Écart médian -14.5 % (40 obligations, 14 émetteurs) ; -18.2 % puis -10.7 % sur deux moitiés d'émetteurs._
- **R10 (poche)** Émetteur inconnu, Pays IT · ESG ESG faible : pour ce segment, relever l'estimation d'environ 14 %. _Écart médian +14.3 % (170 obligations, 22 émetteurs) ; +12.0 % puis +16.7 % sur deux moitiés d'émetteurs._
- **R11 (piste)** Effet non capté par la courbe des pairs : ESG ESG élevé (spread réel en dessous de l'estimation). Variable ou interaction à envisager. _Écart médian -7.8 % (726 obligations)._
- **R12 (piste)** Effet non capté par la courbe des pairs : ESG ESG faible (spread réel au-dessus de l'estimation). Variable ou interaction à envisager. _Écart médian +6.8 % (1205 obligations)._
- **R13 (piste)** Effet non capté par la courbe des pairs : Pays DE · ESG ESG faible (spread réel au-dessus de l'estimation). Variable ou interaction à envisager. _Écart médian +20.5 % (109 obligations)._
- **R14 (piste)** Effet non capté par la courbe des pairs : Pays IT · ESG ESG faible (spread réel au-dessus de l'estimation). Variable ou interaction à envisager. _Écart médian +15.4 % (165 obligations)._
- **R15 (piste)** Effet non capté par la courbe des pairs : Pays US · ESG ESG faible (spread réel au-dessus de l'estimation). Variable ou interaction à envisager. _Écart médian +17.0 % (84 obligations)._
- **R16 (colonnes)** Colonnes supplémentaires à utiliser : esg. _ESG : renseignée à 100 % ; erreur -11.5 % (émetteur connu), -10.0 % (émetteur inconnu) → à activer. Ancienneté de l'émission : renseignée à 100 % ; erreur -0.8 % (émetteur connu), -0.0 % (émetteur inconnu) → inutile. Duration : renseignée à 100 % ; erreur -5.2 % (émetteur connu), -5.3 % (émetteur inconnu) → gain suspect : fuite probable._
- **R17 (garde-fou)** Ne jamais utiliser le prix ni le rendement de l'obligation estimée pour estimer son spread. _Ils sont liés mécaniquement au spread : les utiliser reviendrait à recalculer la cotation._

## Aiguillage

| Notation | Dette | Émetteur | Cas | Erreur actuelle (%) | Meilleure méthode | Son erreur (%) | Recommandation | Couverture | Facteur intervalle |
|---|---|---|---|---|---|---|---|---|---|
| haut rendement (BB+ et moins) | senior | 1 à 2 obligations | 137 | 12,07 | A · comparables | 12,07 | estimation retenue actuelle | 0,89 | 1,00 |
| haut rendement (BB+ et moins) | senior | 3 obligations ou plus | 448 | 10,38 | A · comparables | 10,38 | estimation retenue actuelle | 0,87 | 1,00 |
| haut rendement (BB+ et moins) | senior | inconnu | 605 | 18,91 | A · comparables | 18,91 | estimation retenue actuelle | 0,89 | 1,00 |
| haut rendement (BB+ et moins) | subordonnée | 1 à 2 obligations | 69 | 16,29 | A · comparables | 16,29 | estimation retenue actuelle | 0,94 | 1,00 |
| haut rendement (BB+ et moins) | subordonnée | 3 obligations ou plus | 186 | 15,70 | A · comparables | 15,70 | estimation retenue actuelle | 0,91 | 1,00 |
| haut rendement (BB+ et moins) | subordonnée | inconnu | 263 | 20,98 | P · courbe des pairs | 20,22 | estimation retenue actuelle | 0,95 | 1,00 |
| investissement (BBB- et mieux) | senior | 1 à 2 obligations | 387 | 9,17 | A · comparables | 9,17 | estimation retenue actuelle | 0,90 | 1,00 |
| investissement (BBB- et mieux) | senior | 3 obligations ou plus | 1 151 | 6,82 | A · comparables | 6,82 | estimation retenue actuelle | 0,91 | 1,00 |
| investissement (BBB- et mieux) | senior | inconnu | 1 599 | 14,97 | P · courbe des pairs | 14,57 | estimation retenue actuelle | 0,89 | 1,00 |
| investissement (BBB- et mieux) | subordonnée | 1 à 2 obligations | 52 | 13,71 | S · ancrage senior | 12,57 | S · ancrage senior | 0,94 | 1,00 |
| investissement (BBB- et mieux) | subordonnée | 3 obligations ou plus | 175 | 8,16 | A · comparables | 8,16 | estimation retenue actuelle | 0,91 | 1,00 |
| investissement (BBB- et mieux) | subordonnée | inconnu | 233 | 19,20 | P · courbe des pairs | 18,82 | estimation retenue actuelle | 0,94 | 1,00 |

## Poches de biais de l'estimation retenue

| Situation | Segment | Obligations | Émetteurs | Écart (%) | Action |
|---|---|---|---|---|---|
| émetteur inconnu | ESG ESG élevé | 754 | 116 | -9,6 | Pour ce segment, abaisser l'estimation d'environ 10 % |
| émetteur inconnu | ESG ESG faible | 1 241 | 172 | 8,2 | Pour ce segment, relever l'estimation d'environ 8 % |
| émetteur inconnu | Pays DE · ESG ESG faible | 118 | 23 | 20,5 | Pour ce segment, relever l'estimation d'environ 20 % |
| émetteur inconnu | Pays US · ESG ESG faible | 84 | 21 | 19,0 | Pour ce segment, relever l'estimation d'environ 19 % |
| émetteur inconnu | Maturité > 15 ans · ESG ESG élevé | 41 | 33 | -15,3 | Pour ce segment, abaisser l'estimation d'environ 15 % |
| émetteur inconnu | Séniorité SENIOR_NON_PREFERRED · ESG ESG élevé | 40 | 14 | -14,5 | Pour ce segment, abaisser l'estimation d'environ 15 % |
| émetteur inconnu | Pays IT · ESG ESG faible | 170 | 22 | 14,3 | Pour ce segment, relever l'estimation d'environ 14 % |

## Effets non captés par la courbe des pairs

| Situation | Segment | Obligations | Écart (%) |
|---|---|---|---|
| émetteur connu | ESG ESG élevé | 726 | -7,8 |
| émetteur connu | ESG ESG faible | 1 205 | 6,8 |
| émetteur connu | Pays DE · ESG ESG faible | 109 | 20,5 |
| émetteur connu | Pays IT · ESG ESG faible | 165 | 15,4 |
| émetteur inconnu | ESG ESG élevé | 754 | -9,3 |
| émetteur inconnu | ESG ESG faible | 1 241 | 8,1 |
| émetteur inconnu | Pays DE · ESG ESG faible | 118 | 20,7 |
| émetteur inconnu | Pays US · ESG ESG faible | 84 | 17,0 |
| émetteur inconnu | Pays IT · ESG ESG faible | 170 | 15,5 |