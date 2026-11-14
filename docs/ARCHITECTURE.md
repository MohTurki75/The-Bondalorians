# Architecture

## Cas d'usage

Un broker propose une obligation avec un Z-Spread. Le gérant demande à l'agent
le Z-Spread juste de cette obligation et compare lui-même. L'agent ne reçoit
pas le spread du broker et ne calcule ni prix ni rendement.

Le déroulé détaillé, étape par étape, est dans
[FONCTIONNEMENT.md](FONCTIONNEMENT.md).

## Vue d'ensemble

    Client MCP ou interface
            |
    Agent LLM  -- orchestre et rédige, ne calcule rien
            |  MCP
    mcp_server/server.py  -- outils métier, pas de SQL libre
            |
    service.py  -- façade unique, utilisable sans MCP ni LLM
            |
    quant/ domain/ explain/ learning/
            |
    data/  -- schéma canonique
            |
    PostgreSQL : market (lecture seule), ref (référentiels), app (journaux)

## Décisions

| Décision | Raison |
|---|---|
| Le LLM ne calcule aucun chiffre | Auditabilité ; aucun nombre inventé |
| Schéma canonique et `schema_map.yaml` | Le schéma réel est inconnu avant le jour 1 |
| Cœur transparent, ML en challenger | 4 000 obligations : un boosting seul surapprend, et un jury crédit lit des comparables |
| Cible en log | Les spreads sont à peu près multiplicatifs entre notations |
| Le modèle transparent reste champion sauf gain d'au moins 5 % | Un modèle ML choisi sur un écart minime est choisi par hasard |
| Coefficients appris quand les données le permettent | Moins d'arbitraire, mais chaque réglage est validé hors échantillon |
| Deux schémas de validation | L'erreur sur émetteur jamais vu est bien plus forte que sur émetteur connu |
| Façade indépendante de MCP | La démo survit à une panne du serveur ou du LLM |

## Estimation

En log du spread :

    modèle A = courbe des pairs
             + lambda * écart moyen de l'émetteur à la courbe
             + (1 - lambda) * écart pondéré des comparables hors émetteur

- `lambda = n / (n + k)` : un émetteur avec une seule obligation pèse peu.
  `k` est estimé par analyse de variance des résidus (variance du bruit
  divisée par la variance entre émetteurs).
- Comparables : filtres durs (devise, séniorité, type de coupon), relâchés
  dans cet ordre s'il reste moins de 5 candidats ; distance de Gower pondérée
  sur notation, maturité, secteur, pays, taille ; noyau gaussien.
- Les poids de distance, la largeur du noyau et le nombre de comparables sont
  réglés sur le backtest (`quant/tuning.py`).
- Chaque comparable est translaté aux caractéristiques de la cible par la
  courbe des pairs : on ne compare pas directement un 3 ans et un 10 ans.
- Une obligation déjà dans l'univers est exclue de ses propres comparables.

Deux modèles CatBoost complètent le modèle A :

- **B** prédit le log du spread, sans l'émetteur comme variable, avec
  monotonie sur la notation et le rang de séniorité. Il est indépendant de A
  et sert de contrôle de désaccord.
- **C** prédit l'erreur de A à partir des caractéristiques et de la qualité
  des comparables. C'est le modèle A corrigé.

## Ce qui est appris, et ses garde-fous

| Élément appris | Méthode | Garde-fou |
|---|---|---|
| `k` | Analyse de variance | Borné entre 0,05 et 50 ; valeur forcée possible |
| Paramètres des comparables | 200 essais sur 10 % des obligations | Adoption seulement si gain sur 10 % d'autres obligations ; poids minimal de 5 % par critère ; obligations de réglage exclues des intervalles |
| Modèle C | CatBoost sur l'erreur de A | Entraîné sur des estimations faites sans l'obligation elle-même ; peu profond |
| Champion | Erreur médiane du backtest | Marge de 5 % en faveur du modèle A |
| Difficulté d'un cas | Modèle sur 12 indicateurs | Prédite hors échantillon avant de calibrer l'intervalle |

`calibration_report` (outil MCP et méthode du service) renvoie les valeurs
apprises et le gain mesuré.

## Paliers

| Palier | Condition |
|---|---|
| Simple | Au moins 3 obligations de l'émetteur, même devise et séniorité, encadrant la maturité |
| Intermédiaire | Émetteur connu, mais peu d'obligations comparables ou maturité hors plage |
| Complexe | Émetteur absent de l'univers, ou dette subordonnée ou hybride |

## Intervalle et score de confiance

- Un modèle de difficulté prédit l'erreur attendue de chaque estimation.
- Intervalle : prédiction conforme normalisée. L'erreur hors échantillon est
  divisée par la difficulté prédite ; le quantile est calculé par palier, et
  à part pour l'émetteur inconnu (plis groupés par émetteur). L'intervalle
  d'un cas vaut ce quantile multiplié par sa difficulté.
- La garantie de couverture suppose des observations échangeables. Pour un
  émetteur inconnu, les erreurs d'un même émetteur sont corrélées : la
  couverture mesurée varie beaucoup d'un tirage à l'autre (0,81 à 0,98 sur
  données synthétiques, 0,90 en moyenne).
- Score sur 100 : part des cas du backtest jugés plus difficiles que le cas
  estimé. Il n'y a plus de pondération manuelle.
- Sur données synthétiques à bruit uniforme, le score ne classe pas les
  erreurs à l'intérieur d'un palier, parce qu'il n'y a rien à classer. Avec
  un bruit croissant selon la notation, il les classe (corrélation de rang
  d'environ −0,3) et rééquilibre la couverture.

## Analyses de compréhension

Le module `analysis/` produit, pour l'interface et pour l'agent, une lecture de
l'échantillon et de chaque modèle. Il ne modifie aucune estimation.

- **Échantillon** : composition, niveaux de spread, part des écarts expliquée
  par chaque caractéristique.
- **Modèles** : courbe des pairs, écart émetteur, réglage des comparables,
  SHAP de CatBoost, corrections du modèle C, effets comparés, erreur hors
  échantillon par palier, calibration du score et de l'intervalle.
- **Forêt aléatoire** : modèle de comparaison évalué sur les plis exacts du
  backtest ; elle n'entre pas dans l'estimation.
- **Clustering** : k-means, Ward, mélange gaussien et HDBSCAN sur les mêmes
  variables normalisées ; carte en deux axes (analyse en composantes
  principales), profils de familles, accord entre méthodes, placement de
  l'obligation estimée.

Chaque analyse renvoie des constats rédigés dont tous les chiffres sont
calculés (`findings`), de petits tableaux (`tables`) et les données des
graphiques (`points`), retirées des réponses à l'agent.

## Explicabilité

1. Cascade en points de base, de l'obligation moyenne de l'univers à
   l'estimation. Elle est séquentielle : le montant d'une étape dépend des
   précédentes, parce que le modèle est multiplicatif.
2. Tableau des comparables avec distance, poids et spread ajusté.
3. Contributions SHAP des deux modèles CatBoost, en log (donc multiplicatives).
4. Récit déterministe, que le LLM reformule sans y ajouter de chiffre.

## Retours utilisateurs

Un retour est de type `trade` (spread réellement traité, poids 1) ou `opinion`
(avis, poids 0,3). Il devient une pseudo-obligation ajoutée à l'univers.
La courbe des pairs, les résidus et `k` sont recalculés aussitôt. Les modèles
CatBoost, les paramètres de comparables, le champion et les intervalles ne
changent qu'au prochain ajustement complet ; aucune promotion n'est automatique.

## Limites connues

- **Données synthétiques** : elles sont générées par un modèle log-linéaire,
  ce qui favorise le modèle A par construction. Le classement des modèles
  sera à refaire sur les données réelles.
- **Pas de dimension temporelle** : le code suppose une photographie à une
  date. S'il y a un historique, il faut choisir la date et éviter toute fuite.
- **Z-Spreads supposés présents en base** : s'il n'y a que des prix, il faut
  un pricer et des courbes de taux, absents de ce squelette.
- **Instruments complexes** : les obligations remboursables par anticipation
  sont traitées par une simple indicatrice.
- **Apport du ML non démontré** : sur données synthétiques, ni B ni C ne
  battent le modèle A. Le classement est à refaire sur données réelles.
- **Poids de comparables instables** : les poids réglés varient fortement
  d'un jeu de données à l'autre ; ils ne s'interprètent pas un par un.
- **Un seul `k` pour tout l'univers** : il ne distingue pas les segments où
  les obligations d'un émetteur sont plus ou moins dispersées.
- **Démarrage plus long** : environ 50 secondes avec CatBoost pour
  4 000 obligations.
- **Prime de nouvelle émission** : non modélisée.
- **Information hors base** : liquidité réelle, perspective de notation et
  actualité ne sont pas observées.

## À faire

Avant le hackathon :

- pilotage du serveur MCP depuis un vrai client avec le LLM retenu ;
- règles de paliers et seuils de contrôle revus par le BA ;
- taxonomie sectorielle et correspondance pays vers région dans `ref` ;
- script de promotion d'un modèle dans `app.model_registry`.

Le jour 1 :

- remplir `config/schema_map.yaml` (budget : 2 heures) ;
- profil de qualité des données, valeurs manquantes, doublons ;
- backtest sur données réelles, puis choix des paramètres.

## Questions ouvertes pour les organisateurs

- Les Z-Spreads de l'univers sont-ils en base ?
- Photographie à une date ou historique ?
- Peut-on écrire dans la base et y charger des référentiels ?
- Quel LLM est autorisé, et par quel accès ?
- Y a-t-il des obligations de test cachées, notées sur l'erreur ?
