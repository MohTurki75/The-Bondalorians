# Architecture

## Cas d'usage

Un broker propose une obligation avec un Z-Spread. Le gérant demande à l'agent
le Z-Spread juste de cette obligation et compare lui-même. L'agent ne reçoit
pas le spread du broker et ne calcule ni prix ni rendement.

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
| Le backtest désigne le champion par palier | Pas de préférence a priori entre les deux modèles |
| Deux schémas de validation | L'erreur sur émetteur jamais vu est bien plus forte que sur émetteur connu |
| Façade indépendante de MCP | La démo survit à une panne du serveur ou du LLM |

## Estimation

En log du spread :

    modèle A = courbe des pairs
             + lambda * écart moyen de l'émetteur à la courbe
             + (1 - lambda) * écart pondéré des comparables hors émetteur

- `lambda = n / (n + k)` : un émetteur avec une seule obligation pèse peu.
- Comparables : filtres durs (devise, séniorité, type de coupon), relâchés
  dans cet ordre s'il reste moins de 5 candidats ; distance de Gower pondérée
  sur notation, maturité, secteur, pays, taille ; noyau gaussien ; 15 retenus.
- Chaque comparable est translaté aux caractéristiques de la cible par la
  courbe des pairs : on ne compare pas directement un 3 ans et un 10 ans.
- Une obligation déjà dans l'univers est exclue de ses propres comparables.

Le modèle B est un CatBoost sur le log du spread, sans l'émetteur comme
variable, avec monotonie sur la notation et le rang de séniorité.

## Paliers

| Palier | Condition |
|---|---|
| Simple | Au moins 3 obligations de l'émetteur, même devise et séniorité, encadrant la maturité |
| Intermédiaire | Émetteur connu, mais peu d'obligations comparables ou maturité hors plage |
| Complexe | Émetteur absent de l'univers, ou dette subordonnée ou hybride |

## Intervalle et score de confiance

- Intervalle : prédiction conforme sur les erreurs hors échantillon, par
  palier, et à part pour l'émetteur inconnu (plis groupés par émetteur).
- La garantie de couverture suppose des observations échangeables. Pour un
  émetteur inconnu, les erreurs d'un même émetteur sont corrélées : la
  couverture mesurée varie beaucoup d'un tirage à l'autre (0,81 à 0,98 sur
  données synthétiques, 0,90 en moyenne).
- Score sur 100 : largeur de l'intervalle (40 %), qualité des comparables
  (30 %), accord entre modèles (20 %), densité du segment (10 %). Ces poids
  sont un point de départ, pas un résultat.

## Explicabilité

1. Cascade en points de base, de l'obligation moyenne de l'univers à
   l'estimation. Elle est séquentielle : le montant d'une étape dépend des
   précédentes, parce que le modèle est multiplicatif.
2. Tableau des comparables avec distance, poids et spread ajusté.
3. Contributions SHAP du CatBoost, en log (donc multiplicatives).
4. Récit déterministe, que le LLM reformule sans y ajouter de chiffre.

## Retours utilisateurs

Un retour est de type `trade` (spread réellement traité, poids 1) ou `opinion`
(avis, poids 0,3). Il devient une pseudo-obligation ajoutée à l'univers.
L'effet est immédiat sur les estimations suivantes. Les quantiles de
l'intervalle et le choix du champion ne changent qu'au prochain backtest
complet ; aucune promotion n'est automatique.

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
- **Modèle B sans écart émetteur** : il ignore la courbe de l'émetteur.
- **Prime de nouvelle émission** : non modélisée.
- **Information hors base** : liquidité réelle, perspective de notation et
  actualité ne sont pas observées.

## À faire

Avant le hackathon :

- interface de démonstration (Streamlit) branchée sur `service.py` ;
- apprentissage des poids de distance par minimisation de l'erreur de backtest ;
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
