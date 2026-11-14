"""Analyses de compréhension : échantillon, modèles, clustering.

Chaque fonction renvoie un dictionnaire avec :
- `findings` : constats en français, chiffres calculés (aucun texte libre) ;
- `tables` : petits tableaux résumés, sérialisables en JSON ;
- `points` : données détaillées pour les graphiques (DataFrame), jamais
  envoyées à l'agent.

Ces analyses n'influencent pas l'estimation : elles l'expliquent.
"""
