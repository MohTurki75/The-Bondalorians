# Guide de lecture de l'onglet « Analyse »

Ce guide s'adresse à qui n'est pas analyste de données. Il explique chaque
graphique de l'onglet « Analyse », les mots techniques employés et les pièges
de lecture.

L'onglet ne change aucune estimation : il sert à comprendre l'échantillon et
les modèles. Les phrases « À retenir » sont calculées à partir des données,
jamais écrites à la main ; elles sont aussi disponibles pour l'agent par le
serveur MCP (voir la fin du guide).

Si vous avez lancé une estimation dans l'onglet « Estimation », votre
obligation apparaît sur les graphiques sous forme de losange, avec son
intervalle en trait vertical.

## Les mots à connaître

| Mot | Sens |
|---|---|
| Médiane | La valeur du milieu : la moitié des obligations est au-dessous, l'autre au-dessus. Moins sensible aux valeurs extrêmes que la moyenne. |
| Centile | Le 10e centile est la valeur sous laquelle se trouvent 10 % des obligations. |
| Échelle logarithmique | Chaque graduation multiplie au lieu d'ajouter (10, 100, 1 000). Elle permet de voir sur un même graphique un spread de 30 pb et un de 3 000 pb. |
| Hors échantillon | Une obligation est estimée par un modèle qui ne l'a pas vue pendant son apprentissage. C'est la seule façon honnête de mesurer une erreur. |
| Part de la dispersion expliquée | De 0 % à 100 % : à quel point connaître une information permet de deviner le spread. |
| Couverture | La part des cas où le vrai spread tombe dans l'intervalle annoncé. Un intervalle à 90 % doit couvrir 9 cas sur 10. |

## Partie 1 : Échantillon

| Graphique | Ce qu'il dit | Piège |
|---|---|---|
| Combien d'obligations par notation | Où les comparables sont nombreux | Une catégorie rare donne des estimations moins sûres, même si le modèle affiche un chiffre |
| Niveau des spreads par notation | La fourchette de spreads de chaque notation | Le rectangle est la moitié centrale, pas le minimum et le maximum |
| Spread, maturité et notation | Une vue de toutes les obligations | Les points se superposent : la densité est mieux lue sur le graphique précédent |
| Qu'est-ce qui explique les spreads ? | Le poids de chaque caractéristique prise seule | Les parts ne s'additionnent pas : notation et séniorité se recoupent |
| Secteur, devise, séniorité | La composition de l'échantillon | — |

## Partie 2 : Modèles un par un

L'estimation enchaîne trois étapes : courbe des pairs, écart propre à
l'émetteur, comparables. Les modèles d'apprentissage automatique servent de
contrôle et de correction.

| Graphique | Modèle | Ce qu'il dit |
|---|---|---|
| 1. Courbe des pairs | Régression | Le spread « normal » d'un profil. Les courbes sont tracées pour une obligation typique ; une autre séniorité ou un autre secteur les décalerait. |
| 2. Écart propre à l'émetteur | Rétrécissement | De combien chaque émetteur traite au-dessus ou au-dessous de ses pairs, et quelle part de cet écart est retenue. Avec peu d'obligations, l'écart est réduit par prudence. |
| 3. Comparables de votre obligation | Comparables | Les obligations retenues, leur ressemblance et leur poids. Le spread affiché est ramené au profil de votre obligation. |
| 4. Comment la ressemblance est mesurée | Réglage des poids | Quels critères comptent pour juger deux obligations semblables. Les poids appris varient d'un jeu de données à l'autre. |
| 5. CatBoost | Apprentissage automatique (arbres de décision en série) | Quelles variables il utilise, mesuré par SHAP (la contribution de chaque variable à chaque prédiction). |
| 6. Correcteur | CatBoost sur les erreurs | De combien il corrige le modèle A. Une correction n'est utile que si elle réduit l'erreur hors échantillon. |
| 7. Forêt aléatoire | Centaines d'arbres de décision moyennés | Modèle de comparaison, qui n'entre pas dans l'estimation. Importance mesurée en brouillant une variable. |
| 8. Côte à côte | Courbe des pairs, CatBoost, forêt | Ce que chaque modèle a appris sur la notation et la maturité. Des courbes qui se superposent veulent dire que les modèles sont d'accord. |

## Partie 3 : Comparaison et confiance

| Graphique | Question | Lecture |
|---|---|---|
| Quel modèle se trompe le moins ? | Le choix du modèle est-il justifié ? | Barre la plus basse dans chaque palier. Les paliers n'ont pas les mêmes niveaux de spread : comparer à l'intérieur d'un palier. |
| Le score de confiance tient-il ses promesses ? | Le score distingue-t-il les cas faciles des difficiles ? | La courbe bleue doit monter de gauche à droite. |
| L'intervalle à 90 % … | L'intervalle est-il honnête ? | Les barres doivent être au niveau de la ligne de 90 %. |

## Partie 4 : Qualité et validation

Cette partie dit si chaque modèle est fiable, avec des tests réussis ou
échoués.

**Comment les modèles sont testés.** Les obligations sont réparties en
5 groupes ; chaque groupe est estimé par des modèles entraînés sur les
4 autres. Deux situations : émetteur connu (ses autres obligations restent
disponibles) et émetteur inconnu (tout l'émetteur est retiré). Tous les
modèles sont testés sur les mêmes obligations.

| Indicateur | Sens |
|---|---|
| Erreur médiane (%, pb) | La moitié des estimations se trompent de moins que ce chiffre |
| Biais | Erreur médiane avec son signe : positive, le modèle surestime |
| Cas < 5 %, < 10 %, < 20 % | Part des estimations dont l'erreur est sous ce seuil |
| R² hors échantillon | Part de la dispersion des spreads expliquée sur des obligations non vues |
| Variation entre plis | Écart relatif de l'erreur d'un groupe d'obligations à l'autre |
| Validation / apprentissage | Erreur hors échantillon divisée par l'erreur sur les données d'entraînement |

| Test | Règle | Ce qu'il vérifie |
|---|---|---|
| Biais | entre −2 % et +2 % | pas de surestimation ni sous-estimation systématique |
| Stabilité entre plis | variation ≤ 15 % | le résultat ne dépend pas du découpage |
| Apport | erreur plus faible que la courbe des pairs | la complexité sert à quelque chose |
| Surapprentissage | validation ≤ 1,3 × apprentissage | le modèle n'a pas appris par cœur |
| Couverture | entre 85 et 95 % | l'intervalle à 90 % est honnête (estimation retenue) |

Le test de surapprentissage n'est appliqué qu'aux modèles paramétriques
(courbe des pairs, CatBoost, forêt aléatoire). Pour le modèle A et les
lectures de contrôle, qui s'appuient sur les autres obligations, l'erreur
d'apprentissage n'a pas de sens : l'obligation serait son propre comparable.
Les seuils sont dans `config/settings.yaml` (section `quality`).

En ligne de commande :

    python scripts/qualite.py

Résultat sur les données synthétiques réalistes : 40 tests réussis, 11 échoués.
Les échecs les plus parlants :
- l'estimation retenue ne fait pas mieux que la courbe des pairs pour un
  émetteur inconnu (−1,1 %) : la marge de 5 % garde le modèle A ;
- CatBoost sous-estime systématiquement d'environ 4 % (biais) ;
- la forêt aléatoire surapprend (erreur de validation 1,5 fois l'erreur
  d'apprentissage).

## Partie 5 : Clustering

Le clustering regroupe les obligations qui se ressemblent par leurs
caractéristiques. **Il n'estime pas le spread** : il aide à lire l'échantillon
et à situer votre obligation.

| Méthode | Principe | Quand elle est utile |
|---|---|---|
| k-means | k centres ; chaque obligation rejoint le plus proche | Familles compactes de tailles voisines |
| Hiérarchique (Ward) | Fusionne pas à pas les familles les plus proches | Familles emboîtées, du général au détail |
| Mélange gaussien | Chaque famille est une « cloche » de forme libre | Familles de formes et de tailles différentes |
| HDBSCAN | Cherche les zones denses ; le reste est « atypique » | Repérer les obligations qui ne ressemblent à rien |

| Graphique | Lecture |
|---|---|
| Carte des familles | Deux axes qui résument au mieux les différences. Deux points proches sur la carte peuvent être éloignés en réalité : la carte ne montre qu'une partie de l'information (le pourcentage est indiqué). |
| Niveau de spread de chaque famille | Des rectangles bien étagés : les familles correspondent à des niveaux de risque. |
| Les familles expliquent-elles les spreads ? | Si « familles et notation » dépasse nettement « notation seule », les familles apportent une information nouvelle. |
| Combien de familles ? | La silhouette mesure la séparation : au-dessus de 0,5, familles nettes ; sous 0,25 (ligne pointillée), familles qui se chevauchent. Le point noir est le nombre affiché. |
| Les méthodes trouvent-elles les mêmes familles ? | 1 : familles identiques ; 0 : accord dû au hasard. |

### D'où vient le nombre de familles

k-means, Ward et le mélange gaussien ne choisissent pas leur nombre de
familles : on le leur donne. Le programme teste de 2 à 10 familles avec le
k-means et note chaque découpage par la silhouette.

- **Structure nette** (meilleure silhouette au-dessus de 0,5, ou au-dessus de
  0,25 avec un vrai sommet) : le nombre retenu est celui qui sépare le mieux.
  Un encadré vert le signale.
- **Pas de structure nette** (silhouette faible, ou qui baisse à chaque
  famille ajoutée) : les obligations forment un continuum. Le nombre affiché
  est alors un choix de lecture, 4 par défaut (`analysis.clustering_reading_k`
  dans `config/settings.yaml`). Un encadré orange le signale.

Sur les données synthétiques, la silhouette baisse de 0,35 (2 familles) à
0,13 (10 familles) : il n'y a pas de nombre naturel de familles. Le découpage
en 2 sépare dette senior et dette subordonnée ; en 4, il ajoute une coupe par
notation et par maturité, plus utile pour lire l'échantillon.

Option « Avec le spread » : les familles sont formées aussi sur le spread. On
obtient des familles de risque, mais la mesure « les familles expliquent les
spreads » devient circulaire, donc flatteuse.

## Ce que montrent les données synthétiques

Ces constats valent pour les données d'essai (profil réaliste), pas pour les données réelles.
Ils montrent ce que les graphiques permettent de voir.

- La notation explique à elle seule environ 80 % des écarts de spread.
- La forêt aléatoire et CatBoost ne battent pas le modèle par comparables. Ils
  n'utilisent pas la courbe de l'émetteur.
- Il n'y a pas de structure en groupes : la silhouette baisse à chaque famille
  ajoutée. Les 4 familles de lecture expliquent moins les spreads que la
  notation seule (55 % contre 79 %), mais les deux ensemble en expliquent 88 %.
- Les quatre méthodes de clustering ne trouvent pas les mêmes familles. HDBSCAN
  forme surtout des familles par secteur et devise, et classe environ un tiers
  des obligations comme atypiques.

## Rapport pour un LLM

Le rapport réunit en un fichier Markdown une consigne d'interprétation, un
glossaire, tous les constats et les tableaux de l'onglet, et, si une
obligation est donnée, sa situation (estimation, comparables, contrôles,
famille selon chaque méthode). Il suffit de le donner à un LLM.

- Dans l'interface : onglet « Analyse », encadré « Rapport pour un LLM ».
- En ligne de commande :

      python scripts/export_analyses.py --isin XS0000000010

  Options : `--bond '<JSON>'` pour une obligation hors base, `--out` pour le
  fichier de sortie, `--sans-foret` pour aller plus vite. Le rapport est écrit
  dans `rapports/`.
- Un exemple sur données synthétiques est versionné :
  `rapports/analyses_donnees_synthetiques.md`.

## Pour l'agent

Le serveur MCP expose les mêmes analyses, sans les données détaillées des
graphiques :

| Outil | Contenu |
|---|---|
| `analyze_sample` | Composition et ce qui explique les spreads |
| `analyze_models` | Chaque modèle, comparaison, confiance ; `include_forest=false` pour aller plus vite |
| `analyze_comparables` | Les comparables d'une estimation (`run_id`) |
| `analyze_clusters` | Familles selon `method` (`kmeans`, `ward`, `gmm`, `hdbscan`), avec placement de l'obligation si `run_id` est donné |
| `analysis_report` | Le rapport Markdown complet ci-dessus |
| `model_quality` | Indicateurs et tests de chaque modèle |

Chaque réponse contient une liste `findings` : des phrases en français dont
tous les chiffres sont calculés. L'agent peut les reformuler, mais ne doit pas
y ajouter de chiffre.
