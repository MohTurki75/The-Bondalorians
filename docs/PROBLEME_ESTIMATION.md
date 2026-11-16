# Estimation du Z-Spread juste : problème, modèles et résultats

Document de synthèse à donner à un assistant (humain ou LLM) pour l'aider à
améliorer l'estimateur. Il est autonome : il ne suppose pas d'accès au code.

## 1. Le problème

**Cas d'usage.** Un broker propose à un gérant de portefeuille d'acheter une
obligation à un certain Z-Spread. Le gérant veut savoir quel serait le
Z-Spread « juste » de cette obligation, pour juger si la proposition est chère
ou non. Le système estime ce Z-Spread juste ; la comparaison avec le broker
reste au gérant.

**Cadre.** Hackathon de 2 jours, équipe de 3 développeurs et 1 analyste
métier.

**Contraintes :**

- Univers d'environ 4 000 obligations dans PostgreSQL, accessible seulement le
  premier jour. On ne connaît pas encore son schéma exact.
- Aucune donnée externe ni recherche web pendant l'exécution ; des
  référentiels peuvent être chargés en base avant.
- Résultats explicables et auditables ; jury sensible à la rigueur
  quantitative.
- Un agent LLM, branché par un serveur MCP, orchestre et explique. Il ne
  calcule aucun chiffre.

**Entrée :** une obligation, par son ISIN si elle est dans l'univers, sinon
par ses caractéristiques (émetteur, secteur, pays, devise, notation,
séniorité, maturité résiduelle, type de coupon, option de remboursement
anticipé, encours).

**Sortie :**

- le Z-Spread juste en points de base ;
- un intervalle à 90 % et un score de confiance sur 100 ;
- le palier de difficulté ;
- les comparables, une cascade explicative en points de base, des contrôles
  de cohérence et des réserves.

Ni prix, ni rendement, ni verdict cher/pas cher.

## 2. Hypothèses sur les données

| Hypothèse | Statut |
|---|---|
| Le Z-Spread de chaque obligation de l'univers est en base | Supposé, à confirmer le jour 1. Sinon il faudra un pricer et des courbes de taux. |
| Photographie à une seule date | Supposé ; aucune dynamique temporelle n'est modélisée. |
| Notation disponible pour chaque obligation | Supposé ; une notation inconnue fait écarter la ligne. |
| Schéma réel | Inconnu ; un fichier de correspondance le traduit vers un schéma interne. |

**Variables internes :**

- `notch` : notation convertie en cran, AAA = 1, D = 22 ; S&P, Fitch ou Moody's.
- `log_t` : log de la maturité résiduelle en années.
- `sen_rank` : rang de séniorité, de 0 (garanti) à 4 (AT1, hybride).
- `log_amt` : log de l'encours ; s'il est inconnu, encours médian de l'univers.
- Indicatrice de remboursement anticipé.
- Catégories : secteur, pays, devise, type de coupon, séniorité.

**Cible modélisée :** le log du Z-Spread. Les spreads sont à peu près
multiplicatifs entre notations.

**Données de développement : synthétiques.** Elles servent à tester la chaîne
en attendant les vraies. Deux profils existent ; le profil **réaliste** est
celui par défaut, le profil **simple** est décrit ci-dessous :

- 4 000 obligations, environ 540 émetteurs, répartis selon une loi de Pareto
  plafonnée ; 14 % des émetteurs n'ont qu'une obligation.
- Le log du spread est généré par un modèle log-linéaire : terme quadratique
  en notation, pente de maturité qui dépend de la notation, effets de
  séniorité, de secteur et de devise, effet aléatoire par émetteur
  (écart-type 0,15) et bruit (écart-type 0,08).
- **Conséquence importante :** ce profil simple ressemble beaucoup au modèle
  A d'origine. Ses résultats flattent le modèle A et ne disent rien du marché
  réel.
- **Profil réaliste** : il ajoute des courbes qui s'aplatissent en haut
  rendement, une sensibilité à la notation propre au secteur, une prime de
  subordination plus forte pour les émetteurs faibles, une pente de maturité
  propre à chaque émetteur (écart-type 0,08), un effet pays, une prime
  d'illiquidité des petites souches, un bruit qui croît avec la notation et
  la subordination, et 4 % de notations en retard sur le marché. Ces effets
  ont été écrits par nous : un modèle conçu pour eux les capture forcément
  mieux.

## 3. Les paliers de difficulté

| Palier | Condition |
|---|---|
| Simple | Au moins 3 autres obligations de l'émetteur, même devise et même séniorité, dont les maturités encadrent celle de la cible |
| Intermédiaire | Émetteur connu, mais peu d'obligations comparables ou maturité hors de leur plage |
| Complexe | Dette subordonnée ou hybride (Tier 2, AT1, subordonnée, hybride), émetteur connu |
| Émetteur inconnu | Émetteur absent de l'univers |

Une obligation déjà présente dans l'univers est toujours exclue de ses propres
comparables et de son écart émetteur.

## 4. Les modèles

### 4.1 Courbe des pairs (base de tout)

Régression ridge du log du Z-Spread, pénalité 0,01, constante non pénalisée.
Variables explicatives : constante, notch, notch², log_t, log_t × notch,
remboursable, puis indicatrices de séniorité, secteur, devise et type de
coupon.

Elle donne le spread « normal » d'un profil. Les **résidus** (log observé
moins courbe) portent ce que la courbe n'explique pas.

**Version enrichie (par défaut)** : elle ajoute log_t², notch centré × secteur,
sen_rank × notch centré (centrage sur BBB), le pays et le log de l'encours
centré. Chaque terme est rattaché à un groupe de la cascade explicative.

### 4.2 Modèle A : comparables ajustés (modèle transparent)

En log :

    log_spread = courbe(cible)
               + λ × écart_émetteur
               + (1 − λ) × écart_comparables

**Écart émetteur.** Moyenne des résidus des autres obligations de l'émetteur,
dans la même devise si possible. Il est rétréci par λ = n / (n + k), où n est
le nombre de ces obligations.

**k est estimé par analyse de variance** (estimateur des moments, plan non
équilibré) : variance du bruit à l'intérieur d'un émetteur divisée par la
variance entre émetteurs, bornée entre 0,05 et 50. Sur le profil réaliste,
k = 1,05 : une seule obligation de l'émetteur fait retenir 49 % de son écart.

**Pente de maturité propre à l'émetteur (par défaut).** Pente des résidus de
l'émetteur sur log_t, rétrécie par λ_b = Sxx / (Sxx + k_b), où Sxx est la
somme des carrés des écarts de log_t de l'émetteur. k_b = σ² / τ², avec τ²
= variance des pentes observées moins leur variance d'échantillonnage
moyenne (méthode des moments, émetteurs d'au moins 3 obligations). Effet sur
la cible : λ_b × pente × (log_t cible − log_t moyen de l'émetteur), avec une
extrapolation limitée à 0,3 au-delà des maturités de l'émetteur. Sans pente
réelle dans les données, k_b devient infini et la pente est neutralisée.

**Écart des comparables** (hors émetteur) :

1. *Filtres durs* : même devise, même séniorité, même type de coupon. Ils
   sont relâchés dans cet ordre s'il reste moins de 5 candidats.
2. *Distance de Gower pondérée*, avec des écarts normalisés entre 0 et 1 :

   | Critère | Écart normalisé |
   |---|---|
   | Notation | min(|Δnotch| / 6, 1) |
   | Maturité | min(|Δlog_t| / 1,5, 1) |
   | Secteur | 1 si différent, 0 sinon |
   | Pays | 1 si différent, 0 sinon |
   | Taille | min(|Δlog_amt| / 2, 1) |

3. *Sélection* : les `top_k` plus proches.
4. *Poids* : noyau gaussien exp(−(d / largeur)²), multiplié par le poids de
   l'observation.
5. *Écart* : moyenne pondérée des résidus des comparables. Utiliser les
   résidus revient à ramener chaque comparable au profil de la cible.

**Réglage des comparables.** Les poids des 5 critères, la largeur du noyau et
`top_k` (5 à 60) sont réglés par recherche aléatoire puis locale (200 essais).
Quatre garde-fous :

- le réglage se fait sur 10 % des obligations ;
- il n'est adopté que s'il bat les valeurs par défaut sur 10 % d'autres
  obligations ;
- chaque critère garde un poids d'au moins 5 % ;
- les obligations de réglage sont exclues de la calibration des intervalles.

| Paramètre | Valeur par défaut | Valeur apprise (synthétique) |
|---|---|---|
| Notation | 0,35 | 0,10 |
| Maturité | 0,25 | 0,36 |
| Secteur | 0,25 | 0,28 |
| Pays | 0,10 | 0,10 |
| Taille | 0,05 | 0,16 |
| Largeur du noyau | 0,25 | 0,42 |
| top_k | 15 | 60 |

Erreur de validation : 11,1 % avec les valeurs par défaut, 10,5 % après
réglage. Les poids appris changent beaucoup d'un jeu de données à l'autre.

### 4.3 Modèle B : CatBoost indépendant (challenger)

- **Cible** : log du spread.
- **Variables** : notch, log_t, sen_rank, log_amt, remboursable, et en
  catégories secteur, pays, devise, type de coupon et séniorité.
- **L'émetteur n'est pas une variable**, pour généraliser à un émetteur
  inconnu.
- **Monotonie** : le spread prédit ne baisse jamais quand la notation ou le
  rang de séniorité se dégrade.
- **Réglages** : 300 itérations, profondeur 5, taux d'apprentissage 0,08.

### 4.4 Modèle C : modèle A corrigé par CatBoost

Un CatBoost (200 itérations, profondeur 4, régularisation L2 = 5) apprend
l'erreur du modèle A, c'est-à-dire log observé moins log du modèle A.

- **Variables** : celles du modèle B, plus la prédiction du modèle A, λ,
  l'écart émetteur, l'écart des comparables, le nombre effectif de
  comparables, la distance moyenne, le nombre d'obligations de l'émetteur et
  le niveau de relâchement des filtres.
- **Exemples d'entraînement** : chaque obligation est estimée par le modèle A
  sans elle-même, deux fois — émetteur conservé, puis émetteur retiré.

### 4.5 Forêt aléatoire (comparaison seulement)

Forêt de 200 arbres (feuilles d'au moins 3 obligations, 50 % des variables
tirées à chaque nœud), mêmes variables que le modèle B. Elle n'entre pas dans
l'estimation : elle vérifie qu'un autre type de modèle ne ferait pas mieux.

### 4.6 Backtest et choix du modèle retenu

Validation croisée à 5 plis, sous deux schémas :

- **Émetteur connu** : plis aléatoires, qui servent aux paliers simple,
  intermédiaire et complexe.
- **Émetteur inconnu** : plis groupés par émetteur, qui servent au palier
  « émetteur inconnu ».

Dans chaque palier, le modèle retenu est celui de plus faible erreur médiane.
Les candidats sont A, B, C et la courbe des pairs seule (P). Un candidat doit
faire au moins 5 % mieux que le modèle A pour le remplacer.

**Lecture descendante (contrôle, non retenue).** Démarche de trader : groupe le
plus proche ayant assez de points (même émetteur, devise et séniorité ; puis
même émetteur ; puis secteur, devise, lettre de notation et séniorité ; puis
en relâchant devise et notation ; sinon courbe seule), correction variable par
variable avec les coefficients de la courbe des pairs, puis interpolation
en maturité (par défaut, moyenne des deux points les plus proches de chaque
côté, ramenés à la maturité de la cible ; en option, interpolation linéaire
stricte). Elle est affichée avec ses points et ses corrections ; un écart de
plus de 25 % avec l'estimation déclenche une alerte.

Mesure hors échantillon (profil réaliste), erreur médiane :

| Palier | A | Courbe seule | Descendante stricte | Descendante lissée |
|---|---|---|---|---|
| Simple | 7,1 % | 12,5 % | 8,8 % | 7,6 % |
| Intermédiaire | 8,8 % | 12,9 % | 9,8 % | 9,2 % |
| Complexe | 17,1 % | 19,8 % | 17,6 % | 18,5 % |
| Émetteur inconnu | 14,6 % | 14,4 % | 17,9 % | 16,3 % |

La courbe seule ne bat A que de 1,4 % pour un émetteur inconnu : sous la marge
de 5 %, elle n'est pas retenue avec les réglages par défaut.

### 4.7 Intervalle et score de confiance

**Modèle de difficulté.** Il prédit l'erreur absolue attendue, en log, à
partir de 12 indicateurs : nombre effectif de comparables, distance moyenne,
λ, log du nombre d'obligations de l'émetteur, niveau de relâchement des
filtres, notch, log_t, sen_rank, densité du segment, écart A−B, ampleur de la
correction C, émetteur inconnu ; et trois indicateurs de dispersion
(par défaut) : écart-type pondéré des résidus des comparables autour de leur
moyenne, écart-type des résidus de l'émetteur autour de sa courbe, et une
indicatrice de disponibilité de ce dernier. C'est un CatBoost de profondeur 3 (ou une
régression ridge sans CatBoost), entraîné hors échantillon.

**Intervalle.** Prédiction conforme normalisée : quantile corrigé pour
échantillon fini de |erreur| / difficulté, calculé par palier. Intervalle
= estimation × exp(± quantile × difficulté).

**Score sur 100.** Part des cas du backtest jugés plus difficiles que le cas
estimé.

### 4.8 Contrôles et retours utilisateurs

**Contrôles :**

- spread positif ;
- spread dans la plage 1 %–99 % des notations voisines ;
- spread au moins égal à celui de la dette plus senior du même émetteur ;
- écart A−B inférieur à 25 % ;
- nombre de comparables suffisant.

**Retours.** Un spread traité (poids 1) ou un avis d'expert (poids 0,3)
devient une observation pondérée de l'univers. La courbe des pairs et k sont
recalculés aussitôt ; les modèles CatBoost, les poids des comparables et
l'intervalle ne changent qu'au recalcul complet.

## 5. Résultats mesurés (données synthétiques)

Toutes les erreurs sont hors échantillon.

### Profil réaliste, version actuelle (backtest interne)

| Palier | Obligations testées | Courbe des pairs | A | B (CatBoost) | C (A corrigé) | Forêt aléatoire |
|---|---|---|---|---|---|---|
| Simple | 1 262 | 21,3 | 11,1 | 18,4 | 10,2 | 15,8 |
| Intermédiaire | 1 818 | 19,4 | 14,2 | 19,3 | 14,3 | 19,9 |
| Complexe | 410 | 122 | 116 | 134 | 118 | 134 |
| Émetteur inconnu | 3 600 | 24,1 | 24,8 | 24,6 | 25,0 | 28,0 |

- Le modèle A reste retenu dans tous les paliers. C fait 8 % mieux que A en
  points de base dans le palier simple, mais pas en erreur relative, qui sert
  au choix.
- Score de confiance : erreur médiane de 5,5 % dans les 10 % de cas jugés les
  plus faciles, 20,6 % dans les 10 % les plus difficiles, en progression
  régulière ; corrélation de rang 0,34. L'erreur prévue reste 5 à 25 % sous
  l'erreur réalisée (la calibration conforme corrige la couverture).
- Couverture de l'intervalle à 90 % : 90 % dans chaque palier ; largeur
  médiane de 39 % du spread (simple) à 93 % (complexe).

### Effet des dernières améliorations (profil réaliste, obligations mises de côté)

| Étape | Émetteur connu | Émetteur inconnu |
|---|---|---|
| Version précédente | 18,3 pb (9,8 %) | 32,4 pb (16,5 %) |
| + courbe des pairs enrichie | 15,8 pb (9,0 %) | 28,2 pb (15,2 %) |
| + pente de maturité par émetteur | 15,5 pb (8,4 %) | 28,2 pb (15,2 %) |
| + dispersion dans la confiance | identique ; intervalle 3 % plus étroit | identique |
| + CatBoost (B et C) | aucun gain | aucun gain |

Sur le profil simple, ces ajouts ne gagnent ni ne perdent rien de
significatif.

### Profil simple, version précédente (pour mémoire)

**Erreur médiane en points de base, par palier et par modèle :**

| Palier | Obligations testées | Courbe des pairs | A | B (CatBoost) | C (A corrigé) | Forêt aléatoire |
|---|---|---|---|---|---|---|
| Simple | 1 247 | 16,4 | 8,4 | 14,4 | 8,3 | 13,6 |
| Intermédiaire | 1 843 | 17,5 | 9,1 | 17,2 | 9,4 | 18,5 |
| Complexe | 406 | 74,9 | 40,8 | 86,0 | 41,6 | 89,2 |
| Émetteur inconnu | 3 600 | 19,6 | 19,7 | 21,4 | 20,8 | 22,3 |

Le modèle A est retenu dans tous les paliers. Aucun modèle ML ne l'améliore
de 5 %. Le correcteur C améliore A dans 49 % des cas : c'est du bruit.

**Calibration :**

- **Couverture de l'intervalle à 90 %** : 90 % dans chaque palier.
- **Largeur médiane de l'intervalle** : environ 30 % du spread quand
  l'émetteur est connu, 61 % quand il est inconnu.
- **Score de confiance** : il classe moyennement les cas (corrélation de rang
  0,32 entre erreur prévue et erreur réalisée). Erreur médiane de 5,7 % dans
  les 10 % de cas jugés les plus faciles, 13,2 % dans les 10 % les plus
  difficiles. L'erreur prévue est systématiquement 15 à 30 % sous l'erreur
  réalisée.

**Effet mesuré de chaque amélioration** (erreur médiane hors échantillon) :

| Amélioration | Émetteur connu | Émetteur inconnu |
|---|---|---|
| Départ : k = 2, poids manuels | 11,9 pb | 23,5 pb |
| k estimé par analyse de variance | 9,8 pb | 23,5 pb |
| Poids des comparables réglés | 10,0 pb | 21,4 pb |
| Intervalle adaptatif | Pas d'effet sur l'erreur ; couverture plus homogène si le bruit varie selon la notation | — |
| CatBoost (B et C) | Pas de gain | Pas de gain |

**Lecture de l'échantillon :**

- La notation seule explique environ 80 % de la dispersion des log-spreads.
  Une fois la notation connue, l'émetteur explique encore 37 % de ce qui
  reste.
- Le clustering (k-means, Ward, mélange gaussien, HDBSCAN) ne trouve pas de
  structure en groupes : la silhouette baisse de 0,35 (2 familles) à 0,13
  (10 familles). La seule séparation nette est dette senior contre dette
  subordonnée.

## 6. Faiblesses connues

1. **Validation sur données trop favorables.** Le générateur synthétique est
   presque le modèle A ; le vrai test sera sur les données réelles.
2. **Émetteur inconnu.** L'erreur est double et l'intervalle très large ; les
   comparables y apportent peu par rapport à la courbe des pairs.
3. **Dette subordonnée** (palier complexe). C'est le palier le plus faible
   (plus de 100 pb d'erreur médiane sur le profil réaliste) ; aucune
   modélisation propre des options de remboursement ou de l'extension des
   AT1. La prime de séniorité selon la notation l'a réduite de 20 %.
4. **Option de remboursement anticipé.** Une simple indicatrice : pas d'OAS,
   pas de maturité au « call ».
5. **Score de confiance.** Il classe mieux les cas qu'avant (corrélation de
   rang 0,34) mais sous-estime encore l'erreur avant calibration.
6. **Un seul k et un seul k_b pour tout l'univers.** La dispersion
   intra-émetteur peut varier selon les secteurs et les notations.
7. **Courbe émetteur limitée à un niveau et une pente.** Pas de courbure
   propre à l'émetteur.
8. **Pas de liquidité, de taille relative, d'ancienneté de l'émission ni de
   prime de nouvelle émission.**
9. **Pas de dimension temporelle**, et la notation est supposée toujours
   présente et fiable (pas de perspectives ni de surveillance).
10. **Distance des comparables.** Les échelles de normalisation (6 crans,
    1,5 en log de maturité, 2 en log de taille) sont fixées à la main.

## 7. Ce que j'attends de toi

Propose des améliorations de l'estimateur classées par rapport gain/effort,
réalisables en 48 heures par 3 développeurs, sans données externes. Pour
chacune :

- le problème qu'elle corrige, parmi les faiblesses ci-dessus ;
- la méthode précise : formule ou algorithme, variables, paramètres ;
- comment mesurer son effet avec le backtest existant (deux schémas de plis,
  erreur médiane par palier, couverture et largeur de l'intervalle,
  corrélation de rang du score) ;
- le risque de surapprentissage et le garde-fou associé ;
- son effet sur l'explicabilité. Une amélioration opaque doit apporter un
  gain net, au moins 5 % d'erreur médiane en moins.

Pistes qui m'intéressent, à confirmer ou à écarter avec arguments :

- k et k_b différents par segment ;
- meilleure modélisation de la dette subordonnée et des obligations
  remboursables par anticipation ;
- apprentissage métrique pour la distance des comparables ;
- combinaison pondérée des modèles (stacking) au lieu d'un champion par
  palier ;
- prédiction conforme conditionnelle mieux calibrée ;
- ce qu'il faudrait mesurer en priorité le jour 1, sur les données réelles,
  pour savoir si les effets du profil réaliste existent vraiment.

Indique clairement ce qui relève du fait établi, du consensus de la pratique
crédit, de l'opinion débattue ou de la spéculation.

## 8. Réponse d'un assistant externe

Réponse reçue d'un autre assistant à la consigne de la section 7, reproduite
telle quelle. Elle a été rédigée à partir de ce document dans sa version
précédente : courbe des pairs enrichie, pente émetteur et indicateurs de
dispersion déjà en place, mais sans accès au code.

Oui. À mon avis, le point essentiel est le suivant : vous avez déjà un estimateur assez sophistiqué, mais vous cherchez encore surtout des gains dans les mauvais endroits.

Les résultats montrent que le problème n’est pas principalement un problème de puissance ML. Le modèle A exploite une information structurelle très forte — l’écart historique de l’émetteur par rapport à une courbe de crédit comparable — et CatBoost n’arrive pas à la remplacer. En revanche, vos résultats montrent trois faiblesses beaucoup plus importantes :

1. la dette subordonnée / callable est mal spécifiée ;
2. la variance de l’écart émetteur n’est probablement pas homogène ;
3. la validation et la calibration doivent être extrêmement rigoureuses pour éviter de présenter comme amélioration un artefact du générateur synthétique.

Je classerais les améliorations ainsi.

⸻

1. Priorité absolue : segmenter le shrinkage émetteur k

Gain potentiel : ★★★★★

Effort : ★★☆☆☆

Risque : faible à modéré

Explicabilité : excellente

C’est probablement la meilleure amélioration à tester en premier.

Votre modèle suppose actuellement :

\lambda_i=\frac{n_i}{n_i+k}

avec un k=1,05 commun à tout le marché.

C’est une hypothèse assez forte. Il est parfaitement plausible que la dispersion intra-émetteur soit différente selon :

* Investment Grade / High Yield ;
* secteur ;
* séniorité ;
* notation ;
* éventuellement devise.

Un émetteur IG industriel peut avoir une courbe relativement stable entre ses obligations senior. Un émetteur HY avec plusieurs structures de capital peut avoir des dispersions beaucoup plus fortes.

Je ne ferais cependant pas un k par secteur × rating

Avec 4 000 obligations et 540 émetteurs, vous allez rapidement créer des cellules avec trop peu d’observations.

Je commencerais par 3–4 segments économiquement justifiés :

k_global
k_IG
k_HY
ou
k_IG_senior
k_HY_senior
k_subordonné

Puis éventuellement :

k(segment) = shrink(k_segment, k_global)

par exemple sur l’échelle logarithmique :

\log k_s^* =
w_s \log k_s +(1-w_s)\log k_{global}

avec

w_s=\frac{N_s}{N_s+c}

où N_s est le nombre d’observations réellement informatives du segment.

Cela évite qu’un segment avec 40 observations vous donne un k=7,8 complètement artificiel.

Variante encore meilleure

Vous pouvez estimer :

\sigma^2_{intra,s}

et

\sigma^2_{inter,s}

par segment puis :

k_s =
\frac{\sigma^2_{intra,s}}
{\sigma^2_{inter,s}}

exactement comme vous le faites actuellement, mais avec shrinkage vers le global.

Backtest

Comparer :

A0 : k global
A1 : k IG/HY
A2 : k IG/HY/seniority
A3 : k entièrement par segment

sur les mêmes plis.

Mesurer :

* erreur médiane en % ;
* erreur médiane en pb ;
* simple/intermédiaire/complexe/inconnu ;
* surtout erreur lorsque n=1, 2, 3, 4….

Ce dernier test est particulièrement important.

Si k est meilleur, vous devriez voir quelque chose comme :

n émetteur       A0       A1
1                17%      15%
2                13%      11%
3                10%       9%
10                7%       7%

Si le gain ne concerne que les gros émetteurs, ce n’est pas très intéressant.

Mon verdict

À faire.

C’est une amélioration statistiquement naturelle, transparente et très défendable devant un jury.

⸻

2. Très forte priorité : modéliser séparément la dette subordonnée

Gain potentiel : ★★★★★

Effort : ★★☆☆☆

Risque : faible

Explicabilité : excellente

Votre résultat le montre clairement :

Complexe : 116–134 pb d’erreur médiane.

C’est votre problème numéro 1.

Et il y a une raison structurelle : vous essayez actuellement de faire entrer dans la même courbe :

* senior secured ;
* senior unsecured ;
* Tier 2 ;
* AT1 ;
* hybride.

Or la relation entre ces instruments n’est pas simplement :

Spread=f(rating,maturity,...)

La séniorité crée une interaction avec le risque de crédit de l’émetteur.

Votre amélioration actuelle :

sen\_rank \times notch

est donc dans la bonne direction, mais probablement insuffisante.

⸻

Je construirais une architecture à deux étages

Étape 1 : courbe senior

Construire le spread attendu pour une obligation senior :

S_{senior}=f(
rating,
maturity,
sector,
country,
currency,
coupon,
size
)

Étape 2 : prime de subordination

Puis :

\log S =
\log S_{senior}+
P(subordination)

avec :

P =
\alpha_{sen}
+\beta_{sen}\cdot notch
+\gamma_{sen}\cdot \log(t)
+\delta_{sen}\cdot notch^2

ou, plus simplement au hackathon :

senior
senior_subordinated
Tier2
AT1/hybrid

avec interaction rating × seniority.

Cela donne une cascade extrêmement explicable :

Senior comparable curve             142 bp
+ issuer adjustment                  +31 bp
+ Tier 2 premium                     +47 bp
+ maturity adjustment                 +9 bp
--------------------------------------------
Fair Z-spread                        229 bp

C’est beaucoup plus convaincant qu’un CatBoost qui sort 229 bp.

⸻

3. Ne traitez pas AT1/hybrides comme une simple catégorie

C’est encore plus important.

Je créerais au minimum :

Senior
Senior subordinated
Tier 2
AT1
Hybrid / perpetual

Et pour AT1/hybrides, si les données le permettent :

perpetual
first_call_date
years_to_call
coupon_reset

Même si vous ne faites aucun pricing optionnel, years_to_first_call peut être une variable extrêmement informative.

⸻

4. Callable : ne faites surtout pas un modèle OAS en 48 h

Gain potentiel : ★★★★☆

Effort : ★★★☆☆

Risque : élevé si mal fait

Votre faiblesse est réelle :

une simple indicatrice callable.

Mais je n’essaierais pas de construire un OAS complet pendant le hackathon.

Ce serait disproportionné et probablement fragile sans :

* volatilité de taux ;
* courbe complète ;
* modèle de taux ;
* calendrier exact des calls ;
* conventions de coupon ;
* éventuellement modèle de comportement de l’émetteur.

En revanche, vous pouvez obtenir une grosse partie du gain avec :

Variables

callable
years_to_first_call
call_maturity_ratio
call_to_maturity

par exemple :

x_{call}=
\frac{T_{call}}{T_{maturity}}

et surtout :

T_{firstcall}

Si une obligation 10 ans est callable dans 1 an, ce n’est pas économiquement comparable à une obligation 10 ans non callable.

Si les dates de call sont absentes

Ne rien inventer.

Gardez :

callable = 1

et faites un sous-modèle callable.

⸻

5. Courbe spécifique de l’émetteur : ajouter une courbure, mais seulement si les données le permettent

Gain : ★★★★☆

Effort : ★★☆☆☆

Risque : modéré

Vous avez déjà :

issuer\ adjustment
+
issuer\ slope

Je testerais ensuite :

issuer\ curvature

avec :

r_i =
\alpha_i+
\beta_i(\log t_i-\overline{\log t})+
\gamma_i(\log t_i-\overline{\log t})^2

Mais je ne l’estimerais pas directement par émetteur.

Il faut un shrinkage fort.

Pour un émetteur avec seulement 3 obligations, une parabole est presque toujours du surapprentissage.

Je mettrais une règle :

n < 4      : niveau uniquement
4–6        : niveau + pente fortement shrinkée
>= 7       : niveau + pente
>= 10–12   : tester courbure

Et éventuellement :

\lambda_c=\frac{S_{xx}^2}{S_{xx}^2+k_c}

ou une formulation ridge équivalente.

À tester, mais après k segmenté.

⸻

6. Votre distance de comparables est probablement le deuxième grand axe d’amélioration

Votre Gower actuelle est raisonnable, mais les résultats montrent quelque chose d’intéressant :

les poids optimisés passent de notation 35 % à 10 %, maturité 25 % → 36 %, secteur 25 % → 28 %, taille 5 % → 16 %.

Et surtout :

ils changent beaucoup d’un jeu de données à l’autre.

C’est un signal important.

Je ne conclurais pas que la métrique apprise est meilleure.

Votre gain :

11,1\%\rightarrow10,5\%

est trop faible pour justifier une complexification importante, et la forte instabilité des poids suggère que vous êtes probablement en train d’apprendre du bruit.

⸻

7. Je remplacerais le réglage des poids par une distance économétrique simple

Plutôt que :

d=
w_1d_{rating}+w_2d_{maturity}+...

je testerais une distance de Mahalanobis robuste sur les variables numériques :

d^2=(x_i-x_j)^T\Sigma^{-1}(x_i-x_j)

avec traitement séparé des catégorielles.

Mais surtout, je ne ferais pas un apprentissage sophistiqué de métrique.

Alternative très pratique

Normaliser les variables à partir de leur distribution empirique :

d_{rating}
=
\frac{|\Delta rating|}
{\text{MAD}_{rating}}

plutôt que :

\frac{|\Delta rating|}{6}

Idem :

maturity → écart en percentile
size     → écart en percentile

Cela élimine vos constantes arbitraires :

6 notches
1.5 log-years
2 log-size

et s’adapte automatiquement à la vraie base.

⸻

8. Mais je ne ferais pas de metric learning sophistiqué

Mon verdict : ❌ à écarter pour 48 h

Un Siamese network, triplet loss ou autre metric learning pourrait théoriquement apprendre :

d(x_i,x_j)

qui optimise directement l’erreur de prédiction.

Mais vous avez :

* 4 000 obligations ;
* 540 émetteurs ;
* une seule date ;
* un problème de dépendance intra-émetteur ;
* une validation par groupes.

Le risque de fabriquer une distance qui exploite les particularités de votre dataset est important.

Et surtout :

ce n’est pas nécessaire pour gagner le hackathon.

Je préfère largement une distance empirique robuste + sélection des comparables + modèle A.

⸻

9. Stacking : intéressant, mais pas comme vous l’imaginez

Gain : ★★★☆☆

Effort : ★★☆☆☆

Risque : modéré

Je ne ferais pas :

A si simple
C si...
B si...

avec un champion unique.

Je testerais plutôt :

\hat y =
w_A\hat y_A+
w_B\hat y_B+
w_C\hat y_C

avec :

w_A+w_B+w_C=1

et :

w_i\geq0

Mais les poids doivent être appris OOF.

Par exemple :

A       0.75
B       0.05
C       0.20

Et surtout : un poids par palier, pas un poids global.

Je testerais également un mélange beaucoup plus simple :

\hat y =
(1-\alpha)\hat y_A+\alpha\hat y_C

où α est appris sur les prédictions OOF.

Critère

Il faut imposer votre règle :

le modèle opaque doit apporter au moins 5 % de réduction d’erreur médiane.

Sinon :

A gagne.

Cela vous protège contre le syndrome :

« On a utilisé du CatBoost donc c’est forcément plus sophistiqué. »

Vos résultats actuels montrent justement que cette hypothèse est fausse.

⸻

10. Une amélioration que je ferais absolument : modèle hiérarchique / mixed effects

C’est probablement l’idée statistique la plus intéressante de votre problème.

Vous avez naturellement une structure :

obligation \subset émetteur \subset secteur/pays

Votre modèle A traite déjà cela implicitement :

courbe globale
      +
issuer residual
      +
issuer slope

Un modèle hiérarchique pourrait formaliser cela :

y_{ij}
=
X_{ij}\beta
+
u_j
+
v_j \log(t_{ij})
+
\epsilon_{ij}

avec :

u_j\sim N(0,\tau_u^2)

v_j\sim N(0,\tau_v^2)

et :

\epsilon_{ij}\sim N(0,\sigma^2)

C’est quasiment exactement ce que vous essayez de construire manuellement.

Pourquoi c’est intéressant ?

Parce que le modèle estime automatiquement le shrinkage :

E[u_j|data]

au lieu de construire :

\lambda=\frac{n}{n+k}

à la main.

Mais

Je ne remplacerais pas A par ça immédiatement.

Pour le hackathon :

utilisez-le comme challenger statistique, pas comme modèle principal.

Si vous obtenez une implémentation robuste de Bayesian/mixed model en 48 h, très bien. Sinon, A est déjà une approximation très défendable.

⸻

11. Votre plus gros problème actuel n’est peut-être pas le modèle : c’est la validation

Je serais particulièrement vigilant ici.

Vous avez écrit :

C est entraîné avec chaque obligation estimée par A sans elle-même, deux fois.

C’est bien.

Mais il faut aller plus loin.

Validation imbriquée obligatoire pour tout tuning

Si vous faites :

5-fold CV
    ↓
tuning des paramètres
    ↓
mesure de performance

vous avez potentiellement du leakage de sélection.

Il faut :

OUTER FOLD
│
├── train
│   └── INNER CV
│       └── choix paramètres
│
└── test
    └── mesure finale

C’est particulièrement important pour :

* poids des comparables ;
* top_k ;
* largeur du kernel ;
* k ;
* segmentation ;
* stacking ;
* calibration.

Sinon le chiffre de 10,5 % est potentiellement optimiste.

⸻

12. Et il y a un problème encore plus subtil : votre générateur synthétique

Vous l’avez déjà identifié, mais je pousserais beaucoup plus loin.

Votre générateur réaliste contient :

pente émetteur, effet pays, illiquidité, interaction secteur-rating, etc.

Vous avez ensuite construit un modèle qui connaît précisément ces mécanismes.

Cela crée une situation dangereuse :

\text{synthetic improvement}
\neq
\text{real-market improvement}

Je ferais un test de robustesse extrêmement simple

Créer 5 générateurs synthétiques indépendants.

DGP 1

Votre réaliste actuel.

DGP 2

Même structure mais interactions différentes.

DGP 3

Non-linéarité différente.

DGP 4

Bruit hétéroscédastique différent.

DGP 5

Effets aléatoires plus forts.

Puis mesurer :

A
B
C
stacking

Si C gagne uniquement sur DGP 1, vous savez qu’il est probablement suradapté au générateur.

Cela ne prend pas énormément de temps à coder et c’est très convaincant devant un jury.

⸻

13. La priorité absolue du jour 1 : découvrir la structure réelle des 4 000 obligations

Avant de toucher au ML, je ferais une analyse descriptive très précise.

Distribution

rating
maturity
sector
country
currency
seniority
coupon type
callable
amount

Puis surtout :

# obligations / issuer

Histogramme :

1
2
3
4
5
6-10
11-20
20+

Vous devez savoir immédiatement combien d’émetteurs ont :

* 1 obligation ;
* 2 ;
* ≥3 ;
* ≥5 ;
* ≥10.

Parce que toute votre architecture A dépend de ça.

⸻

14. Test extrêmement important : variance intra-émetteur

Je calculerais immédiatement :

Var(residual | issuer)

puis je regarderais :

par rating

AAA
AA
A
BBB
BB
B
CCC

par secteur

par séniorité

par taille

par pays

par devise

Si vous voyez par exemple :

Segment	σ intra-émetteur
IG senior	0,07
HY senior	0,12
Tier 2	0,19
AT1	0,27

alors votre k=1,05 global est clairement trop simplificateur.

⸻

15. Test encore plus important : la pente de maturité est-elle réellement une propriété de l’émetteur ?

Vous avez déjà commencé à le mesurer.

Je calculerais :

\beta_j

pour chaque émetteur avec ≥3 obligations.

Puis :

Var(\beta_j)

et son intervalle de confiance.

Mais surtout :

comparez

Var(\beta_j)

avec la variance attendue simplement par bruit.

Si :

\tau^2\approx0

alors votre shrinkage vers zéro est exactement ce qu’il faut.

Si :

\tau^2 \gg 0

vous avez une vraie structure issuer-specific.

C’est un très bon test de robustesse.

⸻

16. Je testerais aussi un modèle « nearest issuer » avant tout ML

C’est un benchmark très important que je ne vois pas explicitement dans votre description.

Pour un émetteur inconnu :

S_{fair}
=
curve(target)
+
weighted\ residuals

vous le faites déjà.

Mais testez aussi :

quel est le meilleur émetteur de référence ?

Par exemple :

1. même secteur ;
2. même pays ;
3. même devise ;
4. rating proche ;
5. maturité proche ;
6. taille proche.

Puis utiliser toute la courbe résiduelle de cet émetteur.

Cela permet de répondre à une question fondamentale :

« Pour un nouvel émetteur, avons-nous réellement besoin de CatBoost, ou suffit-il de trouver le bon issuer peer ? »

⸻

17. Je changerais également votre définition du score de confiance

Votre score actuel :

part des cas du backtest jugés plus difficiles.

C’est intuitif mais ce n’est pas réellement un score de probabilité de précision.

Je préférerais définir :

Confidence =
100\times
P(|error|<\epsilon\mid X)

par exemple :

probabilité que l’erreur absolue soit <10 %.

Vous pourriez afficher :

Fair Z-spread: 238 bp
90% interval: [207 ; 274]
Confidence: 82/100
Expected error: ±8.4%

C’est beaucoup plus interprétable.

⸻

18. Conformal : je conserverais, mais je changerais la calibration

Votre approche conforme est bonne conceptuellement.

Le problème est que vous utilisez :

|error| / difficulty

avec une difficulté prédite.

Je testerais une conformalisation conditionnelle par bin de difficulté.

Par exemple :

difficulty 0–20
20–40
40–60
60–80
80–100

Puis calcul du quantile conforme dans chaque bin.

Cela est beaucoup plus facile à expliquer :

« Dans les cas de difficulté similaire, 90 % des erreurs historiques sont inférieures à X. »

Attention

Il faut suffisamment d’observations par bin.

Je mettrais :

minimum 150–200 observations/bin

Sinon fusion automatique des bins.

⸻

19. Votre intervalle doit aussi être asymétrique

C’est une amélioration intéressante mais secondaire.

Vous utilisez :

S\cdot e^{\pm qd}

ce qui donne un intervalle symétrique en log.

C’est cohérent avec votre cible, mais vous pouvez avoir :

Q_{0.95}(error|X)
\neq
-Q_{0.05}(error|X)

notamment avec :

* subordination ;
* callable ;
* HY ;
* faible liquidité.

Vous pourriez donc avoir :

S_{low}=S\exp(q_{05}d)

S_{high}=S\exp(q_{95}d)

C’est plus flexible.

Mais je le mettrais après la calibration correcte du score de difficulté.

⸻

20. Ce que je ne ferais PAS pendant les 48 h

Idée	Décision	Pourquoi
Deep learning	❌	4 000 observations, trop peu
Transformer	❌	totalement disproportionné
Siamese / metric learning	❌	risque élevé, gain incertain
OAS complet	❌	données et temps insuffisants
HDBSCAN/clustering	❌	vous avez déjà démontré que ça n’aide pas
Random Forest	❌	benchmark suffisant
CatBoost plus complexe	❌	vos résultats disent déjà non
Courbure issuer systématique	⚠️	seulement si n suffisant
k segmenté	✅	très bon ratio gain/effort
Subordination dédiée	✅✅	priorité
Call variables	✅	bon ratio gain/effort
Distance empirique robuste	✅	simple
Stacking A/C	⚠️	test rapide, mais pas priorité
Conformal conditionnel	✅	amélioration de fiabilité
Mixed effects	⚠️	excellent challenger si implémentation rapide

⸻

21. Mon classement gain / effort pour les 48 h

Rang	Amélioration	Gain attendu	Effort	Risque
1	Modèle dédié subordination	★★★★★	★★	faible
2	k segmenté + shrinkage	★★★★★	★★	faible
3	Call maturity / first call	★★★★	★★	faible
4	Analyse variance réelle J1	★★★★★	★	très faible
5	Calibration conformal conditionnelle	★★★★	★★	faible
6	Distance empirique / robust scaling	★★★	★★	faible
7	Stacking A/C	★★★	★★	moyen
8	Courbure issuer shrinkée	★★★	★★	moyen
9	Mixed effects challenger	★★★★	★★★★	moyen
10	Metric learning	★★	★★★★	élevé
11	Deep ML	★	★★★★★	très élevé

⸻

22. Architecture que je présenterais au jury

Je simplifierais même votre architecture finale.

                         OBLIGATION
                             │
                             ▼
                    DATA / QUALITY CHECK
                             │
              ┌──────────────┴──────────────┐
              │                             │
       Issuer connu                    Issuer inconnu
              │                             │
              ▼                             ▼
       PEER CURVE                       PEER CURVE
              │                             │
              ▼                             ▼
       ISSUER LEVEL                   BEST COMPARABLES
              │                             │
              ▼                             │
       ISSUER SLOPE                        │
              │                             │
              └──────────────┬──────────────┘
                             ▼
                    SUBORDINATION MODEL
                             │
                             ▼
                     CALL ADJUSTMENT
                             │
                             ▼
                     FAIR Z-SPREAD
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
        CONFORMAL RANGE              DIFFICULTY SCORE
               │                           │
               └─────────────┬─────────────┘
                             ▼
                       EXPLANATION

Et CatBoost devient :

challenger / residual corrector, pas le moteur principal.

C’est beaucoup plus défendable quantitativement.

⸻

23. La cascade explicative devient un véritable avantage

Votre système peut produire quelque chose comme :

Fair Z-spread : 263 bp
Intervalle 90 % : [231–302] bp
Confiance : 78/100
Palier : Intermédiaire

Puis :

Contribution	bp
Courbe marché / profil	184
Prime rating	+21
Prime pays	+8
Prime secteur	+6
Prime taille	+4
Écart émetteur	+27
pente issuer	+5
prime Tier 2	+8
ajustement callable	0
Fair Z-spread	263

C’est beaucoup plus intéressant pour un gérant qu’un modèle qui dit simplement :

CatBoost → 263 bp.

⸻

24. Ce que je mesurerais absolument sur les données réelles le jour 1

Je ferais un dashboard de 10 statistiques, avant toute optimisation :

A. Structure

1. Nombre d’obligations / émetteur.
2. % d’émetteurs avec ≥3 obligations.
3. % d’obligations subordonnées.
4. % callable.
5. % AT1 / hybrides.

B. Dispersion

6. R^2 de la courbe de pairs.
7. variance intra-émetteur des résidus.
8. variance inter-émetteurs.
9. ratio :

k=\frac{\sigma^2_{intra}}{\sigma^2_{inter}}

10. variance des pentes issuer.

Et je segmenterais immédiatement ces statistiques par :

rating
seniority
sector
currency

⸻

25. Le test qui peut complètement changer votre stratégie

Il y a une question fondamentale que vos données synthétiques ne permettent pas encore de trancher :

Combien d’information nouvelle existe réellement dans les caractéristiques d’une obligation une fois qu’on connaît sa notation et son émetteur ?

Je calculerais une séquence de modèles imbriqués :

M0 = rating
M1 = rating + maturity
M2 = + sector/country/currency
M3 = + issuer effect
M4 = + issuer slope
M5 = + size/liquidity
M6 = + callable/subordination

Et regarderais :

\Delta R^2

ou surtout la réduction de l’erreur hors échantillon.

Cela vous dira immédiatement où investir les 48 heures.

Si :

rating                  80%
+ maturity              +7%
+ issuer                +8%
+ sector/country        +2%
+ size                  +1%
+ ML                    +0.2%

alors le message est très clair :

le problème n’est pas un problème de ML.

C’est un problème de bonne estimation du risque relatif de crédit et de la structure de capital.

⸻

26. Classification des affirmations

Fait établi / statistique robuste

* Votre modèle A domine actuellement CatBoost.
* CatBoost B ne fournit pas de gain suffisant.
* C ne fournit pas de gain robuste.
* La subordination est votre principal point faible.
* L’émetteur inconnu est nettement plus difficile.
* Les poids optimisés des comparables sont instables.
* La couverture conforme à 90 % est actuellement bonne après calibration.
* Le score de difficulté a un pouvoir discriminant mais imparfait.

Consensus de pratique crédit

* Les obligations d’un même émetteur sont généralement des comparables particulièrement informatifs.
* La séniorité et la structure du capital doivent être traitées explicitement.
* Une obligation callable n’est pas parfaitement comparable à une obligation bullet de même maturité.
* Le spread n’est pas expliqué uniquement par rating + maturité.
* Le shrinkage est approprié lorsqu’on dispose de peu d’observations par émetteur.

Opinion méthodologique

* k segmenté avec shrinkage est probablement votre meilleur prochain gain.
* Une architecture subordination dédiée est préférable à un CatBoost plus complexe.
* Le stacking mérite un test mais ne doit pas devenir automatiquement le champion.
* La métrique de distance devrait être empirique plutôt que reposer sur des échelles arbitraires.

Spéculation à tester

* L’existence d’une pente de maturité véritablement spécifique à chaque émetteur.
* L’existence d’un effet de liquidité suffisamment important pour améliorer le modèle.
* La supériorité d’un modèle hiérarchique sur A.
* L’existence d’un signal exploitable dans les variables de call sans véritable OAS.

⸻

Mon avis le plus honnête

Je ne chercherais pas à rendre votre modèle plus “ML”.

Votre propre backtest vous donne déjà un signal assez fort : le modèle structurel A fonctionne mieux que les modèles ML génériques. Le risque serait maintenant de tomber dans le biais de sophistication : chercher un CatBoost, du metric learning ou du stacking simplement parce que cela paraît plus avancé.

Je consacrerais les 48 heures restantes à :

1. k segmenté + shrinkage
2. vraie modélisation de la subordination
3. variables de call
4. validation imbriquée stricte
5. calibration conditionnelle de l’intervalle
6. analyse empirique des données réelles le jour 1

Et seulement ensuite à un stacking A/C.

Le point potentiellement le plus important est même plus fondamental : le jour 1, ne cherchez pas immédiatement à optimiser le modèle. Cherchez à mesurer si les hypothèses du profil réaliste existent dans les 4 000 obligations réelles. Si elles n’existent pas, toute l’optimisation réalisée sur le synthétique devient secondaire.

En particulier, je voudrais voir dès le premier jour les quatre chiffres suivants :

\boxed{
\sigma^2_{intra-émetteur},
\quad
\sigma^2_{inter-émetteurs},
\quad
Var(\text{pentes issuer}),
\quad
Var(\text{résidus subordination})
}

Ce sont probablement ces quatre statistiques qui vous diront plus vite que n’importe quel CatBoost où se trouve le vrai gain.
