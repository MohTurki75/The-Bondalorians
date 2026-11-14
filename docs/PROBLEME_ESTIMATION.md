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
Un modèle ML doit faire au moins 5 % mieux que le modèle A pour le remplacer.

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
