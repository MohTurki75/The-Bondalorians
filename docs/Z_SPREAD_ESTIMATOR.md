# Fair Z-Spread Estimator — Technical Documentation

> **Purpose.** This document describes the repository's hybrid fair-value estimator as analyzed from its code and documentation. It is intended to be loaded into the project as a standalone reference. Examples with invented values are explicitly illustrative; implementation details should be checked against the current repository revision before relying on them operationally.

## 1. What the estimator does

The estimator predicts a **fair Z-spread** for a bond from its characteristics and the observed spreads of bonds in a reference universe. In simplified form:

\[
\widehat{Z}_{fair}=f(\text{rating, maturity, issuer, currency, seniority, sector, country, size, coupon, callability, ...})
\]

The observed `z_spread` in the universe is the target signal. Much of the modeling is performed on the log scale:

\[
y_i=\ln(Z_i), \qquad Z_i>0.
\]

This makes relative errors more comparable across spread levels. A residual of 0.10 in log space is approximately a 10.5% multiplicative difference, not 10 basis points.

### Fair-value estimation versus classical Z-spread calculation

These are different operations:

- **Classical market Z-spread calculation:** given a bond's cash flows, market price, and a benchmark discount curve, solve for the constant spread \(z\) that makes discounted cash flows equal the observed price.
- **This estimator:** given bond attributes and a universe of observed bond spreads, infer a model-based fair spread. It does not solve the bond price equation and does not derive a spread from the target bond's market price.

The output can be compared with a quoted or market-implied Z-spread to assess relative value, but it is not a substitute for a cash-flow pricing engine.

## 2. Architecture at a glance

The estimator is a layered pipeline:

```text
Bond universe and target characteristics
                 │
                 ▼
       Schema normalization / filtering
                 │
                 ▼
          PeerCurve (Ridge)
                 │
       log-spread residuals
          ┌──────┴──────┐
          ▼             ▼
 Issuer offset and    External comparable
 maturity slope       selection / weighting
          └──────┬──────┘
                 ▼
       Model A: structural blend
          ┌──────┴──────┐
          ▼             ▼
 Model B: CatBoost   Model C: CatBoost residual correction
          └──────┬──────┘
                 ▼
       Out-of-sample model selection
                 ▼
 Difficulty / uncertainty model
                 ▼
 Conformal interval + controls + explanation
```

The named components in the analysis include `data/schema.py`, `quant/peer_curve.py`, and `quant/tuning.py`, with a service-level `estimate()` assembling and returning the result. Confirm exact module paths and current interfaces in the checked-out revision.

## 3. Data preparation

The input universe contains fields such as:

- `issuer`, `rating`, `maturity_years`, `currency`, `seniority`
- `sector`, `country`, `coupon_type`, `callable`
- `amount_outstanding`, `z_spread`

The preparation described in `data/schema.py` creates numerical representations used by the models:

- **Rating:** mapped to an ordered `notch` (for example AAA = 1, AA+ = 2, …, BBB = 9, BBB− = 10); larger values mean weaker credit quality.
- **Maturity:** `log_t = ln(maturity_years)`.
- **Issue size:** `log_amt = ln(amount_outstanding)`.
- **Seniority:** mapped to an ordinal `sen_rank`; the prior analysis listed secured = 0, senior = 1, senior non-preferred = 2, tier 2 = 3, AT1 / hybrid = 4.

Rows with unknown rating, non-positive maturity, or non-positive Z-spread are excluded from the described modeling path. These constraints are necessary for the ordinal and logarithmic transformations. Missing or unrecognized categories should be reviewed because their handling can affect coverage and predictions.

## 4. PeerCurve baseline

`PeerCurve` is the broad-market structural baseline, described as a Ridge regression on log spread. A representative specification is:

\[
\ln(Z_i)=\beta_0+\beta_1 r_i+\beta_2 r_i^2+\beta_3\ln(T_i)
 +\beta_4 r_i\ln(T_i)+\text{seniority} +\text{sector}+\text{currency}+\text{coupon}+\cdots+\epsilon_i.
\]

Here, (r_i) is rating notch and (T_i) is maturity. The analysis notes optional terms for maturity curvature, rating-by-sector and rating-by-seniority interactions, country, and issue size. Ridge regularization stabilizes coefficients when predictors are numerous or correlated.

For bond (i), the residual relative to the baseline is:

\[
e_i=\ln(Z_i)-\widehat{\ln(Z_i)}_{peer}.
\]

A positive residual means the observed spread is wider than the peer curve predicts, in multiplicative/log terms.

## 5. Issuer shrinkage and issuer maturity slope

### Issuer-level offset

The raw mean residual for an issuer can be noisy, especially when few of its bonds are observed. The analysis describes an empirical-Bayes style shrinkage:

\[
\widehat{\alpha}_j=\lambda_j\bar e_j,
\qquad \lambda_j=\frac{n_j}{n_j+k},
\]

where (n_j) is the issuer's observation count and (k) is estimated from between-issuer and within-issuer variation. With little issuer data, (lambda_j) is small and the estimate stays near the broad peer baseline. With more data, the issuer-specific offset receives more weight.

For illustration, if (n=2) and (k=8), then (lambda=0.20). If (n=40), then (lambda\approx0.83). These are examples, not asserted production parameters.

### Issuer-specific maturity slope

The analysis also describes a trend in issuer residuals across log maturity. A raw slope can be written as:

\[
b_j=\frac{\operatorname{Cov}(\ln T,e)}{\operatorname{Var}(\ln T)}.
\]

This slope is shrunk separately and applied relative to the issuer's reference maturity, with extrapolation limited by the implementation. Conceptually:

\[
IssuerEffect_j(T)=\widehat{\alpha}_j+widehat b_j\bigl(\ln T-\ln T_{j,ref}\bigr).
\]

This lets the model reflect that an issuer's spread premium may vary by tenor rather than applying one constant offset to every bond.

## 6. Comparable selection and weighting

External peers are selected outside the target issuer. The selection relaxes hard filters in stages to avoid an empty or tiny peer set:

1. same currency, seniority, and coupon type;
2. same currency and seniority;
3. same currency;
4. no hard categorical filter.

The described minimum candidate count is `min_candidates = 5`; the exact configured value should be verified in the current code.

For candidate (i) and target (t), a normalized distance combines rating, maturity, sector, country, and size:

\[
D_{it}=w_rD_{rating}+w_tD_{maturity}+w_sD_{sector}+w_cD_{country}+w_aD_{size}.
\]

The analysis gives examples of normalized components:

\[
D_{rating}=\min(|\Delta notch|/6,1),\qquad
D_{maturity}=\min(|\ln T_i-\ln T_t|/1.5,1).
\]

Initial weights were described as rating 35%, maturity 25%, sector 25%, country 10%, and size 5%. These may be tuned; do not treat them as fixed without checking configuration. `quant/tuning.py` was described as searching weights, bandwidth, and peer count, then accepting changes only if they improve on a separate validation sample.

Distance becomes a kernel weight, optionally multiplied by an observation-quality weight:

\[
w_i=\exp\left[-(D_{it}/h)^2\right]\,obs\_weight_i.
\]

The weighted comparable residual is:

\[
\widehat e_{peers,t}=\frac{\sum_i w_i e_i}{\sum_iw_i}.
\]

The effective peer count is more informative than the raw count:

\[
N_{eff}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

A set with many candidates but one dominant weight can therefore have a low effective count.

## 7. Model A — transparent structural blend

The central Model A combination is described on the **log-spread scale**:

\[
\boxed{\widehat y_A=\widehat y_{peer}+IssuerEffect+(1-\lambda_j)\widehat e_{peers,t}}
\]

and the spread in basis points is recovered as:

\[
\widehat Z_A=\exp(\widehat y_A).
\]

This blends the broad market level, the target issuer's shrunk effect (including its tenor slope), and evidence from external peers. The peer residual contribution is down-weighted as issuer-specific evidence becomes stronger. The exact implementation's treatment of the issuer slope and peer residual should be checked against the current source before reproducing the equation in another system.

## 8. CatBoost Model B — challenger

Model B is a nonlinear challenger that predicts log spread directly from bond attributes, described as rating, maturity, seniority, size, callability, sector, country, currency, and coupon type. The prior analysis says it does **not** use issuer identity, so it can generalize to issuers absent from training.

Monotonic constraints were described for rating notch and seniority rank: worsening credit quality or moving to a riskier subordination rank should not reduce predicted spread, all else equal. Confirm exact CatBoost feature encoding and supported constraints in the active model configuration.

Model B is evaluated against the structural approach; it need not become the production champion.

## 9. CatBoost Model C — residual corrector

Model C learns the systematic error of Model A instead of replacing the whole model. Its target is:

\[
Correction=\ln(Z_{observed})-\widehat y_A.
\]

It then forms:

\[
\widehat y_C=\widehat y_A+\widehat{Correction},
\qquad \widehat Z_C=\exp(\widehat y_C).
\]

This architecture retains Model A's interpretable decomposition while allowing a nonlinear model to correct recurring patterns. Its performance still depends on out-of-sample training and leakage control.

## 10. Model selection and backtesting

The described comparison includes A (structural), B (CatBoost), and C (A plus correction). Validation separates two use cases:

- **Known issuer:** random observation splits can test prediction when the issuer is represented in the training data.
- **Unseen issuer:** folds are grouped by issuer so all bonds of a held-out issuer are excluded from training.

This distinction matters: random splitting alone can overstate performance for a new issuer. The score described is median absolute error in log spread. A challenger must beat A by a configured margin (the earlier analysis cited 5%) to be selected; otherwise A remains champion. The exact metric, margin, split strategy, and data chronology must be verified in the current code.

For credible backtests, ensure every feature and peer observation was available as of the prediction date. Random or grouped folds do not by themselves prevent look-ahead leakage in time-sensitive market data.

## 11. Difficulty model and conformal interval

A separate difficulty/uncertainty model estimates a scale σ from case-quality signals. The analysis lists effective comparable count, average peer distance, issuer weight and issuer sample size, rating, maturity, segment peer count, disagreement between A/B and A/C, known-versus-unseen issuer status, and peer/issuer dispersion.

Conformal calibration then scales this estimate using held-out calibration residuals. The described default is α = 0.10 (nominal 90% coverage). On log scale:

\[
h=q_{1-\alpha}\,\sigma,
\qquad Interval=\left[\exp(\widehat y-h),\exp(\widehat y+h)\right].
\]

The interval is therefore multiplicative and case-dependent, not a fixed number of basis points. A nominal conformal level is not a guarantee for every segment or future regime; coverage depends on calibration quality and exchangeability/stability assumptions. Confirm the exact quantile convention and calibration split in the implementation.

## 12. Controls and returned result

The previous code analysis described `run_controls()` checks including:

- positive predicted spread;
- position relative to the 1st and 99th percentiles of a nearby segment;
- seniority consistency against more senior debt of the same issuer;
- material disagreement between A and B;
- effective comparable count and peer support.

The result of `estimate()` was described as including the predicted spread in bp, interval and level, tier/reason, champion, component model values, confidence, issuer offset, comparables, a waterfall/explanation, controls, caveats, model version, and CatBoost/ correction explanations such as SHAP when available. Exact field names are version-dependent and should be checked in the public result schema.

Controls are diagnostics: they can flag weak support or inconsistency, but do not turn a model estimate into a market quote or a guarantee of fair value.

## 13. End-to-end numerical example (illustrative)

Consider a hypothetical EUR senior unsecured bond with BBB+ rating and seven years to maturity. Assume the fitted PeerCurve predicts 135 bp. Since the model works in logs, use the following deliberately simplified illustrative inputs:

- Peer baseline: 135 bp.
- Shrunk issuer offset: +9 bp in an *approximate spread-point interpretation*.
- Weighted comparable residual contribution: +4.8 bp in the same approximation.

An intuitive additive summary gives (135+9+4.8=148.8) bp. The exact Model A calculation is instead additive in log space. To show that calculation coherently, suppose the calibrated terms are:

\[
\widehat y_{peer}=\ln(135),\quad IssuerEffect=0.0645,\quad
(1-\lambda)\widehat e_{peers}=0.0339.
\]

Then:

\[
\widehat y_A=\ln(135)+0.0645+0.0339,
\qquad \widehat Z_A=135\,e^{0.0984}\approx149.0\text{ bp}.
\]

Suppose B predicts 151 bp and C predicts 150 bp. If backtesting selects A as champion for this deployment or case category, the point estimate is about 149 bp. If the calibrated conformal half-width is (h=0.12), then:

\[
[149e^{-0.12},149e^{0.12}]\approx[132,168]\text{ bp}.
\]

This example explains the stages; none of these values are claimed to be output by the repository for a real bond. In particular, the simple +9/+4.8 bp explanation is only an intuitive approximation; production components and intervals are combined in log space.

## 14. Operational interpretation and limitations

Use the output as a model-based relative-value estimate with its peer evidence, uncertainty interval, and diagnostics. Compare it with a market Z-spread calculated from price/cash flows only after aligning valuation date, benchmark curve, conventions, and instrument features.

Key risks to monitor include:

- quality and timestamp alignment of the observed Z-spread target;
- survivorship, stale pricing, and outliers in the reference universe;
- look-ahead leakage in peer selection, feature construction, tuning, or calibration;
- thin segments and unseen issuers;
- model drift across market regimes;
- callable or otherwise option-embedded bonds, for which a classical Z-spread itself may be less suitable than an option-adjusted measure.

Sophisticated modeling does not remove dependence on clean data, sound validation, and a clearly defined valuation target.

## 15. Quick glossary

| Term | Meaning here |
|---|---|
| Market Z-spread | Constant spread solved from price and cash flows against a benchmark curve. |
| Fair Z-spread | Model estimate inferred from characteristics and observed peer spreads. |
| PeerCurve | Regularized broad-market regression baseline for log spread. |
| Issuer shrinkage | Pulls noisy issuer residual estimates toward the broad peer level. |
| Comparable residual | Weighted residual information from similar bonds outside the issuer. |
| Model A | PeerCurve plus shrunk issuer effect and peer evidence. |
| Model B | Direct CatBoost challenger. |
| Model C | CatBoost correction applied to Model A. |
| Conformal interval | Empirically calibrated prediction interval, described here on log scale. |
