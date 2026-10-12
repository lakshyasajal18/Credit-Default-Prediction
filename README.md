# Credit Card Default Prediction

Predicting which credit card clients will miss a payment next month, comparing an interpretable logistic regression baseline, a lasso logistic regression, and gradient boosting, with an honest look at validation, calibration, and decision thresholds.

**Data:** UCI "Default of Credit Card Clients" (Yeh and Lien, 2009): 30,000 credit card clients in Taiwan, April to September 2005, with 6 months of repayment status, bill amounts, and payment amounts. Target: default payment next month (22.1% of clients). Download from the UCI Machine Learning Repository (a Kaggle mirror was used here) and place the CSV at `data/UCI_Credit_Card.csv`.

## Results (held out test set, 6,000 clients, 20% stratified split)

| Model | ROC AUC (95% bootstrap CI) | PR AUC | Brier score |
|---|---|---|---|
| Logistic regression (statsmodels) | 0.742 (0.726 to 0.759) | 0.501 | 0.1428 |
| Lasso logistic regression | 0.742 (0.725 to 0.757) | 0.501 | 0.1428 |
| Gradient boosting | **0.780 (0.764 to 0.794)** | **0.555** | **0.1349** |
| Predict base rate (reference) | 0.500 | n/a | 0.1723 |

- Gradient boosting improves test AUC by about 0.038 over the linear models. Hyperparameters were chosen by 5 fold cross validation on the training set only (CV AUC 0.786, close to the 0.780 test AUC), and the test set was used once.
- Lasso did **not** help: cross validation selected an essentially unpenalized model and kept all 27 features, so it matches plain logistic regression. With 24,000 training rows and 27 features there is little for regularization to fix.
- All three models are well calibrated (see `results/calibration.png`), so predicted probabilities can be read as risk estimates, not just rankings.

### What drives default (logistic regression, standardized inputs)

Strongest signal by far is repayment behavior: the engineered count of months with a late payment has an odds ratio of 2.17 per standard deviation (p < 1e-100), followed by the worst delay in the 6 month window (1.32) and the most recent month's repayment status (1.25). A higher credit limit is associated with lower default odds (0.81). Full table in `results/logit_coefficients.csv`. These are associations in observational data, not causal effects.

## Threshold analysis

A model outputs probabilities; a lender needs a decision. I picked the cutoff that minimizes total cost under two **assumed** cost ratios (a missed defaulter costs 2x or 5x as much as a wrongly flagged good client). These ratios are illustrative assumptions, not any real lender's economics. Thresholds were chosen with out of fold predictions on the training set, then applied to the test set.

| Assumed cost ratio | Threshold | Recall | Precision | Cost vs. best naive policy |
|---|---|---|---|---|
| 2 : 1 | 0.33 | 51.6% | 55.9% | 1,824 vs 2,654 (approve everyone), 31% lower |
| 5 : 1 | 0.14 | 81.4% | 33.8% | 3,348 vs 4,673 (decline everyone), 28% lower |

The point: the best cutoff is a business decision, and it moves a lot (0.33 vs 0.14) as the cost of a miss changes. See `results/threshold_cost.png`.

## Method

1. Cleaned undocumented education codes into one "other" category.
2. Engineered features from repayment history: months delayed, maximum delay, average credit utilization, average payment to bill ratio.
3. **Excluded SEX and MARRIAGE** as inputs. Using protected attributes in credit decisions is legally problematic in the US, so they are dropped on purpose. AGE is kept and would need review before any real use.
4. Stratified 80/20 train/test split (seed 42). All tuning by 5 fold stratified CV on the training set.
5. Bootstrap (1,000 resamples) for test AUC confidence intervals.

## Limitations

- **No time dimension.** The data has no per client timestamps, so the split is random, not time based. Real credit models are validated on later periods, and performance here is likely optimistic relative to deployment under drift.
- One country, one 6 month window in 2005, from a period of a card debt crisis in Taiwan. Results should not be assumed to transfer to other populations.
- Cost ratios in the threshold analysis are assumptions.
- No fairness audit beyond removing SEX and MARRIAGE. Proxies for those attributes may remain in the other features.
- Gradient boosting is less interpretable than the logistic baseline; explanation tooling (for example SHAP) was not added.

## Reproduce

```
pip install -r requirements.txt
cd src
python run_models.py --data ../data/UCI_Credit_Card.csv --out ../results
```

All numbers above come from `results/metrics.json`, written by the script.
