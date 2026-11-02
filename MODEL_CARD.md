# Model Card: Card Fraud Classifier (XGBoost)

## Intended use

- Rank card transactions by fraud risk so that analysts review the riskiest ones first.
- A decision-support tool: a human reviews every alert before a card is blocked or a customer is contacted.
- Not intended for automatic account closure, credit decisions or any decision about a person without human review.

## Model

- Gradient-boosted trees (XGBoost, depth 4, learning rate 0.05, 306 trees after early stopping on validation PR-AUC).
- Fraud class up-weighted by the square root of the class ratio (21.8), chosen on validation data.
- Alert threshold chosen to maximise F1 on the validation period.
- Baseline for comparison: logistic regression with balanced class weights.

## Data

- 284,807 card transactions made by European cardholders over two days in September 2013; 492 (0.17%) are fraud.
- Features: 28 anonymised PCA components (V1-V28), log(1 + amount) and hour of day.
- Split by time: train = first 60% (360 frauds), validation = next 20% (57 frauds), test = last 20% (75 frauds).

## Performance on the test period

| Metric | XGBoost | Logistic regression |
|---|---|---|
| PR-AUC | 0.787 | 0.748 |
| ROC-AUC | 0.981 | 0.983 |
| Precision at chosen threshold | 88.9% | 96.2% |
| Recall at chosen threshold | 74.7% (56 / 75) | 66.7% (50 / 75) |
| False alarms | 7 | 2 |
| Fraud value caught | 65.9% | 60.3% |

### Slice check by transaction amount (XGBoost)

| Amount band | Transactions | Frauds | Recall | Precision |
|---|---|---|---|---|
| 0-10 | 21,786 | 45 | 77.8% | 85.4% |
| 10-100 | 25,081 | 15 | 73.3% | 100% |
| 100-500 | 8,480 | 11 | 72.7% | 100% |
| 500+ | 1,615 | 4 | 50.0% | 66.7% |

The largest transactions, where losses per fraud are highest, have the weakest recall. With only 4 frauds in that band the estimate is very uncertain, but it should be watched in production.

## Limitations

- **Small number of frauds.** The test period has 75 frauds, so every percentage above moves by about 1.3 points per transaction. Results would need confirming on a larger, more recent dataset.
- **Old, short data window.** Two days of 2013 European transactions. Fraud patterns change quickly; the model would need retraining on a bank's own recent data.
- **Anonymised features.** V1-V28 are PCA components, so explanations cannot name business fields (merchant, country, channel). In a real deployment the raw features would be used, which also makes explanations meaningful to analysts.
- **No protected attributes.** The data contains no customer demographics, so fairness across customer groups cannot be measured here. A production version must test for it.

## Monitoring recommendations

- Track precision and recall of reviewed alerts weekly, overall and per amount band.
- Watch for drift in the score distribution and in the top SHAP features.
- Retrain on a fixed schedule and whenever recall on reviewed cases drops below an agreed floor.
