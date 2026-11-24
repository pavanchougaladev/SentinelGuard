# Threshold policy simulation

Decision-support artifacts for analyst review, not approved rules.

- Review cost per alert: $1.25
- Handling cost per missed fraud: $15.00
- Review capacity: 2.00% of traffic
- Recall target: 95%
- Precision target: 90%

| Policy | Purpose | Threshold | Precision | Recall | FPR | Flagged | Alerts | Expected cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| operating | Cutoff currently used by the deployed model | 0.290 | 90.7% | 75.0% | 0.009% | 0.10% | 43 | $2,572 |
| cost_optimized | Minimises review cost plus missed-fraud cost | 0.013 | 18.2% | 82.7% | 0.452% | 0.55% | 236 | $1,746 |
| balanced_f1 | Balances precision and recall (F1) | 0.243 | 90.7% | 75.0% | 0.009% | 0.10% | 43 | $2,572 |
| high_recall | Catches at least 95% of fraud | 0.001 | 2.1% | 84.6% | 4.825% | 4.92% | 2,103 | $4,051 |
| high_precision | Keeps precision at or above 90% | 0.243 | 90.7% | 75.0% | 0.009% | 0.10% | 43 | $2,572 |
| review_capacity | Stays within 2% of traffic flagged | 0.007 | 6.6% | 82.7% | 1.418% | 1.52% | 648 | $2,261 |
