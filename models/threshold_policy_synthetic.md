# Threshold policy simulation

Decision-support artifacts for analyst review, not approved rules.

- Review cost per alert: $1.25
- Handling cost per missed fraud: $15.00
- Review capacity: 2.00% of traffic
- Recall target: 95%
- Precision target: 90%

| Policy | Purpose | Threshold | Precision | Recall | FPR | Flagged | Alerts | Expected cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| operating | Cutoff currently used by the deployed model | 0.618 | 22.1% | 10.9% | 0.395% | 0.50% | 226 | $137,087 |
| cost_optimized | Minimises review cost plus missed-fraud cost | 0.071 | 2.3% | 84.8% | 36.886% | 37.38% | 16,819 | $27,814 |
| balanced_f1 | Balances precision and recall (F1) | 0.494 | 15.7% | 20.0% | 1.111% | 1.30% | 587 | $112,154 |
| high_recall | Catches at least 95% of fraud | 0.032 | 1.5% | 95.0% | 63.101% | 63.43% | 28,542 | $37,083 |
| high_precision | Keeps precision at or above 90% | 0.690 | 35.6% | 3.5% | 0.065% | 0.10% | 45 | $162,066 |
| review_capacity | Stays within 2% of traffic flagged | 0.464 | 13.4% | 22.4% | 1.491% | 1.70% | 767 | $108,212 |
