# Fraud Detection in Financial Transactions  

## Project Overview  
This project focuses on proactive fraud detection using a real-world financial transaction dataset containing over 6 million records.  
The dataset is highly imbalanced with only 0.13% fraudulent transactions, making fraud detection a challenging problem.  

The main goal was to build machine learning models that can identify fraudulent transactions with high recall (catching maximum frauds) while keeping false positives low.  

---

## Dataset Information  
- Source: [Kaggle – PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)  
- Rows: ~6.3 Million  
- Columns: 11  

### Key Features
- `step`: Time step (1 step = 1 hour, 30 days total).  
- `type`: Transaction type (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).  
- `amount`: Transaction amount.  
- `oldbalanceOrg` / `newbalanceOrig`: Sender’s balance before and after transaction.  
- `oldbalanceDest` / `newbalanceDest`: Receiver’s balance before and after transaction.  
- `isFraud`: Target variable → 1 if fraud, else 0.  
- `isFlaggedFraud`: Rule-based flag for transfers > 200k.  

---

## Key Insights from EDA  
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions.  
- Fraud cases usually involve very high amounts and leave sender’s account balance near zero.  
- The existing rule (`isFlaggedFraud`) flagged only large transfers but missed most frauds → ineffective business rule.  
- Strong multicollinearity between balances → dropped `newbalanceOrig` and `newbalanceDest`.  

---

## Feature Engineering & Selection  
- Dropped ID-like columns (`nameOrig`, `nameDest`) as they provide no predictive power.  
- Removed highly correlated features (`newbalanceOrig`, `newbalanceDest`).  
- Final Features:  
  - `step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, `isFlaggedFraud`.  
- Target Variable: `isFraud`.  

---

## Models & Results  

### Logistic Regression (baseline)  
- ROC-AUC: 0.82  
- High recall but very low precision → misclassified many normal transactions.  

### Random Forest  
- ROC-AUC: 0.98  
- Precision: 94%, Recall: 70%  
- Strong performance, balanced results.  

### XGBoost (final model)  
- ROC-AUC: 0.9995  
- Recall: 98% (almost all frauds detected)  
- Precision: 0.22 (some false positives, but acceptable trade-off).  

---

## Feature Importance (XGBoost)  
- Top predictors:  
  1. Transaction type  
  2. Sender’s old balance  
  3. Transaction amount  
- Business rule flag (`isFlaggedFraud`) had very low importance.  

---

## Prevention Strategies  
- Real-time monitoring of TRANSFER and CASH_OUT transactions.  
- Use dynamic thresholds instead of fixed rules.  
- Apply multi-factor authentication for suspicious transactions.  
- Monitor accounts with sudden balance drops.  
- Strengthen KYC to detect account takeovers.  

---

## Monitoring Effectiveness  
- Compare fraud detection rate before & after system.  
- Track false positive rate (legit transactions flagged).  
- Measure cost savings (fraud loss prevented).  
- Regularly retrain ML model to adapt to new fraud patterns.  

---

## Tech Stack  
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  

---

## Conclusion  
This project shows that XGBoost is highly effective for fraud detection in imbalanced datasets.  
It achieved ROC-AUC 0.999 and detected 98% of fraud cases, proving that ML models significantly outperform static rule-based systems.  
