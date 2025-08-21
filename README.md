# 📝 Reddit Rule Violation Detection using BERT

This project implements a **binary text classification system** to predict whether a Reddit comment breaks a specific rule.  
It was developed for the Kaggle competition, which challenges participants to model subreddit moderation decisions.  

---

## 📌 Problem Overview
Online communities like Reddit have **unique moderation rules**. Identifying rule-breaking comments automatically is a complex NLP problem due to:
- Diverse subreddit norms and tones.  
- Context-specific rules.  
- The need for fairness and scalability in moderation.  

The goal of this competition is to **classify Reddit comments** as either:
- ✅ Compliant (no rule violation), or  
- ❌ Breaking a specific rule.  

---

## ⚙️ Approach
This solution is built using **BERT for Sequence Classification** with a **5-fold ensemble** setup:

1. **Text Preprocessing**
   - Concatenated comment body and rule text → `"body [RULE] rule"`.
   - Tokenized using `BertTokenizer`.

2. **Model**
   - `BertForSequenceClassification` (binary output).
   - Trained separately on each fold.

3. **Ensembling**
   - During inference, predictions from all folds are averaged for robustness.

4. **Evaluation Metric**
   - Competition uses **ROC AUC** on hidden test data.

---
