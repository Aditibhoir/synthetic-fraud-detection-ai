# Credit Card Fraud Detection Using Machine Learning & Synthetic Data

## **Project Overview**
This project aims to detect fraudulent credit card transactions using machine learning models enhanced with synthetic data. Due to the inherent class imbalance in real-world fraud datasets, we generate synthetic data using **CTGAN** to improve model performance.

---

## **1. Data Description**
- **Dataset:** Credit card transactions (`creditcard_small.csv`), including features `Time`, `V1` to `V28`, `Amount`, and `Class`.
- **Problem:** Fraudulent transactions are rare (~0.5% of data), leading to class imbalance.
- **Solution:** Synthetic data generation using **CTGAN** to augment minority class (fraudulent transactions).

---

## **2. Data Augmentation**
- Used **CTGAN** (Conditional Tabular GAN) to generate realistic synthetic fraud samples.
- Augmented dataset ensures balanced training, improving model recall and F1-score.
- Visualizations show distribution similarity between real and synthetic data.

---

## **3. Modeling**
- **Models Trained:**
  - Random Forest (best-performing model)
  - Logistic Regression
  - Gradient Boosting
- **Training:**
  - Models trained on real-only and augmented datasets.
  - Hyperparameter tuning applied to Random Forest for optimal performance.
- **Evaluation Metrics:**
  - **Recall:** Ability to detect fraud correctly.
  - **Precision:** Accuracy of fraud predictions.
  - **F1-Score:** Balance between recall and precision.
  - **ROC-AUC:** Model discrimination ability.
- **Saved Artifacts:**
  - Best model: `fraud_model.pkl`
  - Scaler: `scaler.pkl` (StandardScaler used during preprocessing)

---

## **4. Model Deployment**
- **Flask API** implemented in `4_Model_Deployment.ipynb`.
- **Endpoints:**
  1. `/` → Home route (confirms API is running)
  2. `/predict` → Accepts transaction features and returns:
     - `prediction`: 0 (normal) or 1 (fraud)
     - `fraud_probability`: model’s probability for fraud
- **Testing:**
  - Random normal and fraud rows tested.
  - Fraud probability correctly reflects likelihood of fraud.
- **API Run:** Using `nest_asyncio` to run Flask inside Jupyter Notebook.

---

## **5. Performance Comparison**
- Metrics collected for **real-only vs augmented datasets**:

| Metric     | Real Data | Augmented Data |
|------------|-----------|----------------|
| Recall     | 0.55      | 0.76           |
| Precision  | 1.00      | 1.00           |
| F1-Score   | 0.71      | 0.86           |
| ROC-AUC    | 0.85      | 0.93           |

- **Visualization:** `model_performance_comparison.png` shows bar chart comparison.
- **Conclusion:** Augmentation significantly improves fraud detection recall and F1-score without compromising precision.

---

## **6. Usage**
1. **Run the deployment notebook (`4_Model_Deployment.ipynb`)**:
   - Ensure `fraud_model.pkl`, `scaler.pkl`, and `creditcard_small.csv` are in the same directory.
   - Run the Flask API cell.
2. **Send POST request to `/predict`** with a list of 30 transaction features:

```python
import requests

sample_features = [0.1, -1.2, 0.5, ..., 1.0]  # Replace with actual feature values

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"features": sample_features}
)
print(response.json())

Output:

{
  "prediction": 0,
  "fraud_probability": 0.02
}

## **7. Project Structure**
Final Project/
├── 2_CTGAN_Training.ipynb
├── 3_Model_Training.ipynb
├── 4_Model_Deployment.ipynb
├── fraud_model.pkl
├── scaler.pkl
├── creditcard_small.csv
├── model_performance_comparison.png
├── README.md

## **8. Key Takeaways**

Synthetic data helps solve class imbalance and improves model recall and F1-score.

Random Forest performs best among tested models for fraud detection.

Flask API allows easy deployment and testing of the model.

Visualizations effectively communicate the impact of data augmentation.

## **9. References**

Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

CTGAN: https://sdv.dev/SDV/user_guides/tabular/ctgan.html

Scikit-learn Documentation: https://scikit-learn.org/

Flask Documentation: https://flask.palletsprojects.com/