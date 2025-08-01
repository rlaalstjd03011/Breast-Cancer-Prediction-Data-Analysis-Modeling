# Breast-Cancer-Prediction-Data-Analysis-Modeling

# 🧠 Breast Cancer Diagnosis with Logistic Regression

This project uses the famous Breast Cancer Wisconsin dataset to build a **logistic regression model** that classifies whether a tumor is malignant (cancerous) or benign (non-cancerous).

It includes **data preprocessing, model training, evaluation**, and simple **data visualization** with `matplotlib` and `seaborn`.

---

## 📁 Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numeric features (e.g., radius, texture, perimeter, area...)
- **Target:** 0 = Malignant, 1 = Benign

---

## 🛠️ Tools & Libraries

- Python 3.x  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🚀 Steps

1. **Load the Dataset**
2. **Convert to DataFrame**
3. **Explore the Data**
4. **Train/Test Split**
5. **Train Logistic Regression Model**
6. **Evaluate the Model**
7. **Visualize the Data**

---

## 📊 Results

- **Accuracy:** 95.6%
- **Confusion Matrix:**
- **Precision:** 0.946  
- **Recall:** 0.986  
- **F1 Score:** 0.966

---

## ✅ Conclusion

The model shows high accuracy and recall, especially good at detecting malignant tumors (minimizing false negatives).  
Also, based on the data, **"mean radius"** has a strong correlation with the target class, while features like **"mean texture"** seem less predictive.

---

## 📌 Future Improvements

- Try other models like Random Forest, SVM, or XGBoost  
- Perform hyperparameter tuning  
- Use cross-validation  
- Deploy the model as a simple web app  

---


