# data-science-talent-competition

# DSTC 2024 Final Round – Loan Default Prediction & Banking Chatbot

A data science solution combining predictive modeling for loan defaults and a smart banking chatbot to enhance financial service efficiency and customer experience.

## 🚀 Project Overview

This project was developed for the **Data Science Talent Competition (DSTC) 2024** Final Round by **The Byte Squad**. It consists of two main components:

- **Loan Default Prediction** using machine learning and ensemble methods
- **AI Banking Chatbot** powered by OpenAI API and RAG (Retrieval-Augmented Generation)

---

## 📊 Loan Default Prediction

### 🔧 Data Preprocessing

- Handled **NULL values** using median imputation and Random Forest Regression
- Removed ~3% rows with irrecoverable data
- Rounded positive decimals and replaced negative values with median
- Feature selection using:
  - Correlation matrix
  - Recursive Feature Elimination (RFE) with Random Forest

### 🏷️ Label Imbalance Handling

- Imbalanced dataset: Only **18.76%** of samples are defaults (`label = 1`)
- Applied **SMOTE** and **undersampling** to balance data

### 🧠 Model Development

- Compared models: `Decision Tree`, `Random Forest`, `XGBoost`
- Final model: **XGBoost** (via `XGBClassifier` and `DMatrix`)
- Evaluation metrics:
  - **Accuracy**
  - **F1 Score** (for better handling of imbalance)

### 📈 Model Performance

| Model          | Accuracy | F1 Score |
|----------------|----------|----------|
| Decision Tree  | 80.87%   | 0.789    |
| Random Forest  | 86.10%   | 0.848    |
| XGBoost        | **89.42%**   | **0.887**    |

---

## 💬 AI Banking Chatbot

### 🎯 Purpose

- Answer user questions about finance, legal documents, and solutions
- Enhance accessibility via **Text-to-Speech (TTS)**
- Personalize responses using **System Instructions**
- Powered by **OpenAI GPT-4o + tts-1 API** with **RAG (Retrieval-Augmented Generation)**

### ✅ Advantages

- Reduced labor cost
- 24/7 availability
- Personalized and scalable
- Faster and more efficient customer support

### ⚠️ Limitations

- Inability to handle complex cases
- Less human trust and empathy
- Heavily depends on data quality

---

## 🛠️ Tech Stack

- Python (scikit-learn, xgboost, pandas, etc.)
- OpenAI API (GPT-4o, TTS-1)
- RAG (Retrieval-Augmented Generation)
- Jupyter Notebook / Colab

---

## 👥 Team

**The Byte Squad** – DSTC 2024 Final Round

---

## 📄 License

This project is created for academic and competition purposes.

---

## 🙏 Acknowledgements

Thanks to the DSTC 2024 organizers, mentors, and all supporting platforms.

