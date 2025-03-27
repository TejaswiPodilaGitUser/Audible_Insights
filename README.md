<<<<<<< HEAD
# Audible_Insights
Intelligent Book Recommendation System
=======
# Audible Insights: Intelligent Book Recommendations

## 📌 Project Overview
This project builds a **personalized book recommendation system** using **machine learning, NLP, and clustering**. The system suggests books based on user preferences and book features. The final product is a **Streamlit app**, deployed on **AWS**.

## 📂 Directory Structure
```
- data/: Contains raw and processed datasets.
- notebooks/: Jupyter notebooks for data exploration and model training.
- src/: Core scripts for data processing, feature engineering, clustering, and recommendation models.
- app/: Streamlit-based web application.
- models/: Trained models stored in .pkl format.
- config/: Configuration files for model and deployment settings.
- deployment/: AWS & Docker-related files.
- tests/: Unit tests for data processing and model validation.
```

## 🚀 Features
- **Content-Based Filtering**: Recommends books based on features like genres and descriptions.
- **Clustering-Based Recommendation**: Groups books into similar clusters.
- **Hybrid Model**: Combines multiple approaches for better recommendations.
- **Streamlit UI**: User-friendly interface for book recommendations.
- **AWS Deployment**: Hosted using AWS EC2 or Elastic Beanstalk.

## 🛠️ Setup Instructions
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```
### 3️⃣ Deployment on AWS
Refer to `deployment/aws_setup.md` for deployment steps.

## 📊 Data Sources
- `Audible_Catalog.csv`
- `Audible_Catalog_Advanced_Features.csv`

## 🛠️ Technologies Used
- **Python**
- **Machine Learning**
- **NLP (TF-IDF, Word2Vec)**
- **Streamlit**
- **AWS (EC2, Elastic Beanstalk)**
- **Docker**
>>>>>>> 71128a7 (Initial commit - Added Streamlit app for Audible Insights)
