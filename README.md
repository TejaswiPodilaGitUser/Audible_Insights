
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


## Exploratory Data Analysis (EDA)
<img width="1607" alt="Screenshot 2025-04-10 at 10 43 21 PM" src="https://github.com/user-attachments/assets/155aad63-6e15-49a4-9374-db0558117877" />

<img width="1647" alt="image" src="https://github.com/user-attachments/assets/1856788e-6d9a-4937-951f-9f06eb0f2a2d" />

<img width="1568" alt="image" src="https://github.com/user-attachments/assets/7871cf8f-e698-463b-9fdb-fce3a9fbf9e2" />

## Streamlit App

<img width="1657" alt="image" src="https://github.com/user-attachments/assets/122f794d-1514-4bfc-b4ec-cda4031190ee" />

<img width="1636" alt="image" src="https://github.com/user-attachments/assets/4d6cfd8b-dd1e-4000-a577-c490d1a90565" />

<img width="1616" alt="image" src="https://github.com/user-attachments/assets/24686c7c-b774-4c73-8395-6cd24a76d90a" />

<img width="1626" alt="image" src="https://github.com/user-attachments/assets/f8b8ddce-f02b-4063-ba11-517758b455a4" />


## AWS 
<img width="1702" alt="AWS Port Connected" src="https://github.com/user-attachments/assets/e1886fe8-02fa-4381-be1d-54315504232c" />

