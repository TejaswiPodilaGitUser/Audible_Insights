
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
<img width="1652" alt="image" src="https://github.com/user-attachments/assets/18a53c18-c3a9-4335-bb60-bd785d862c69" />

<img width="1591" alt="image" src="https://github.com/user-attachments/assets/a9054a43-cf62-4a7c-938e-da4861949644" />

<img width="1045" alt="image" src="https://github.com/user-attachments/assets/94e0634e-6f38-4c1e-90c1-ee8ec21335fb" />


## Streamlit App

<img width="1674" alt="image" src="https://github.com/user-attachments/assets/b7c04717-8d2f-486c-a2bd-9afe09d9ccb7" />


<img width="1636" alt="image" src="https://github.com/user-attachments/assets/79a59242-60f1-458c-8a32-96285b2c639c" />


## AWS 
<img width="1702" alt="AWS Port Connected" src="https://github.com/user-attachments/assets/e1886fe8-02fa-4381-be1d-54315504232c" />

