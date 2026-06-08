# 📱 AI-Powered Screen Time & Productivity Prediction System

## 📌 Project Overview

This project investigates the relationship between screen time habits and productivity using Machine Learning and Data Science techniques. The system analyzes user behavior, performs exploratory data analysis (EDA), and predicts both screen time category and productivity level through an interactive web application.

The project was developed as part of the **Introduction to Data Science** course and demonstrates the complete data science workflow, including data cleaning, exploratory analysis, statistical testing, machine learning model development, and web deployment.

---

## 🎯 Objectives

* Analyze patterns between screen usage and productivity.
* Identify factors affecting user productivity.
* Perform statistical analysis on survey data.
* Build machine learning models for prediction.
* Develop an interactive web application for real-time predictions.

---

## 📊 Dataset Description

The dataset contains responses from approximately **200 participants** and includes:

* Age Group
* Gender
* Education Level
* Occupation
* Average Screen Time
* Device Type
* Screen Activity
* App Category
* Screen Time Period
* Work Environment
* Productivity Level
* Attention Span
* Work Strategy
* Notification Handling
* Usage of Productivity Applications

After preprocessing and cleaning missing values, **196 valid records** were used for analysis.

---

## 🔍 Exploratory Data Analysis (EDA)

The project includes:

### Data Cleaning

* Missing value handling
* Feature transformation
* Data encoding

### Statistical Analysis

* Pearson Correlation Analysis
* Chi-Square Testing
* Distribution Analysis
* Productivity Trend Analysis

### Visualizations

* Correlation Heatmaps
* Productivity Distribution Charts
* Device vs Productivity Analysis
* Notification Handling Impact
* Screen Time vs Productivity Analysis
* Interactive Productivity Dashboard

Generated outputs:

* `eda_visualizations.png`
* `productivity_dashboard.png`

---

## 🤖 Machine Learning Models

### 1. Screen Time Prediction

Model Used:

* Extra Trees Regressor

Purpose:

Predicts the user's likely screen time category based on behavioral and demographic information.

### 2. Productivity Prediction

Model Used:

* Extra Trees Classifier

Purpose:

Predicts whether a user is:

* Unproductive
* Moderately Productive
* Highly Productive

---

## 🧠 Feature Engineering

Additional features were created to improve model performance:

* Attention Span Numeric Encoding
* Age × Attention Interaction Feature
* Notification × Attention Interaction Feature

Feature Scaling:

* StandardScaler

Categorical Encoding:

* LabelEncoder

---

## 🌐 Flask Web Application

The project includes an interactive web application where users can:

* Enter personal and behavioral information
* Receive predicted screen time category
* Receive predicted productivity level
* View prediction confidence score
* View automatically generated visual analytics

### Application Features

✅ Real-Time Predictions

✅ Interactive User Interface

✅ Visualization Generation

✅ REST API Endpoint

✅ Machine Learning Integration

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Science Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy

### Machine Learning

* Scikit-Learn

### Web Development

* Flask
* HTML
* CSS
* Jinja2 Templates

### Image Processing

* Pillow (PIL)

---

## 📂 Project Structure

```text
Data Science Project/
│
├── app.py
├── data.csv
├── main.ipynb
├── background.png
├── eda_visualizations.png
├── productivity_dashboard.png
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
│
└── README.md
```

---

## 🚀 Installation & Setup

### Clone Repository

```bash
git clone <repository-url>
cd Data-Science-Project
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn flask pillow
```

### Run Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 📈 Key Findings

* Most participants were moderately productive.
* Screen time alone showed very weak correlation with productivity.
* Productivity was influenced more by behavioral factors such as:

  * Work strategies
  * Attention span
  * Notification handling
  * Work environment
* Users who effectively managed distractions generally achieved higher productivity scores.

---

## 🔮 Future Improvements

* Larger dataset collection
* Deep Learning models
* Personalized productivity recommendations
* User account system
* Cloud deployment
* Mobile application version
* Real-time analytics dashboard

---

## 👨‍💻 Author

**Muhammad Mohsin Ahmad**

BS Data Science

Superior University

Semester Project – Introduction to Data Science

---

## 📄 License

This project is developed for educational and academic purposes.
