# 💳 Credit Card Customer Segmentation using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview

This project focuses on grouping credit card customers into distinct segments based on their usage behavior. By leveraging unsupervised machine learning techniques like **K-Means Clustering** and **Hierarchical Clustering**, we aim to identify patterns in credit card usage to help businesses tailor their marketing strategies and improve customer engagement.

The segmentation is primarily driven by features such as credit limit, number of credit cards, and frequency of bank/online visits.

## 📂 Dataset

The dataset involves credit card customer details, including:
- **Avg_Credit_Limit**: Average credit limit of the customer.
- **Total_Credit_Cards**: Total number of credit cards held.
- **Total_visits_bank**: Number of visits to the bank.
- **Total_visits_online**: Number of online logins.
- **Total_calls_made**: Number of calls made to customer service.

> **Note**: The dataset does not contain missing values or duplicates, ensuring high data quality for modeling.

## ⚙️ Methodology

1.  **Data Preprocessing**:
    -   Removal of irrelevant columns (e.g., `Sl_No`, `Customer Key`).
    -   Standardization of features using `StandardScaler` to ensure all variables contribute equally to the distance calculations.

2.  **Exploratory Data Analysis (EDA)**:
    -   Univariate analysis to understand distributions.
    -   Bivariate analysis (correlation heatmaps, pair plots) to identify relationships between features.

3.  **Modeling**:
    -   **K-Means Clustering**: Partitioned data into $k$ distinct clusters. Optimal $k$ was determined using the Elbow Method and Silhouette Analysis.
    -   **Hierarchical Clustering**: Built a hierarchy of clusters using Ward's linkage method to minimize variance within clusters.

4.  **Evaluation**:
    -   **Elbow Method**: Identified the "elbow" point where adding more clusters yields diminishing returns in variance reduction.
    -   **Silhouette Score**: Measured how similar an object is to its own cluster compared to other clusters.

## 📊 Key Results

-   **Optimal Clusters**: Both K-Means and Hierarchical Clustering suggested **3 distinct customer segments**.
    -   **Segment 0 (Low Credit / High Calls)**: Customers with low credit limits and few cards, who frequently contact support.
    -   **Segment 1 (Medium Credit / Bank Visits)**: Customers with moderate credit limits who prefer visiting the bank in person.
    -   **Segment 2 (High Credit / Online Users)**: Premium customers with high credit limits and many cards, who predominantly use online banking.

-   **Visualizations**:
    -   _Elbow Curve_ and _Silhouette Plots_ confirm $k=3$ as the optimal choice.
    -   Cluster profiling highlights clear distinctions in behavior across the three groups.

## 🚀 Repository Structure

```
├── data
│   ├── raw              # Original dataset (Credit_Card_Customer_Data.xlsx)
│   └── processed        # Cleaned/Processed data (if saved)
├── notebooks            # Interactive Jupyter Notebooks
│   └── analysis.ipynb   # Main analysis notebook
├── src                  # Source code for modular logic
│   ├── data_loader.py       # Data loading script
│   ├── data_preprocessing.py# Data cleaning and scaling
│   ├── modeling.py          # Clustering models
│   ├── evaluation.py        # Evaluation metrics and plots
│   └── generate_visuals.py  # Script to generate project visuals
├── visuals              # Generated plots and charts
│   ├── elbow_curve.png
│   └── silhouette_plot.png
├── reports              # Documentation & Business Report
│   └── Customer_Business_Report.pdf
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 💻 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nabankur14/credit-card-customer-segmentation-using-machine-learning.git
    cd credit-card-customer-segmentation-using-machine-learning
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis**:
    -   Open `notebooks/analysis.ipynb` in Jupyter/Colab for interactive exploration.
    -   Or run the visual generation script:
        ```bash
        python src/generate_visuals.py
        ```

## 🔮 Future Improvements

-   Integrate supervised learning to predict customer LTV (Lifetime Value) based on segments.
-   Deploy the clustering model as a web app using Streamlit or Flask.
-   Experiment with DBScan for density-based clustering to handle potential noise.

## 👨‍💻 Author

**Nabankur Ray**
-   **GitHub**: [nabankur14](https://github.com/nabankur14)
-   **LinkedIn**: [Nabankur Ray](https://www.linkedin.com/in/nabankur-ray/)

---
*Created as part of a Data Science Portfolio Project.*
