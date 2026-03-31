# Credit Card Customer Segmentation using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Unsupervised Learning](https://img.shields.io/badge/Unsupervised_Learning-Clustering-blueviolet?style=for-the-badge)

> Segmenting AllLife Bank's credit card customers using K-Means and Hierarchical Clustering to drive targeted marketing campaigns and smarter service delivery.

---

## Project Overview

This project applies unsupervised machine learning techniques — specifically **K-Means Clustering** and **Hierarchical (Agglomerative) Clustering** — to segment AllLife Bank's credit card customer base into distinct, actionable groups based on their financial behaviour and service interaction patterns.

The segmentation enables the bank's Marketing and Operations teams to design personalized campaigns, upsell relevant products, and upgrade service delivery for each customer group.

👉 [Open the notebook to explore full analysis](notebook/Credit_Card_Customer_Segmentation.ipynb)

---

## Business Problem

AllLife Bank seeks to improve its credit card market penetration and upgrade its customer service model. Two core problems drive this project:

**Marketing Problem:** The bank's marketing research team identified that market penetration can be improved through personalized campaigns — both for acquiring new customers and upselling to existing ones.

**Operations Problem:** Customers perceive the bank's support services poorly. The Operations team wants to upgrade its service delivery model to resolve queries faster and more efficiently.

**Stakeholders:**
- Head of Marketing — needs customer segments for targeted campaign design
- Head of Delivery (Operations) — needs segment profiles to optimize service channels
- Data Science Team — responsible for deriving actionable clustering insights

**Decision Impact:** The clustering output directly informs which customers receive premium credit offers, which require enhanced call center support, and which should be targeted through digital-first channels.

---

## Dataset

| Attribute | Details |
|---|---|
| **Source** | AllLife Bank Internal CRM Dataset |
| **Size** | 660 rows × 7 columns |
| **Missing Values** | None |
| **Duplicates** | None |
| **Data Types** | All numerical (int64) |

**Feature Dictionary:**

| Feature | Description |
|---|---|
| `Sl_No` | Primary key / serial number |
| `Customer Key` | Unique customer identifier |
| `Avg_Credit_Limit` | Average credit limit across all credit cards (₹) |
| `Total_Credit_Cards` | Total number of credit cards held by the customer |
| `Total_visits_bank` | Total in-person bank visits per year |
| `Total_visits_online` | Total online logins/visits per year |
| `Total_calls_made` | Total calls made to customer service per year |

---

## Methodology

### 1. Data Understanding

The dataset was loaded and inspected to understand its structure and quality. Key observations included 660 customer records with 7 numerical features. No missing values, null entries, or duplicate records were found. All columns are of `int64` data type. The `Total_calls_made` column was identified as a behavioural response variable of interest, while credit limit and card count represent financial profile dimensions.

A statistical summary revealed significant right-skew in `Avg_Credit_Limit` — the mean (₹34,574) sits well above the median (₹18,000), indicating a small number of high-value customers pulling the average upward. Most customers hold between 3 and 6 credit cards, and the average number of yearly interactions (bank visits, online visits, calls) ranges from approximately 2 to 4.

### 2. Exploratory Data Analysis

**Univariate Analysis**

Each feature was examined through combined histogram-boxplot visualisations:

- **`Avg_Credit_Limit`**: Right-skewed distribution — the majority of customers have lower credit limits, with a long tail of high-limit outliers. This suggests a two-tier credit profile in the customer base.
- **`Total_Credit_Cards`**: Customers typically hold between 2 and 6 cards. The distribution shows distinct count levels rather than a continuous spread, hinting at structured credit product tiers.
- **`Total_visits_bank`**: Roughly uniform distribution across 0–5 visits per year, with a slight peak at 2 visits. In-person engagement is moderate but consistent.
- **`Total_visits_online`**: Heavily right-skewed with most customers making 0–3 visits, but a meaningful segment making 10+ visits annually — indicating a digitally engaged sub-group.
- **`Total_calls_made`**: Spread across 0–10 calls, with a peak around 4 calls. The bimodal nature suggests two distinct call behaviour patterns.

**Bivariate Analysis — Correlation Heatmap & Pair Plot**

A Pearson correlation heatmap was generated across all five behavioural features:

- **`Avg_Credit_Limit` ↔ `Total_Credit_Cards`**: Moderate positive correlation (0.61) — customers with higher credit limits tend to hold more cards, suggesting credit profile diversification.
- **`Total_visits_online` ↔ `Total_calls_made`**: Positive correlation (0.13) — customers who engage online frequently also tend to call more, indicating a remote-service-preferred segment.
- **`Total_visits_bank` ↔ `Total_calls_made`**: Negative correlation (-0.51) — frequent in-person visitors make fewer calls, suggesting these channels serve as substitutes.
- **`Total_Credit_Cards` ↔ `Total_calls_made`**: Strong negative correlation (-0.65) — customers with more credit cards make significantly fewer calls, suggesting higher financial literacy or self-sufficiency.

The pair plot reinforced these findings, visually confirming the positive relationship between credit limit and card count, and the inverse relationship between cards held and calls made.

### 3. Data Preprocessing

**Outlier Check:** Boxplots were generated for all numerical columns. Variables `Total_Credit_Cards`, `Total_visits_bank`, and `Total_calls_made` showed no visible outliers requiring treatment. `Avg_Credit_Limit` and `Total_visits_online` displayed upper-end outliers, but these were deemed natural representations of high-value and digitally active customers respectively — no treatment was applied.

**Feature Scaling:** Prior to clustering, the five behavioural features (`Avg_Credit_Limit`, `Total_Credit_Cards`, `Total_visits_bank`, `Total_visits_online`, `Total_calls_made`) were standardised using `StandardScaler` (Z-score normalisation) to ensure no single feature dominated distance calculations due to its scale.

### 4. K-Means Clustering

**Optimal K Selection:**

The Elbow Method was applied across k = 1 to 14, plotting average distortion at each value. The curve exhibited a clear inflection point at **k = 3**, where the distortion score was 933.044 — confirmed visually using `KElbowVisualizer`.

Silhouette scores were computed for k = 2 to 14:
- k = 2: silhouette score = 0.5703 (highest)
- k = 3: silhouette score = 0.5157 (reasonably high)
- Scores decrease steadily beyond k = 3

Silhouette coefficient plots for k = 2 and k = 3 confirmed that k = 3 yields interpretable, well-separated clusters with a sufficiently high silhouette score to justify the added segmentation granularity over k = 2. **k = 3 was selected as the final number of clusters.**

**Cluster Profiling (K-Means, k = 3):**

| Cluster | Size | Avg Credit Limit | Total Cards | Bank Visits | Online Visits | Calls Made | Segment Label |
|---|---|---|---|---|---|---|---|
| 0 | 386 | ₹33,782 | 5.5 | 3.5 | 0.98 | 2.0 | Medium-Value, Branch-Oriented |
| 1 | 224 | ₹12,174 | 2.4 | 0.93 | 3.56 | 6.87 | Low-Value, High-Support Seekers |
| 2 | 50 | ₹1,41,040 | 8.7 | 0.60 | 10.9 | 1.08 | High-Value, Digital Champions |

- **Cluster 0:** Medium credit limit customers with a moderate number of credit cards. They tend to visit the bank in-person more than using online services or calling — a branch-reliant segment.
- **Cluster 1:** Low credit limit customers with fewer credit cards. They prefer remote channels and make frequent calls to customer service, likely reflecting unresolved support needs or lower financial confidence.
- **Cluster 2:** High-value customers with large credit limits and many cards. They predominantly use online channels and rarely visit or call — indicating high digital engagement and financial autonomy.

### 5. Hierarchical Clustering

**Cophenetic Correlation Analysis:**

To select the best linkage method, cophenetic correlations were computed for 4 distance metrics (Euclidean, Chebyshev, Mahalanobis, Cityblock) × 4 linkage methods (single, complete, average, weighted):

The highest cophenetic correlation was **0.8977**, achieved with **Euclidean distance and average linkage** — confirming this combination best preserves the data's hierarchical clustering structure.

When further compared across all linkage methods with Euclidean distance (including centroid and ward), average linkage retained the top cophenetic correlation of 0.8977, followed by centroid (0.8939) and weighted (0.8862).

**Dendrogram Analysis:**

Six dendrograms were generated (single, complete, average, centroid, ward, weighted). Visual inspection confirmed that **Ward linkage** produced the most distinct, well-separated clusters despite its lower cophenetic correlation (0.74) — this highlights the trade-off between mathematical structure preservation (cophenetic) and visual cluster separability.

From the Ward linkage dendrogram, **4 clusters** was identified as the appropriate cut point.

**Final Hierarchical Model:** `AgglomerativeClustering(n_clusters=4, linkage='average')` was fitted on the scaled data.

**Cluster Profiling (Hierarchical, k = 4):**

| Cluster | Size | Avg Credit Limit | Total Cards | Bank Visits | Online Visits | Calls Made | Segment Label |
|---|---|---|---|---|---|---|---|
| 0 | 223 | ₹12,197 | 2.4 | 0.93 | 3.56 | 6.88 | Low-Credit, Support-Heavy |
| 1 | 50 | ₹1,41,040 | 8.7 | 0.60 | 10.9 | 1.08 | High-Value, Digital-First |
| 2 | 386 | ₹33,541 | 5.5 | 3.49 | 0.98 | 2.01 | Mid-Range, Branch-Active |
| 3 | 1 | ₹1,00,000 | 2.0 | 1.0 | 1.0 | 0.0 | Outlier / Ultra-Niche |

- **Cluster 0:** Low-credit-limit customers with fewer cards, moderate online activity, and high call frequency — suggesting a strong need for enhanced support services.
- **Cluster 1:** High-spending, multi-card customers with high credit limits and infrequent interactions overall, but more online engagement — the bank's most valuable and self-sufficient segment.
- **Cluster 2:** Mid-range customers with moderate credit limits, notably high in-person bank visits, and limited online interaction — a traditional banking segment.
- **Cluster 3:** A single-customer outlier with high credit limit but minimal interaction — edge case warranting individual relationship management.

### 6. Algorithm Comparison: K-Means vs Hierarchical

| Metric | K-Means | Hierarchical (Average Linkage) |
|---|---|---|
| Optimal Clusters | 3 | 4 |
| Evaluation Metric | Silhouette Score: 0.516 (k=3) | Cophenetic Correlation: 0.898 |
| Execution Time | 0.0103 seconds | 0.0137 seconds |
| Cluster Interpretability | Good | Better (via dendrogram) |
| Cluster Separation | Moderate | More distinct (Ward linkage) |

K-Means is computationally faster and adequate for real-time segmentation. Hierarchical clustering — especially with Ward linkage — yields more distinct and interpretable customer segments for strategic planning.

---

## Key Results

- **K-Means** identified 3 customer segments: medium-value branch visitors, low-value high-callers, and high-value digital champions.
- **Hierarchical Clustering** identified 4 segments with finer granularity, separating a very high credit limit outlier from the high-value digital group.
- The highest **silhouette score** achieved was **0.5703** (k=2) and **0.5157** (k=3) for K-Means.
- The highest **cophenetic correlation** was **0.8977** for average linkage with Euclidean distance.
- Hierarchical clustering with Ward linkage produced the most visually distinct dendrogram clusters.
- K-Means was **24.8% faster** in execution time (0.0103s vs 0.0137s).

---

## Business Impact

**1. Personalised Marketing Campaigns by Segment**
High-value customers (Cluster 2 / HC Cluster 1) — with large credit limits and strong digital engagement — are prime candidates for exclusive premium card offers, higher credit limit upgrades, and digital loyalty programmes. Marketing budgets directed here will yield the highest ROI.

**2. Targeted Support Reduction for Low-Value Callers**
Low-credit, high-call customers (K-Means Cluster 1 / HC Cluster 0) represent a service cost burden. Proactive outreach, self-service FAQs, AI chatbot integration, and improved onboarding can reduce inbound call volume and improve satisfaction without increasing headcount.

**3. Digital-First Service Strategy**
The low frequency of in-person bank visits across all segments confirms that strengthening the online portal and mobile app is the highest-impact service investment. Features like live chat, AI-driven query resolution, and digital statements reduce reliance on physical branches and call centers.

**4. Branch Experience Optimisation for Mid-Range Customers**
Mid-range customers (K-Means Cluster 0 / HC Cluster 2) who visit branches frequently represent cross-sell opportunities. In-branch personalised consultations and product recommendations for credit card upgrades or additional financial products can increase wallet share.

**5. Upsell and Cross-Sell through Behavioural Signals**
Customers with more credit cards make significantly fewer calls — signalling higher financial literacy and engagement. These customers can be targeted with investment products, premium insurance, or co-branded card partnerships.

---

## Skills

### Technical Skills
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC1D?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Exploratory Data Analysis](https://img.shields.io/badge/Data_Analysis-FFA500?style=for-the-badge&logo=google-analytics&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

### Soft Skills
![Analytical Thinking](https://img.shields.io/badge/Analytical_Thinking-4B0082?style=for-the-badge&logo=mindmap&logoColor=white)
![Communication](https://img.shields.io/badge/Communication-25D366?style=for-the-badge&logo=google-messages&logoColor=white)
![Problem Solving](https://img.shields.io/badge/Problem_Solving-FF4500?style=for-the-badge&logo=brainly&logoColor=white)
![Attention to Detail](https://img.shields.io/badge/Attention_to_Detail-00CED1?style=for-the-badge&logo=google-search-console&logoColor=white)

---

## Key Learnings

- **Choosing k is both science and art:** The Elbow Method and Silhouette scores can point to different optimal k values (k=2 vs k=3 in this project). Domain context — specifically the need for actionable and differentiated marketing segments — should guide the final choice over pure metric optimisation.
- **Cophenetic correlation ≠ best visual clusters:** Average linkage yielded the highest cophenetic correlation (0.898), yet Ward linkage produced more visually interpretable and well-separated dendrograms. Both metrics serve different purposes and should be evaluated together.
- **Scaling is non-negotiable for distance-based algorithms:** Without `StandardScaler`, the high-variance `Avg_Credit_Limit` feature (range: ₹3,000–₹2,00,000) would dominate Euclidean distances and render clustering results meaningless.
- **Outlier handling in clustering requires domain judgement:** High credit limit values that appear as statistical outliers are genuinely important business records representing premium customers — blindly treating them would lose critical segment information.
- **Cluster profiling is where insights live:** Raw cluster labels are meaningless without systematic profiling (mean aggregation + visualisation). The boxplots and cluster average tables transformed mathematical output into business narratives.

---

## Future Improvements

1. **Incorporate transactional data:** Adding monthly spend amounts, payment history, and default rates would dramatically enrich customer profiles and enable risk-adjusted segmentation.

2. **DBSCAN for outlier-aware clustering:** The single-customer Cluster 3 in hierarchical clustering highlights a limitation of partition-based methods. DBSCAN would naturally handle outliers without forcing them into artificial clusters.

3. **Dimensionality reduction with PCA or t-SNE:** Reducing features to 2–3 principal components before clustering could improve cluster separation and enable richer visual exploration of the data space.

4. **Automated cluster labelling with LLMs:** Fine-tuned language models could automatically generate human-readable segment labels and marketing copy from cluster profiles, accelerating deployment.

5. **Real-time segmentation pipeline:** Deploying the final K-Means model as a REST API endpoint would allow the CRM system to classify new customers in real-time upon onboarding, enabling immediate personalised targeting.

---

## Author

**Nabankur Ray**

Passionate about real-world data-driven solutions

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nabankur-ray-876582181/)

![GitHub Stats](https://github-readme-stats-eight-theta.vercel.app/api?username=nabankur14&show_icons=true)

---

⭐ If you like this project — give it a ⭐ on GitHub — it helps a lot!