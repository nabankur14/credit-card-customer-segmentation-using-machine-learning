<h1 align="center" style="color:#2b7a78;">Customer Segmentation Using Unsupervised Learning – AllLife Bank</h1>
<h3 align="center" style="color:#17252a;">Segmenting Credit Card Customers Using K-Means and Hierarchical Clustering for Targeted Marketing and Service Optimization</h3>

<p align="center">
  <strong>Author:</strong> <a href="https://github.com/nabankur14" target="_blank" style="color:#3aafa9;">Nabankur Ray</a>
</p>

<hr>

<h2 style="color:#17252a;">Overview</h2>
<p>
This project applies <strong>Unsupervised Machine Learning</strong> techniques to segment <strong>credit card customers</strong> of <strong>AllLife Bank</strong> based on their spending behavior and interaction patterns. 
The goal is to identify distinct customer groups to enable <strong>personalized marketing strategies</strong>, <strong>service improvements</strong>, and <strong>revenue optimization</strong>. 
Comprehensive <em>EDA</em>, <em>feature scaling</em>, <em>clustering validation</em>, and <em>business insight generation</em> were performed to support data-driven decision-making.
</p>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Objective</summary>
  <ul>
    <li>Segment credit card customers based on behavioral and financial attributes.</li>
    <li>Apply <strong>K-Means</strong> and <strong>Hierarchical Clustering</strong> to uncover hidden customer patterns.</li>
    <li>Validate cluster quality using <strong>Elbow</strong>, <strong>Silhouette</strong>, and <strong>Cophenetic Correlation</strong> methods.</li>
    <li>Profile each cluster to guide targeted marketing and customer retention initiatives.</li>
    <li>Provide actionable insights to enhance customer experience and operational efficiency.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Dataset</summary>
  <ul>
    <li><strong>Source:</strong> AllLife Bank’s credit card customer database.</li>
    <li><strong>Records:</strong> 660 customers.</li>
    <li><strong>Variables:</strong>
      <ul>
        <li><code>Customer Key</code> – Unique identifier for each customer.</li>
        <li><code>Avg_Credit_Limit</code> – Average credit card limit.</li>
        <li><code>Total_Credit_Cards</code> – Number of credit cards held.</li>
        <li><code>Total_visits_bank</code> – Number of branch visits.</li>
        <li><code>Total_visits_online</code> – Number of online portal visits.</li>
        <li><code>Total_calls_made</code> – Number of calls to customer support.</li>
      </ul>
    </li>
    <li><strong>Data Quality:</strong> No missing values; numeric variables standardized for clustering.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Methodology</summary>
  <ol>
    <li><strong>Data Preprocessing:</strong>
      <ul>
        <li>Checked for missing values and outliers (none removed as outliers were genuine data points).</li>
        <li>Standardized numeric features using scaling for clustering algorithms.</li>
      </ul>
    </li>
    <li><strong>Exploratory Data Analysis (EDA):</strong>
      <ul>
        <li>Univariate and bivariate analysis using histograms, boxplots, and pairplots.</li>
        <li>Correlation heatmap to identify relationships between key features.</li>
      </ul>
    </li>
    <li><strong>Model Development:</strong>
      <ul>
        <li><strong>K-Means Clustering:</strong> Determined optimal clusters using the <em>Elbow</em> and <em>Silhouette</em> methods.</li>
        <li><strong>Hierarchical Clustering:</strong> Used multiple linkage methods; <em>Ward linkage</em> with Euclidean distance performed best.</li>
        <li>Cluster profiles created to interpret behavioral patterns.</li>
      </ul>
    </li>
    <li><strong>Validation Metrics:</strong> Elbow curve, Silhouette score, and Cophenetic correlation for hierarchical models.</li>
  </ol>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Tools & Technologies</summary>
  <p>
  <code>Python</code>, <code>Pandas</code>, <code>NumPy</code>, <code>Scikit-learn</code>, 
  <code>Matplotlib</code>, <code>Seaborn</code>, <code>Scipy</code>, <code>Jupyter Notebook</code>
  </p>
</details>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Results & Insights</summary>
  <ul>
    <li><strong>K-Means Clustering:</strong> Optimal number of clusters = 3.</li>
    <li><strong>Hierarchical Clustering (Ward):</strong> Suggested 4 clusters with strong cophenetic correlation (~0.898).</li>
    <li><strong>Cluster Profiles:</strong>
      <ul>
        <li>Cluster 0 – Medium credit limit, frequent in-branch visitors.</li>
        <li>Cluster 1 – Low credit limit, high online and call activity (support-seeking customers).</li>
        <li>Cluster 2 – High credit limit, tech-savvy and digitally active customers.</li>
      </ul>
    </li>
    <li><strong>Business Recommendations:</strong>
      <ul>
        <li>Introduce premium offers for high-value digital customers.</li>
        <li>Enhance call-center support for frequently contacting customers.</li>
        <li>Promote online banking features for in-branch heavy users.</li>
      </ul>
    </li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Future Scope</summary>
  <ul>
    <li>Deploy clustering results in CRM dashboards for real-time segmentation.</li>
    <li>Integrate demographics and transaction data for deeper segmentation.</li>
    <li>Use advanced algorithms like <strong>DBSCAN</strong> or <strong>Gaussian Mixture Models</strong> for better boundary flexibility.</li>
    <li>Apply predictive modeling to anticipate customer churn and retention probability.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Key Learnings</summary>
  <ul>
    <li>Developed expertise in <strong>Unsupervised Learning</strong> and <strong>Clustering Validation</strong>.</li>
    <li>Learned to translate data-driven insights into <strong>business strategies</strong> and <strong>marketing actions</strong>.</li>
    <li>Strengthened EDA, visualization, and feature scaling skills.</li>
    <li>Enhanced understanding of <strong>hierarchical linkage methods</strong> and their interpretability.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">Folder Structure</summary>
  <pre style="background:#f0f0f0; padding:10px; border-radius:8px;">
  ul_coded_project_customer_segmentation/
  │
  ├── Customer_Segmentation.ipynb             → Main Jupyter Notebook (EDA + Clustering)
  ├── Customer_Segmentation_Report.pdf            → Detailed analytical & business report
  ├── Credit_Card_Customer_Data.csv                               → Credit card customer dataset
  └── README.md                                       → Project documentation (this file)
  </pre>
</details>

<p align="center" style="color:#555;">
>>> All project files are organized and accessible for easy reproducibility and reference.
</p>

<h2 style="color:#17252a;">#Tags</h2>
<p>
#CustomerSegmentation #UnsupervisedLearning #KMeans #HierarchicalClustering #EDA #Python #MachineLearning #Analytics #Banking #DataScience #CustomerInsights #MarketingOptimization
</p>

<hr>
<p align="center" style="font-size:14px; color:#555;">
© 2025 <strong>Nabankur Ray</strong> | Data Scientist
</p>
