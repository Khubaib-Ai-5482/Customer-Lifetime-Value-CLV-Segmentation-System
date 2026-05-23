import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Online Retail.csv")

df = df.dropna(subset=["CustomerID"])

df = df[df["Quantity"] > 0]

df = df[df["UnitPrice"] > 0]

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=True
)

# print(df.info())
# print(df.head())

latest_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (latest_date - x.max()).days,
    "InvoiceNo" : "nunique",
    "Totalsale" : "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4,random_state=42)

rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby("cluster").mean()

print(cluster_summary)

def label_cluster(row):
    if row["cluster"] == cluster_summary["Monetary"].idxmax():
        return "VIP Customers"
    
    elif row["cluster"] == cluster_summary["Recency"].idxmax():
        return "Lost Customers"
    
    elif row["cluster"] == cluster_summary["Frequency"].idxmax():
        return "Loyal Customers"
    
    else:
        return "Regular Customers"

rfm["Segment"] = rfm.apply(label_cluster, axis=1)

print(rfm["Segment"].value_counts())

rfm["CLV_Score"] = (
    rfm["Monetary"] * 0.5 +
    rfm["Frequency"] * 0.3 -
    rfm["Recency"] * 0.2
)

top_customers = rfm.sort_values("CLV_Score", ascending=False).head(10)

print(top_customers)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=rfm["Recency"],
    y=rfm["Monetary"],
    hue=rfm["Segment"]
)
plt.title("Customer Segments")
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(rfm["CLV_Score"], bins=50, kde=True)
plt.title("CLV Score Distribution")
plt.show()

rfm.to_csv("CLV_Final_Project.csv")
print("Project Completed & Saved!")