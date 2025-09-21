# E-commerce-Customer-Segmentation
This project uses Python to perform customer segmentation on a simulated e-commerce dataset. The objective is to identify distinct customer groups based on their purchasing behavior to enable targeted marketing strategies.
# This script uses Python to perform customer segmentation on simulated data.
# It demonstrates skills in using libraries like pandas and scikit-learn.

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_data():
    """Generates a sample DataFrame for e-commerce customer data."""
    np.random.seed(42)
    data = {
        'customer_id': range(1, 101),
        'total_spent': np.random.normal(loc=500, scale=200, size=100).round(2),
        'purchase_frequency': np.random.randint(1, 20, size=100)
    }
    df = pd.DataFrame(data)
    # Introduce some clear clusters
    df.loc[df['customer_id'] > 75, 'total_spent'] += 1500
    df.loc[df['customer_id'] > 75, 'purchase_frequency'] += 10
    return df

def main():
    """Main function to perform customer segmentation."""
    # Step 1: Generate or load the dataset
    df = generate_sample_data()
    print("Simulated E-commerce Customer Data (first 5 rows):")
    print(df.head().to_markdown(index=False))
    print("\n" + "="*50 + "\n")

    # Step 2: Select features for clustering
    features = df[['total_spent', 'purchase_frequency']]

    # Step 3: Apply K-Means clustering
    # We'll use 3 clusters for demonstration purposes
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features)

    print("Customer Segmentation Results (first 5 rows with cluster ID):")
    print(df.head().to_markdown(index=False))
    print("\n" + "="*50 + "\n")

    # Step 4: Analyze the clusters and provide insights
    cluster_summary = df.groupby('cluster')[['total_spent', 'purchase_frequency']].mean()
    print("Cluster Analysis:")
    print(cluster_summary.to_markdown())

    # Example actionable recommendations based on clusters
    print("\nActionable Recommendations:")
    print("- Cluster 0 (Loyal High-Spenders): Target with exclusive loyalty programs and high-value offers.")
    print("- Cluster 1 (New or Inactive): Offer introductory discounts or special promotions to encourage engagement.")
    print("- Cluster 2 (Frequent Shoppers): Recommend new products and cross-sell related items to increase average order value.")
    
    # Step 5: Visualize the clusters
    plt.figure(figsize=(10, 6))
    for cluster_id, color in zip(range(3), ['red', 'green', 'blue']):
        cluster_data = df[df['cluster'] == cluster_id]
        plt.scatter(cluster_data['total_spent'], cluster_data['purchase_frequency'], 
                    c=color, label=f'Cluster {cluster_id}')
    
    plt.title('Customer Segmentation using K-Means Clustering')
    plt.xlabel('Total Spent')
    plt.ylabel('Purchase Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
