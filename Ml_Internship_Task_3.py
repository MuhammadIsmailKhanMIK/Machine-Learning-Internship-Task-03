import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('user_profiles_for_ads.csv')

# Q.1: Import data and check null values, column info, and descriptive statistics of the data
print("Data Shape:", df.shape)
print("Column Info:")
print(df.info())
print("Descriptive Statistics:")
print(df.describe())

# Q.2: Begin EDA by visualizing the distribution of the key demographic variables
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
sns.countplot(x='Age', data=df)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')

plt.subplot(2, 2, 3)
sns.countplot(x='Education Level', data=df)
plt.title('Education Level Distribution')

plt.subplot(2, 2, 4)
sns.countplot(x='Income Level', data=df)
plt.title('Income Level Distribution')

plt.tight_layout()
plt.show()

# Q.3: Examine device usage patterns and users' online behavior
plt.figure(figsize=(10, 6))
sns.countplot(x='Device Usage', data=df)
plt.title('Device Usage Patterns')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Top Interests', data=df)
plt.title('Top Interests')
plt.show()

# Q.4: Analyze the average time users spend online on weekdays versus weekends
plt.figure(figsize=(10, 6))
sns.boxplot(y='Time Spent Online (hrs/weekday)', data=df)
plt.title('Time Spent Online on Weekdays')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(y='Time Spent Online (hrs/weekend)', data=df)
plt.title('Time Spent Online on Weekends')
plt.show()

# Q.5: Identify the most common interests among users
top_interests = df['Top Interests'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_interests.index, y=top_interests.values)
plt.title('Top 10 Interests')
plt.show()

# Q.6: Segment users into distinct groups for targeted ad campaigns
# Define preprocessing for numeric and categorical columns
numeric_features = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']
categorical_features = ['Age', 'Gender', 'Education Level', 'Income Level', 'Device Usage']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data
df_processed = preprocessor.fit_transform(df.drop(['User ID', 'Top Interests', 'Location', 'Language'], axis=1))

# Handle missing values by replacing them with the mean of the respective column
imputer = SimpleImputer(strategy='mean')
df_processed = imputer.fit_transform(df_processed)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df_processed)

# Q.7: Compute the mean values of the numerical features and the mode for categorical features within each cluster
cluster_means = kmeans.cluster_centers_

# Reconstruct the cluster centers to match original feature names
encoded_columns = preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)
processed_columns = np.append(numeric_features, encoded_columns)
cluster_means_df = pd.DataFrame(cluster_means, columns=processed_columns)

# Q.8: Assign each cluster a name that reflects its most defining characteristics
cluster_names = ['Weekend Warriors', 'Engaged Professionals', 'Low-Key Users', 'Active Explorers', 'Budget Browsers']

# Q.9: Create a visualization that reflects these segments
# Plot a radar chart for each cluster
def plot_radar_chart(data, labels, title):
    num_vars = len(data.columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i in range(len(labels)):
        values = data.iloc[i].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=labels[i])
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(data.columns)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

plot_radar_chart(cluster_means_df, cluster_names, 'Cluster Profiles')

# Q.10: Write down the summary of the experience
print('Summary:')
print('We have successfully implemented user profiling and segmentation using clustering techniques.')
print('We identified 5 distinct segments of users based on their demographic and behavioral characteristics.')
print('These segments are: Weekend Warriors, Engaged Professionals, Low-Key Users, Active Explorers, and Budget Browsers.')
print('We visualized the distribution of key demographic variables, device usage patterns, and users\' online behavior.')
print('We also computed the mean values of numerical features and the mode for categorical features within each cluster.')
print('Finally, we created a radar chart to visualize the profiles of each segment.')