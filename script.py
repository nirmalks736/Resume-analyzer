import pandas as pd  

# Load the dataset
df = pd.read_csv('./archive (1)/Resume/Resume.csv')  # Update path if needed

# Display the first 5 rows
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Plot the category distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df["Category"], order=df["Category"].value_counts().index, palette="viridis")

# Add labels
plt.xlabel("Number of Resumes")
plt.ylabel("Job Category")
plt.title("Distribution of Resume Categories")

# Show plot
plt.show()
