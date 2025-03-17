import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np

df = pd.read_csv('attack_dataset.csv')
model = RandomForestClassifier()
encoder = OneHotEncoder()

#feature engineering
encoded_array = encoder.fit_transform(df[[' Label']]).toarray()
feature_names = encoder.get_feature_names_out([' Label'])
encoded_df = pd.DataFrame(encoded_array, columns = feature_names)
df = pd.concat([df.drop(columns=[' Label']), encoded_df], axis = 1)
print(df.columns)

#visualing relationships
# ' Label_Web Attack � Brute Force', ' Label_Web Attack � Sql Injection', ' Label_Web Attack � XSS'
correlation_matrix = df.corr()
plt.figure(figsize=(24, 12))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # You can change 'coolwarm' to other colormaps
plt.title('Correlation Heatmap of Features')
plt.show()

#finding relevant features for feature selection in terms of Brute Force Web Attacks
target_correlation = correlation_matrix[' Label_Web Attack � Brute Force'].sort_values(ascending=False)
print("\nCorrelation with Brute Force Web Attacks:\n", target_correlation)
relevant_features = correlation_matrix[target_correlation > 0]


