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
# plt.show()
#features are too many making visualising relationships difficult

#finding relevant features for feature selection in terms of Brute Force Web Attacks
target_correlation = correlation_matrix[' Label_Web Attack � Brute Force'].sort_values(ascending=False)
print("\nCorrelation with Brute Force Web Attacks:\n", target_correlation)
brute_relevant_features = target_correlation[target_correlation > 0]
brute_relevant_features = brute_relevant_features.index.tolist()
brute_positive_df = df[brute_relevant_features]
brute_positive_df[' Label_Web Attack � Brute Force'] = df[' Label_Web Attack � Brute Force']

#splitting data into adequate portions
train_data, temp_data = train_test_split(brute_positive_df, test_size=0.3)
val_data, test_data = train_test_split(temp_data, test_size=0.5)

#splitting training data into x and y values
x_train = train_data.drop(columns=[' Label_Web Attack � Brute Force'])
y_train = train_data[' Label_Web Attack � Brute Force']

x_val = val_data.drop(columns=[' Label_Web Attack � Brute Force'])
y_val = val_data[' Label_Web Attack � Brute Force']

#training model
model.fit(x_train, y_train)

y_pred = model.predict(x_val)

accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Model Accuracy for Brute Force Web Attacks: ", accuracy)
print("Classification Report for Brute Force Web Attacks", report)

cm = confusion_matrix(y_val, y_pred)
plt.figure()
sns.heatmap(cm, cmap='coolwarm', annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#ROC_Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
