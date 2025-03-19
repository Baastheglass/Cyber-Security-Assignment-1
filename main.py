import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np

df = pd.read_csv('attack_dataset.csv')
model = RandomForestClassifier()
model2 = GradientBoostingClassifier()
model3 = AdaBoostClassifier()
model4 = DecisionTreeClassifier()
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
# plt.figure(figsize=(24, 12))  # Adjust the figure size as needed
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # You can change 'coolwarm' to other colormaps
# plt.title('Correlation Heatmap of Features')
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
train_data, temp_data = train_test_split(brute_positive_df, test_size = 0.3)
val_data, test_data = train_test_split(temp_data, test_size = 0.5)

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
#plt.show()

#ROC_Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
#plt.show()

#finding relevant features for feature selection in terms of SQL injections
target_correlation = correlation_matrix[' Label_Web Attack � Sql Injection'].sort_values(ascending=False)
sql_relevant_features = target_correlation[target_correlation > 0].index.tolist()
sql_df = df[sql_relevant_features]
sql_df[' Label_Web Attack � Sql Injection'] = df[' Label_Web Attack � Sql Injection']

#splitting data
train_data, temp = train_test_split(sql_df, test_size = 0.3)
val_data, tmst_data = train_test_split(temp, test_size = 0.5)

x_train = train_data.drop(columns=[' Label_Web Attack � Sql Injection'])
y_train = train_data[' Label_Web Attack � Sql Injection']

x_val = val_data.drop(columns=[' Label_Web Attack � Sql Injection'])
y_val = val_data[' Label_Web Attack � Sql Injection']

#training model
model2.fit(x_train, y_train)

#predicting results
y_pred = model2.predict(x_val)

#outputting results

accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Model Accuracy for SQL Injection Attacks: ", accuracy)
print("Classification Report for SQL Injection Attacks", report)

cm = confusion_matrix(y_val, y_pred)
plt.figure()
sns.heatmap(cm, cmap='coolwarm', annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

#ROC_Curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
#plt.show()

#' Label_Web Attack � XSS'
#finding relevant features for feature selection in terms of XSS web attacks

target_correlation = correlation_matrix[' Label_Web Attack � XSS'].sort_values(ascending=False)
xss_features = target_correlation[target_correlation > 0].index.tolist()
xss_df = df[xss_features] 
xss_df[' Label_Web Attack � XSS'] = df[' Label_Web Attack � XSS']

#splitting data
train_data, temp = train_test_split(xss_df, test_size = 0.5)
val_data, test_data = train_test_split(temp,test_size = 0.5)

x_train = train_data.drop(columns=[' Label_Web Attack � XSS'])
y_train = train_data[' Label_Web Attack � XSS']

x_val = val_data.drop(columns=[' Label_Web Attack � XSS'])
y_val = val_data[' Label_Web Attack � XSS']

#training model
model3.fit(x_train, y_train)

#predicting values
y_pred = model3.predict(x_val)

#generating report
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print("Model Accuracy for XSS Web Attacks: ", accuracy)
print("Classification Report for XSS Web Attacks", report)

#confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure()
sns.heatmap(cm, cmap='coolwarm', annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

#roc_curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
#plt.show()

#' Label_BENIGN'
#finding relevant features for feature selection in terms of BENIGN attacks
target_correlation = correlation_matrix[' Label_BENIGN'].sort_values(ascending=False)
benign_features = target_correlation[target_correlation > 0].index.to_list()
benign_df = df[benign_features]
benign_df[' Label_BENIGN'] = df[' Label_BENIGN']

#splitting data
train_data, temp = train_test_split(benign_df, test_size=0.3)
val_data, test_data = train_test_split(temp, test_size=0.5)

x_train = train_data.drop(columns=[' Label_BENIGN'])
y_train = train_data[' Label_BENIGN']

x_val = val_data.drop(columns=[' Label_BENIGN'])
y_val = val_data[' Label_BENIGN']

x_train = x_train.replace([np.inf, -np.inf], np.nan)  # Convert Inf to NaN
x_train = x_train.fillna(x_train.mean())  # Fill NaN with mean

x_val = x_val.replace([np.inf, -np.inf], np.nan)  # Convert Inf to NaN
x_val = x_val.fillna(x_val.mean())  # Fill NaN with mean

#training the model
model4.fit(x_train,y_train)

#making predictions
y_pred = model4.predict(x_val)

#outputting results
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val,y_pred)

print("Model Accuracy for Benign Attacks: ", accuracy)
print("Classification Report for Benign Attacks", report)

#confusion matrix
cm = confusion_matrix(y_val,y_pred)
plt.figure()
sns.heatmap(cm, cmap='coolwarm', annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

#roc curve
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
#plt.show()
