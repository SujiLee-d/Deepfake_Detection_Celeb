
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load all pickle files
base_dir = '/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/CHECKPOINT_DST'
sub_dirs = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']  
all_data = []

for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read pkl files and add as DataFrame
                df = pd.read_pickle(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"An error occurred while reading {file_path}: {e}")

# Concatenate all DataFrames into one
df = pd.concat(all_data, ignore_index=True)

X_preprocessed = df.drop(columns=['class', 'original', 'videosubject'])  # Drop non-numeric columns

# 1 feature vector : 1 video by grouping ('video' column)

# Select statistical feature: mean / sum / median
grouped_df = X_preprocessed.groupby('video').mean()  #0.75
# grouped_df = X_preprocessed.groupby('video').sum() #0.75
# grouped_df = X_preprocessed.groupby('video').median()   #0.74


# first value of 'label' column for prediction result used
labels = X_preprocessed.groupby('video')['label'].first()

# Define feature vectors and labels
X = grouped_df.drop(columns=['label'])  
y = labels  
# debugging
# print(X.info)
# print(y.head(150))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# XGBoost Model
xgb_model = XGBClassifier(eval_metric='logloss')

# Model fitting (learning)
xgb_model.fit(X_train, y_train) 

# Prediction with test data
y_pred = xgb_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of XGBoost_mean model: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion Matrix visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for XGBoost Model')
plt.show()
