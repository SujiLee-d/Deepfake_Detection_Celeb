# Stacking Classifier - Xception(CNN)+XGBoost(mean) // not complete
# by Suji Lee

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Data Loading and Preprocessing
base_dir = '/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177'
subdirs = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']
data_frames = []

for subdir in subdirs:
    path = os.path.join(base_dir, 'CHECKPOINT_DST', subdir)
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            df = pd.read_pickle(os.path.join(path, file))
            data_frames.append(df)

# Concatenate all DataFrames into one
df_full = pd.concat(data_frames, ignore_index=True)
X_preprocessed = df_full.drop(columns=['class', 'original', 'videosubject'])  # Drop non-numeric columns


# 1 feature vector : 1 video by grouping ('video' column)
df_features = X_preprocessed.groupby('video').mean().reset_index()
X_features = df_features.drop(columns=['label', 'video']).values
y = df_features['label'].values

# Get Image paths for CNN
def get_image_paths_for_video(base_dir, video_id, subdirs):
    paths = []
    for subdir in subdirs:
        video_folder_path = os.path.join(base_dir, 'FACES_DST', subdir, video_id)
        if os.path.isdir(video_folder_path):
            for img_file in os.listdir(video_folder_path):
                if img_file.endswith('.jpg'):
                    paths.append(os.path.join(video_folder_path, img_file))
    return paths

# CNN Input Data Formatting for each video
X_img_paths = [get_image_paths_for_video(base_dir, video_id, subdirs) for video_id in df_features['video']]

# Train/Test Split
X_img_train, X_img_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_img_paths, X_features, y, test_size=0.2, random_state=42
)

# Definition of Xception Model wrapper
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf

class XceptionModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = Xception(weights='imagenet', include_top=False, pooling='avg')

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_processed = []
        for img_paths in X:  # Image path list 
            video_images = [img_to_array(load_img(img, target_size=(299, 299))) for img in img_paths]
            video_images = np.array(video_images)
            video_images = tf.keras.applications.xception.preprocess_input(video_images)

            # Mean vector calculation
            features = self.model.predict(video_images)
            video_features = np.mean(features, axis=0)
            X_processed.append(video_features)
        return np.array(X_processed)

# Ensemble (Xception + XGBoost) 

# Xception - sklearn compatibility
xception_clf = XceptionModel()

# XGBoost
xgboost_clf = XGBClassifier()

# Ensemble with StackingClassifier
estimators = [
    ('xception', xception_clf),
    ('xgboost', xgboost_clf)
]
meta_clf = LogisticRegression()

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_clf, stack_method='predict_proba')

# Fit / Learn
stacking_clf.fit([X_img_train, X_features_train], y_train)  # both Image Path & Feature Vector
y_pred = stacking_clf.predict([X_img_test, X_features_test])

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy of Stacking model: {accuracy:.2f}")
print("Confusion Matrix of Stacking model:")
print(conf_matrix)
