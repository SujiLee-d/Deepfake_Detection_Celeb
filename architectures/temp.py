# Ensemble – Stacking Code example

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 분할
# df_faces = pd.read_pickle("/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl")

# X = df_faces[['height', 'width', 'frames', 'nfaces']]
# y = df_faces['label']

# 1. 데이터 로드 (IMG)
df_faces = pd.read_pickle("/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl")

# 이미지가 저장된 기본 경로
base_image_dir = '/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACES_DST/'

# 'path' 열을 사용해 이미지 경로 리스트를 생성
df_faces['image_paths'] = df_faces['path'].apply(lambda x: [os.path.join(base_image_dir, x, img) for img in os.listdir(os.path.join(base_image_dir, x)) if img.endswith(".jpg")])

# X에는 이미지 경로 리스트를, y에는 label을 설정
X = df_faces['image_paths'].tolist()
y = df_faces['label'].tolist()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 확인
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print("Example of X_train:", X_train[0])  # X_train의 첫 번째 샘플 확인
print("Example of y_train:", y_train[0])  # y_train의 첫 번째 라벨 확인

# 2. 기본 학습자 (Base Learners)
# RandomForest, GradientBoosting, Logistic Regression을 기본 학습자로 사용
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_clf = LogisticRegression(max_iter=1000)

# 3. Voting Classifier (앙상블)
# hard voting 방식으로 예측값을 결합합니다.
voting_clf = VotingClassifier(estimators=[('rf', rf_clf), ('gb', gb_clf), ('lr', lr_clf)], voting='hard')
# 'rf', 'gb', 'lr' = 부여한 이름. 
# 이 이름들은 나중에 필요할 때 개별 모델에 접근하거나, 앙상블의 예측 결과를 분석할 때 사용할 수 있습니다.
# 예를 들어, 앙상블 모델의 특정 기본 학습자를 출력하거나 성능을 확인할 때 이 이름들이 유용합니다.
# voting_clf.named_estimators_['rf']


# 4. 학습
voting_clf.fit(X_train, y_train)

# 5. 예측 및 평가
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'앙상블 모델의 정확도: {accuracy:.2f}')

#---------------------------------------------------------------------------------
# Stacking 기법 예시 (Xception, EfficientNetB4 사용):
from sklearn.ensemble import StackingClassifier
from tensorflow.keras.applications import Xception, EfficientNetB4
from sklearn.linear_model import LogisticRegression

# CNN 모델 (기본 학습자) 정의
xception = Xception(weights='imagenet', include_top=False, pooling='avg')
efficientnet = EfficientNetB4(weights='imagenet', include_top=False, pooling='avg')

# Base Learners 설정 (필요한 전처리 추가)
estimators = [
    ('xception', xception),
    ('efficientnet', efficientnet),
]

# Meta Estimator 정의
final_estimator = LogisticRegression()

# StackingClassifier 생성
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# 모델 학습 및 평가
stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Stacking 모델의 정확도: {accuracy:.2f}')
