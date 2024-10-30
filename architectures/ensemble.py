# Ensemble - Stacking method trial 1
# Video Face Manipulation Detection Through Ensemble of CNNs

import sys
sys.path.append('/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/PGM/DeepFakeDetection_icpr2020dfdc')

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import pandas as pd
from architectures import fornet,weights
from architectures.fornet import FeatureExtractor

# added
from fornet import XceptionST, EfficientNetAutoAttB4ST

# PyTorch 모델을 scikit-learn과 호환하도록 래핑하는 클래스
class SklearnCompatibleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X, y):
        # 이 부분에 PyTorch 모델의 학습 로직을 추가합니다.
        pass

    def predict(self, X):
        self.model.eval()  
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device) 
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

# 모델 설정 및 래핑
xception_st = SklearnCompatibleModel(XceptionST())
efficientnet_autoattb4st = SklearnCompatibleModel(EfficientNetAutoAttB4ST())

# Stacking 모델 구성
estimators = [
    ('xception_st', xception_st),
    ('efficientnet_autoattb4st', efficientnet_autoattb4st),
]

meta_model = LogisticRegression()
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)


# 1. 데이터 로드 및 분할
# df_faces = pd.read_pickle("data/celebdf_faces.pkl")
df_faces = pd.read_pickle("/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl")

X = df_faces[['height', 'width', 'frames', 'nfaces']]
y = df_faces['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 기본 학습자 (Base Learners)
xception_st = XceptionST()  # 이미 정의된 XceptionST 모델
efficientnet_autoattb4st = EfficientNetAutoAttB4ST()  # 이미 정의된 EfficientNetAutoAttB4ST 모델

# 기본 학습자 설정
estimators = [
    ('xception_st', xception_st),
    ('efficientnet_autoattb4st', efficientnet_autoattb4st),
]

# 메타 모델 정의 (Logistic Regression 사용)
meta_model = LogisticRegression()

# StackingClassifier 생성
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model)

# 데이터를 로드하고 학습 및 평가 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 앙상블 모델 학습
stacking_clf.fit(X_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = stacking_clf.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Stacking 모델의 정확도: {accuracy:.2f}')
