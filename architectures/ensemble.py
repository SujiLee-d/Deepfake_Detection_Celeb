# Ensemble - Stacking method trial 1
# Video Face Manipulation Detection Through Ensemble of CNNs


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from architectures import fornet,weights
from architectures.fornet import FeatureExtractor

# 1. 데이터 로드 및 분할
#data = weights.load_breast_cancer()
#X, y = data.data, data.target   

# 2. 기본 학습자 (Base Learners)

# XceptionST와 EfficientNetAutoAttB4ST을 Base Learner로 설정
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
