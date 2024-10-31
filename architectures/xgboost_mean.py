
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 및 통합
base_dir = '/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/CHECKPOINT_DST'
sub_dirs = ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']  
all_data = []

for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # 각 pkl 파일을 읽어와 DataFrame 형태로 추가
                df = pd.read_pickle(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"파일 {file_path}을(를) 읽는 중 오류가 발생했습니다: {e}")

# 모든 데이터를 하나의 DataFrame으로 합침
df = pd.concat(all_data, ignore_index=True)

X_preprocessed = df.drop(columns=['class', 'original', 'videosubject'])  # 필요 없는 문자열 열 제외

# 2. 'video'별로 그룹핑하여 각 비디오별로 하나의 특성 벡터로 변환
# 평균을 사용하는 경우 (각 video의 여러 row를 평균으로 결합)
grouped_df = X_preprocessed.groupby('video').mean()  # 또는 .sum(), .median() 등 다른 집계 함수를 사용할 수 있음

# 그룹화 후, 'label' 열은 대부분 동일할 것이므로 첫 번째 값을 사용하여 유지
labels = X_preprocessed.groupby('video')['label'].first()

# 3. 특성과 레이블 정의
X = grouped_df.drop(columns=['label'])  # 각 비디오별로 평균으로 결합된 특성 벡터
y = labels  # 레이블 (첫 번째 값 사용)
# print number of columns in X
# print(X.info)
# print(y.head(150))

# 4. 학습 및 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# 5. XGBoost 모델 초기화
xgb_model = XGBClassifier(eval_metric='logloss')

# 6. 모델 학습
xgb_model.fit(X_train, y_train) 

# 7. 테스트 데이터로 예측 수행
y_pred = xgb_model.predict(X_test)

# 8. 모델 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of XGBoost_mean model: {accuracy:.2f}')

# Confusion Matrix 계산
conf_matrix = confusion_matrix(y_test, y_pred)

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for XGBoost Model')
plt.show()
