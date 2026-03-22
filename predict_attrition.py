import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import os

# 1. 데이터 로드
# 사용자의 환경에 맞는 파일 경로를 사용합니다.
file_path = os.path.join('data', '2_PAproject_2_4_machine.csv')
try:
    df = pd.read_csv(file_path)
    print("데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다. 경로를 다시 확인해주세요: {file_path}")
    exit(1)

# 2. 독립변수(X)와 종속변수(y) 설정
# 원인변수: 부서, 성과등급, 급여, 근무시간
# 결과변수: 퇴직 여부(Left)
X = df[['Department', 'Performance_Rating', 'Salary', 'Work_Hours']]
y = df['Left']

# 3. 전처리 파이프라인 설정
# - Department: 범주형 변수이므로 One-Hot Encoding 적용
# - 나머지: 수치형 변수이므로 SVM의 성능 향상을 위해 Standard Scaling 적용
categorical_features = ['Department']
numeric_features = ['Performance_Rating', 'Salary', 'Work_Hours']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. SVM 모델 구성 (확률 예측을 위해 probability=True 설정)
svm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# 5. 모델 학습
# 실제 프로젝트에서는 train_test_split으로 검증을 수행하는 것이 좋습니다.
svm_model.fit(X, y)
print("모델 학습이 완료되었습니다.")

# 6. 새로운 데이터 예측
# 새로운 파일 경로: data/2_PAproject_2_4_machine_prediction.csv
pred_file_path = os.path.join('data', '2_PAproject_2_4_machine_prediction.csv')
try:
    new_employee_df = pd.read_csv(pred_file_path)
    print(f"\n예측을 위한 데이터를 성공적으로 불러왔습니다. (총 {len(new_employee_df)}건)")
except FileNotFoundError:
    print(f"\n예측 파일을 찾을 수 없습니다: {pred_file_path}")
    exit(1)

prediction = svm_model.predict(new_employee_df)
pred_proba = svm_model.predict_proba(new_employee_df)

# 7. 결과 출력 및 저장
print("\n=== 예측 결과 (각 직원별 이직 예측) ===")
# 결과를 보기 쉽게 DataFrame에 추가
new_employee_df['Prediction'] = ['이직(Left)' if p == 1 else '잔류(Stay)' for p in prediction]
new_employee_df['Left_Probability(%)'] = np.round(pred_proba[:, 1] * 100, 2)

print(new_employee_df.to_string(index=False))

# 8. 결과를 CSV 파일로 저장
result_file_path = os.path.join('data', 'machine_results.csv')
new_employee_df.to_csv(result_file_path, index=False, encoding='utf-8-sig')
print(f"\n예측 결과가 '{result_file_path}' 파일로 성공적으로 저장되었습니다.")
