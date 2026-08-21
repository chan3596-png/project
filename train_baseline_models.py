import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib

# 1. 데이터 로드
data_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train_clean.csv"
df = pd.read_csv(data_path)

# 2. 피처(Feature) 및 타겟(Target) 설정
# 사용하지 않을 메타데이터 및 다른 타겟 변수 제외
features = ['tool_id', 'rf_source_w', 'rf_bias_w', 'pressure_mtorr', 'gas_flow_sccm', 
            'esc_temp_c', 'chamber_age_runs', 'oes_peak_ratio', 'endpoint_slope', 'endpoint_time_s']

X = df[features].copy()

# 범주형 변수(tool_id) One-Hot Encoding 처리
X = pd.get_dummies(X, columns=['tool_id'], drop_first=False)

# Target 1: 회귀 (cd_bias_nm)
y_reg = df['cd_bias_nm']

# Target 2: 분류 (overetch_risk)
# 원래 실수형태인 위험도를 기반으로 임계값 설정 (예: 0.1을 초과하면 '고위험군(1)', 이하면 '정상(0)')
# 임계값 0.1은 EDA 당시 75% 백분위수가 0.097이므로 상위 약 20%를 불량군으로 정의하는 논리적 기준
threshold = 0.1
y_clf = (df['overetch_risk'] > threshold).astype(int)

# 3. 데이터 분리 (Train 70% / Test 30%)
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.3, random_state=42
)

print("=== 데이터 분리 완료 ===")
print(f"Train 데이터 크기: {X_train.shape[0]}건 (70%)")
print(f"Test 데이터 크기: {X_test.shape[0]}건 (30%)")

# 4. 회귀 모델 학습 (CD Bias 예측)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("\n=== 1. CD Bias 예측 (회귀 모델: Random Forest) ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# 5. 분류 모델 학습 (Overetch Risk 분류)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_model.fit(X_train, y_clf_train)
y_clf_pred = clf_model.predict(X_test)

accuracy = accuracy_score(y_clf_test, y_clf_pred)
print("\n=== 2. Overetch Risk 예측 (분류 모델: Random Forest) ===")
print(f"분류 기준: overetch_risk > {threshold} (High Risk=1)")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print("\n[상세 분류 리포트]")
print(classification_report(y_clf_test, y_clf_pred, target_names=['Normal (0)', 'High Risk (1)']))

# 6. (선택) 모델 파일 저장
model_dir = "c:/Users/chan/Documents/semiconductor-ai-project/models"
import os
os.makedirs(model_dir, exist_ok=True)
joblib.dump(reg_model, f"{model_dir}/rf_cd_bias_model.pkl")
joblib.dump(clf_model, f"{model_dir}/rf_overetch_clf_model.pkl")
# 컬럼 정보 저장 (추론 시 활용)
joblib.dump(list(X.columns), f"{model_dir}/model_features.pkl")
print(f"\n학습된 기준 모델이 '{model_dir}' 폴더에 저장되었습니다.")
