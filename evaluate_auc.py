import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import joblib

# 1. 데이터 로드 및 분리 (이전과 동일한 random_state=42 적용)
data_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train_clean.csv"
df = pd.read_csv(data_path)

features = ['tool_id', 'rf_source_w', 'rf_bias_w', 'pressure_mtorr', 'gas_flow_sccm', 
            'esc_temp_c', 'chamber_age_runs', 'oes_peak_ratio', 'endpoint_slope', 'endpoint_time_s']
X = df[features].copy()
X = pd.get_dummies(X, columns=['tool_id'], drop_first=False)

threshold = 0.1
y_clf = (df['overetch_risk'] > threshold).astype(int)

X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.3, random_state=42
)

# 2. 기 저장된 분류 모델 로드
model_path = "c:/Users/chan/Documents/semiconductor-ai-project/models/rf_overetch_clf_model.pkl"
clf_model = joblib.load(model_path)

# 3. AUC 계산을 위한 확률(Probability) 예측
# 클래스 1(High Risk)에 속할 확률 추출
y_pred_proba = clf_model.predict_proba(X_test)[:, 1]

# 4. ROC AUC Score 계산
auc_score = roc_auc_score(y_clf_test, y_pred_proba)

print("=== Overetch Risk 분류 모델 (Random Forest) AUC 평가 ===")
print(f"ROC-AUC Score: {auc_score:.4f}")

# 변별력에 대한 간단한 해석 출력
if auc_score > 0.9:
    interpretation = "매우 뛰어남 (Excellent) - 양품과 불량을 거의 완벽하게 구분합니다."
elif auc_score > 0.8:
    interpretation = "우수함 (Good) - 실제 불량 웨이퍼를 매우 높은 확률로 위험군으로 분류해 냅니다."
elif auc_score > 0.7:
    interpretation = "보통 (Fair) - 어느 정도 변별력이 있으나 오탐(False Alarm)이 발생할 수 있습니다."
else:
    interpretation = "미흡 (Poor) - 랜덤 찍기(0.5)와 큰 차이가 없어 Feature Engineering 등 추가 개선이 필요합니다."

print(f"변별력 평가: {interpretation}")
