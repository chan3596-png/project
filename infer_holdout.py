import pandas as pd
import numpy as np
import joblib

import sys

# 1. 파일 경로 설정
train_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train_clean.csv"
model_dir = "c:/Users/chan/Documents/semiconductor-ai-project/models"

if len(sys.argv) > 1:
    holdout_path = sys.argv[1]
    output_path = sys.argv[2]
else:
    holdout_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_holdout_features.csv"
    output_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_holdout_predictions.csv"

# 2. 데이터 로드
print("Loading data...")
df_train = pd.read_csv(train_path)
df_holdout = pd.read_csv(holdout_path)
holdout_samples = df_holdout[['sample_id', 'lot_id', 'tool_id']].copy()

# 3. Holdout 전처리 (Train 세트와 동일한 규칙 적용)
# 3.1 이상치 보정 (단위 오기입 등)
if 'esc_temp_c' in df_holdout.columns:
    df_holdout.loc[df_holdout['esc_temp_c'] > 200, 'esc_temp_c'] -= 273.15
if 'gas_flow_sccm' in df_holdout.columns:
    df_holdout.loc[df_holdout['gas_flow_sccm'] > 500, 'gas_flow_sccm'] /= 10.0
if 'pressure_mtorr' in df_holdout.columns:
    df_holdout.loc[df_holdout['pressure_mtorr'] > 100, 'pressure_mtorr'] /= 10.0

# 3.2 결측치(NaN) 처리 - Train Data의 Median 사용 (Data Leakage 방지)
numeric_cols = df_holdout.select_dtypes(include=[np.number]).columns
train_medians = df_train[numeric_cols].median()
df_holdout[numeric_cols] = df_holdout[numeric_cols].fillna(train_medians)

# 4. Feature 엔지니어링 및 정렬
print("Applying preprocessing and loading models...")
# 카테고리 변수 원핫 인코딩
df_holdout_encoded = pd.get_dummies(df_holdout, columns=['tool_id'])

# 모델 학습 시 사용했던 컬럼명 로드
model_features = joblib.load(f"{model_dir}/model_features.pkl")

# 학습 데이터의 피처 순서와 완벽히 맞추기 (없는 컬럼은 0으로)
X_holdout = pd.DataFrame(columns=model_features)
for col in model_features:
    if col in df_holdout_encoded.columns:
        X_holdout[col] = df_holdout_encoded[col]
    else:
        X_holdout[col] = 0

# 5. 추론 (Inference)
reg_model = joblib.load(f"{model_dir}/rf_cd_bias_model.pkl")
clf_model = joblib.load(f"{model_dir}/rf_overetch_clf_model.pkl")

print("Running predictions...")
# CD Bias 예측 (Regression)
cd_bias_pred = reg_model.predict(X_holdout)

# Overetch Risk 예측 (Classification & Probability)
risk_class = clf_model.predict(X_holdout)
risk_prob = clf_model.predict_proba(X_holdout)[:, 1]

# 6. 결과 저장
holdout_samples['predicted_cd_bias_nm'] = cd_bias_pred
holdout_samples['predicted_overetch_risk_class'] = risk_class
holdout_samples['predicted_overetch_risk_prob'] = risk_prob

holdout_samples.to_csv(output_path, index=False)

print(f"\n==================================================")
print(f" [검증 완료] Holdout 데이터 예측 결과 요약")
print(f"==================================================")
print(f"- 대상 데이터 수: {len(holdout_samples)} 건")
print(f"- 예상 CD Bias 평균: {cd_bias_pred.mean():.3f} nm")
print(f"- 고위험군(High Risk) 탐지 건수: {sum(risk_class)} 건 ({(sum(risk_class)/len(risk_class))*100:.1f}%)")
print(f"- 결과 저장 위치: {output_path}")
print(f"==================================================")
