import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# 데이터 로드 및 전처리
data_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train_clean.csv"
df = pd.read_csv(data_path)

features = ['tool_id', 'rf_source_w', 'rf_bias_w', 'pressure_mtorr', 'gas_flow_sccm', 
            'esc_temp_c', 'chamber_age_runs', 'oes_peak_ratio', 'endpoint_slope', 'endpoint_time_s']
X = pd.get_dummies(df[features].copy(), columns=['tool_id'], drop_first=False)
y = df['cd_bias_nm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

results = {}

# 1. Random Forest (튜닝)
rf_tuned = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42)
rf_tuned.fit(X_train, y_train)
y_pred_rf = rf_tuned.predict(X_test)
results['Random Forest (Tuned)'] = {'R2': r2_score(y_test, y_pred_rf), 'MSE': mean_squared_error(y_test, y_pred_rf)}

# 2. XGBoost
if xgb is not None:
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = {'R2': r2_score(y_test, y_pred_xgb), 'MSE': mean_squared_error(y_test, y_pred_xgb)}
else:
    print("XGBoost not installed.")

# 3. LightGBM
if lgb is not None:
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    results['LightGBM'] = {'R2': r2_score(y_test, y_pred_lgb), 'MSE': mean_squared_error(y_test, y_pred_lgb)}
else:
    print("LightGBM not installed.")

# 4. Feature Engineering 적용 (RF Bias와 Endpoint Time 교차 변수 등 파생 변수 추가)
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()
# 이온 에너지 관련 파생 변수 (RF Bias * OES Ratio)
X_train_fe['rf_oes_interaction'] = X_train_fe['rf_bias_w'] * X_train_fe['oes_peak_ratio']
X_test_fe['rf_oes_interaction'] = X_test_fe['rf_bias_w'] * X_test_fe['oes_peak_ratio']

if xgb is not None:
    xgb_fe = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, objective='reg:squarederror')
    xgb_fe.fit(X_train_fe, y_train)
    y_pred_xgb_fe = xgb_fe.predict(X_test_fe)
    results['XGBoost (w/ Feature Engineering)'] = {'R2': r2_score(y_test, y_pred_xgb_fe), 'MSE': mean_squared_error(y_test, y_pred_xgb_fe)}

print("\n=== 향상된 모델 성능 비교 ===")
print(f"{'Model':<35} | {'R2 Score':<10} | {'MSE':<10}")
print("-" * 60)
for model_name, metrics in results.items():
    print(f"{model_name:<35} | {metrics['R2']:.4f}     | {metrics['MSE']:.4f}")
