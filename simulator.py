import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_models():
    model_dir = "c:/Users/chan/Documents/semiconductor-ai-project/models"
    reg_model = joblib.load(f"{model_dir}/rf_cd_bias_model.pkl")
    clf_model = joblib.load(f"{model_dir}/rf_overetch_clf_model.pkl")
    features = joblib.load(f"{model_dir}/model_features.pkl")
    return reg_model, clf_model, features

def simulate_etch_process(scenario_name, input_data):
    reg_model, clf_model, feature_columns = load_models()
    
    # 입력 데이터를 DataFrame으로 변환
    df_input = pd.DataFrame([input_data])
    
    # 빈 딕셔너리로 초기화
    row_data = {col: 0.0 for col in feature_columns}
    
    # 매핑
    for col, val in input_data.items():
        if col == 'tool_id':
            tool_col = f"tool_id_{val}"
            if tool_col in row_data:
                row_data[tool_col] = 1.0
        elif col in row_data:
            row_data[col] = float(val)
            
    # DataFrame으로 변환
    df_encoded = pd.DataFrame([row_data])
    
    # 예측 수행
    cd_bias_pred = reg_model.predict(df_encoded)[0]
    risk_prob = clf_model.predict_proba(df_encoded)[0][1]
    is_high_risk = risk_prob > 0.5  # 분류 모델의 최종 판단
    
    print(f"\n==================================================")
    print(f" [시나리오] {scenario_name}")
    print(f"==================================================")
    print(f"▶ 입력 파라미터 (주요 변수):")
    print(f"  - Tool ID: {input_data['tool_id']}")
    print(f"  - RF Bias Power: {input_data['rf_bias_w']} W")
    print(f"  - ESC Temp: {input_data['esc_temp_c']} °C")
    print(f"  - Chamber Age: {input_data['chamber_age_runs']} runs")
    print(f"\n▶ AI 예측 결과:")
    print(f"  - 예상 CD Bias : {cd_bias_pred:+.3f} nm")
    
    if is_high_risk:
        print(f"  - 과식각 위험도: [High Risk] (확률: {risk_prob*100:.1f}%)")
    else:
        print(f"  - 과식각 위험도: [Normal] (확률: {risk_prob*100:.1f}%)")
        
    print(f"\n▶ AI 자동 추천 (Action Plan):")
    
    # 추천 로직 (규칙 기반 전문가 시스템 결합)
    recommendations = []
    
    if input_data['esc_temp_c'] > 60:
        recommendations.append("온도 이상 감지: 정전척(ESC)의 헬륨(He) 냉각 라인 누설 여부 즉시 점검 요망.")
        
    if is_high_risk and input_data['rf_bias_w'] > 200:
        recommendations.append("과식각 위험: RF Bias Power가 너무 높습니다. 10% 하향 조정(Step-down)하여 포토레지스트(PR) 마스크 손상을 방지하세요.")
        
    if input_data['chamber_age_runs'] > 150:
        recommendations.append("챔버 노후화: 누적 Run 수가 많아 폴리머(Polymer) 증착에 의한 CD 양수(+) 팽창이 우려됩니다. WAC(Wafer Auto Clean) 주기 단축 또는 Wet PM 일정을 검토하세요.")
        
    if cd_bias_pred < -3.0:
        recommendations.append("CD 대폭 감소 우려: 식각률(Etch Rate)이 비정상적으로 높습니다. 가스 유량(Gas Flow) 및 OES 종말점(EPD) 타겟 시간을 재설정하세요.")

    if not recommendations:
        recommendations.append("현재 파라미터는 안정적인 공정 윈도우(Process Window) 내에 있습니다. 그대로 양산을 진행하세요.")
        
    for idx, rec in enumerate(recommendations, 1):
        print(f"  {idx}. {rec}")
    print(f"==================================================\n")

if __name__ == "__main__":
    # 시나리오 1: 완벽하게 정상적인 기준선 파라미터
    scenario_normal = {
        'tool_id': 'T01', 'rf_source_w': 800, 'rf_bias_w': 100, 
        'pressure_mtorr': 15.0, 'gas_flow_sccm': 100.0, 'esc_temp_c': 25.0, 
        'chamber_age_runs': 50, 'oes_peak_ratio': 1.2, 'endpoint_slope': 0.05, 'endpoint_time_s': 45.0
    }
    
    # 시나리오 2: ESC 냉각 불량 및 RF Bias 과도 (불량 발생 상황 모사)
    scenario_fault = {
        'tool_id': 'T01', 'rf_source_w': 800, 'rf_bias_w': 250, 
        'pressure_mtorr': 15.0, 'gas_flow_sccm': 100.0, 'esc_temp_c': 75.0, 
        'chamber_age_runs': 50, 'oes_peak_ratio': 1.2, 'endpoint_slope': 0.05, 'endpoint_time_s': 85.0
    }

    # 시나리오 3: 챔버 노후화 한계 도달 (CD Drift 상황 모사)
    scenario_aged = {
        'tool_id': 'T04', 'rf_source_w': 800, 'rf_bias_w': 100, 
        'pressure_mtorr': 15.0, 'gas_flow_sccm': 100.0, 'esc_temp_c': 25.0, 
        'chamber_age_runs': 190, 'oes_peak_ratio': 1.0, 'endpoint_slope': 0.03, 'endpoint_time_s': 45.0
    }

    simulate_etch_process("안정적인 양산 가동 (Normal Process)", scenario_normal)
    simulate_etch_process("ESC 냉각 결함 및 과식각 폭주 (Fault Injection)", scenario_fault)
    simulate_etch_process("챔버 PM 주기 임박 (Aged Chamber Drift)", scenario_aged)
