import pandas as pd
import numpy as np
import os

# 1. 데이터 로드
input_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train.csv"
output_path = "c:/Users/chan/Documents/semiconductor-ai-project/3.Dry Etch/dry_etch_A_train_clean.csv"

df = pd.read_csv(input_path)
initial_len = len(df)

# 2. 중복 데이터 제거
df.drop_duplicates(inplace=True)
dedup_len = len(df)

# 3. 단위 오류 보정 (Kelvin, 10x Scale)
df.loc[df['esc_temp_c'] > 100, 'esc_temp_c'] -= 273.15
df.loc[df['gas_flow_sccm'] > 500, 'gas_flow_sccm'] *= 0.1
df.loc[df['pressure_mtorr'] > 100, 'pressure_mtorr'] *= 0.1

# 4. 물리적 오류 삭제 (selectivity < 0)
drop_cond = df['selectivity'] < 0
df = df[~drop_cond]
dropped_rows = drop_cond.sum()

# 5. 센서 이상치 결측치(NaN) 처리
esc_temp_glitch = df['esc_temp_c'] > 50
oes_ratio_glitch = (df['oes_peak_ratio'] > 3.0) | (df['oes_peak_ratio'] < 0.2)

df.loc[esc_temp_glitch, 'esc_temp_c'] = np.nan
df.loc[oes_ratio_glitch, 'oes_peak_ratio'] = np.nan

# 6. 결측치 Tool별 중앙값(Median) 대체
missing_cols = df.columns[df.isnull().any()].tolist()
for col in missing_cols:
    if df[col].dtype in [np.float64, np.int64]:
        # Tool ID별 중앙값으로 채우기
        df[col] = df.groupby('tool_id')[col].transform(lambda x: x.fillna(x.median()))
        # 혹시라도 특정 Tool에 해당 컬럼 전체가 NaN인 경우를 대비한 2차 전체 중앙값 대체
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# 7. 클린 데이터 저장
df.to_csv(output_path, index=False)

# 8. 결과 출력
print("=== 전처리 완료 보고서 ===")
print(f"1. 원본 데이터 수: {initial_len} rows")
print(f"2. 중복 데이터 제거: {initial_len - dedup_len} rows 삭제 (남은 수: {dedup_len})")
print(f"3. 물리적 오류(Selectivity < 0) 제거: {dropped_rows} rows 삭제 (최종 행 수: {len(df)})")
print(f"4. 센서 오류 결측치 처리 (ESC 온도: {esc_temp_glitch.sum()}건, OES 비율: {oes_ratio_glitch.sum()}건)")
print(f"5. 남은 결측치 수: {df.isnull().sum().sum()}건 (Tool별 중앙값으로 모두 대체됨)")
print(f"\n최종 클린 데이터가 저장되었습니다: {output_path}")
