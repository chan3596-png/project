# Semiconductor AI - Dry Etch Process Analysis

본 프로젝트는 반도체 Dry Etch(건식 식각) 공정 데이터를 활용하여 품질 지표(CD Bias, Selectivity)와 불량(Over-etch Risk)을 분석하고 AI 기준 모델을 구축하기 위한 프로젝트입니다.

## 📌 진행 내역 및 데이터 전처리 요약 (재현 가능성 확보)

### 1. 원본 데이터 분석 (`dry_etch_A_train.csv`)
* **파라미터 분류:** 제어 파라미터(RF Power, Pressure 등), 센서/상태 파라미터(OES, EPD, PM 주기 등), 타겟 지표(CD Bias, Selectivity 등)로 분류.
* **초기 문제점 탐색:** 중복 데이터(5건), 온도 센서 단위 오류(K 표기), 스케일 오류(유량/압력 10배), 식각 선택비 음수 등 물리적 오류 식별.

## 🚀 실행 방법 (Reproducibility)

본 프로젝트는 아래의 순서대로 스크립트를 실행하여 결과를 재현할 수 있습니다.

1. **데이터 전처리 (Data Cleaning)**
   ```bash
   python preprocess_dry_etch.py
   ```
   *결과:* `3.Dry Etch/dry_etch_A_train_clean.csv` 생성

2. **EDA 및 고급 리포트 생성 (Reporting)**
   ```bash
   python generate_adv_html.py
   ```
   *결과:* 인터랙티브 그래프, 주석(Tooltip), 다중공선성(VIF) 검증 결과가 포함된 `3.Dry Etch/index.html` 리포트 생성

3. **AI 기준 모델 학습 (Model Training)**
   ```bash
   python train_baseline_models.py
   ```
   *결과:* `models/` 디렉토리에 Random Forest 회귀(CD 예측) 및 분류(위험도) 모델(`.pkl`) 저장

4. **공정 시뮬레이터 실행 (Simulation & Recommendation)**
   ```bash
   python simulator.py
   ```
   *결과:* 장비 노후화, 파라미터 결함 등 다양한 입력 시나리오에 따른 예측 결과 및 전문가 시스템 기반 자동 추천(Action Plan) 터미널 출력

## ⚠️ 모델 한계점 (Limitations)
* **CD 예측 한계:** 현재 데이터(장비 센서 데이터)만으로는 R² 성능이 약 38% 수준에 머뭅니다. 이는 식각 공정의 선폭이 이전 공정(노광, 증착)의 영향을 크게 받기 때문이며, 양산 투입을 위해서는 이전 공정의 계측(Metrology) 데이터 병합이 필수적입니다.
* **과식각 예측 한계:** 10Hz 수준의 FDC 원형(Trace) 데이터가 아닌 Summary 데이터로 학습되어 미세한 밸브 헌팅이나 찰나의 아킹 현상을 잡아내는 데는 한계가 있습니다.

### 2. 데이터 전처리 파이프라인 (`preprocess_dry_etch.py`)
`AGENTS.md` 규칙 2(원본 수정 금지)에 따라, 파이썬 스크립트를 통해 다음 보정 작업을 수행하고 `dry_etch_A_train_clean.csv`를 별도 생성했습니다.
1. **중복 행 제거:** 메타데이터 및 모든 값이 동일한 5행 삭제.
2. **단위 보정:** 100도가 넘는 온도는 Kelvin으로 간주하여 -273.15 적용. 가스 유량 및 압력 극단값(100 이상)은 1/10 스케일링.
3. **물리적 오류 삭제:** 식각 선택비(Selectivity)가 음수(-1.14)인 잘못된 계측 데이터 1행 완전 삭제.
4. **결측치 및 센서 오류 대체:** OES 비율, 온도 등 극단적 통신 오류 값을 NaN 처리한 후, **공정 설비(`tool_id`)별 중앙값(Median)** 으로 일괄 결측치 대체. (최종 799행, 결측치 0 확보)

### 3. 탐색적 데이터 분석 (EDA) 및 공선성 검증
* **상관관계 및 가설 검증:** 
  - RF Bias와 식각 선택비 간의 강한 물리적 역상관관계(-0.92) 확인.
  - 챔버 노후화(`chamber_age_runs`)에 따른 CD Bias Drift가 모든 Tool에서 공통적으로 발생하는 진짜 인과관계임을 증명.
  - 설비 `T04`의 CD Bias 및 선택비가 타 설비 대비 하향 편향(Bias)되어 있음을 확인.
* **VIF 다중공선성 검증 (`scratch_vif.py`):** 모든 독립 변수의 VIF 수치가 10 미만(최대 1.63)으로 도출되어 선형/트리 모델 입력으로 매우 안정적인 상태임을 확인.

### 4. 고급 분석 리포트 생성 (`index.html`)
* Python(Matplotlib/Seaborn)을 통해 다양한 통계 그래프(KDE 밀도 함수, Hexbin, 다변량 회귀 등)를 생성했습니다.
* 한글 폰트(`Malgun Gothic`) 깨짐을 방지하고, 그래프 이미지들을 Base64로 인코딩하여 HTML 단일 파일에 임베딩했습니다.
* 플라즈마 물리 수식(MathJax)과 모식도(Mermaid)를 포함하여 도메인 지식 기반 해석 리포트를 `3.Dry Etch/` 폴더 내에 완성했습니다.
