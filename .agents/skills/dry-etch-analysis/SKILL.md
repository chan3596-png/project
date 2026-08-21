---
name: dry-etch-analysis
description: "반도체 Dry Etch 공정 데이터의 전처리, EDA, 물리적 수식 및 모식도(Mermaid)가 포함된 Base64 기반 HTML 고급 리포트 자동 생성 워크플로우"
---

# Dry Etch 공정 분석 특화 스킬 (dry-etch-analysis)

이 스킬은 반도체 Dry Etch(건식 식각) 데이터를 정제하고, 통계 분석(EDA 및 VIF)을 수행하며, 관련 논문 근거 및 물리적 수식이 결합된 단일 파일 형태의 고급 HTML 리포트(`index.html`)를 자동 생성하는 표준화된 작업 절차입니다.

## 🚀 적용 대상
* 반도체 식각 공정 데이터를 처음 전달받고 전처리와 EDA를 수행할 때
* 사용자가 "물리적 해석", "한글 폰트 적용", "단일 HTML (Base64)", "논문 레퍼런스 첨부"를 포함한 리포팅을 요구할 때

## 📝 필수 작업 단계 (Workflow)

### Step 1: 단위 및 물리적 오류 정제 (Preprocessing)
반드시 다음 도메인 지식(Domain Knowledge)을 기반으로 스크립트를 작성하여 데이터를 보정합니다. (원본 훼손 금지 원칙 준수)
1. **온도 센서 오류:** ESC 온도가 100도를 초과할 경우 Kelvin 스케일 오류로 간주하고 `- 273.15`를 적용합니다.
2. **유량/압력 오류:** 가스 유량이나 챔버 압력이 비정상적으로 높을 경우(예: 10배), 소수점 누락으로 간주하여 `* 0.1` 스케일링을 수행합니다.
3. **물리적 절대 규칙:** 식각 선택비(Selectivity)는 두 물질 간 식각 속도의 비율이므로 절대 음수(-)가 될 수 없습니다. 음수 값은 Drop 처리합니다.
4. **Tool별 결측치 대체:** 공정은 챔버(Tool) 고유의 레시피에 영향을 받으므로, 결측치(NaN)는 무조건 전체 평균이 아닌 **`tool_id`별 중앙값(Median)** 으로 대체합니다.

### Step 2: 다중공선성(VIF) 검증
* 모델링 안정성을 위해 수치형 변수(독립 변수) 간의 VIF를 계산하는 코드를 작성합니다.
* VIF > 10 인 컬럼이 발견될 경우, 사용자와 상의하여 차원 축소(PCA) 또는 변수 제거를 수행합니다.

### Step 3: 고급 시각화 (Seaborn + 한글 폰트 셋업)
파이썬 시각화 코드 작성 시 반드시 아래 코드를 상단에 삽입하여 한글 깨짐을 방지합니다.
```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic') # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font="Malgun Gothic", rc={"axes.unicode_minus": False})
```
* **필수 생성 그래프:** 상관 매트릭스(Triangular), 설비별 KDE 밀도, 다변량 산점도(Hexbin), Drift 트렌드 선형 회귀(lmplot).

### Step 4: HTML Base64 임베딩 및 렌더링 규격
단일 파일 구동 및 보안(AGENTS.md 규칙 9)을 위해 아래 사항을 엄수하는 HTML 렌더링 파이썬 스크립트를 작성합니다.
1. **Base64 인코딩:** 생성된 모든 플롯 이미지를 Base64 문자열로 인코딩하여 `<img>` 태그에 직접 삽입합니다.
2. **보안 모듈:** `polyfill.io` 절대 금지. 스타일은 Tailwind CDN (`https://cdn.tailwindcss.com`)을 사용합니다.
3. **물리적 수식 (MathJax):** `<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>` 추가 후, `$$ S = \frac{ER_{target}}{ER_{PR}} $$` 등의 수식으로 이온 에너지와 선택비의 관계를 기술합니다.
4. **모식도 (Mermaid):** Plasma -> Sheath -> Wafer 로 이어지는 이온 충돌 메커니즘을 Mermaid.js 로 HTML 내부에 삽입합니다.
5. **학술적 근거(References):** 결과물 하단에 플라즈마 공정(Lieberman 등)과 관련된 실제 논문 인용(References) 항목을 작성하여 분석의 타당성을 보강합니다.
6. **저장 위치:** 리포트는 반드시 분석 대상 CSV 파일과 동일한 경로 디렉토리에 `index.html`로 저장합니다.
