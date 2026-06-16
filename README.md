# scikit-learn 당뇨병 데이터 분석 — 멘토링 실습 프로젝트

scikit-learn 내장 당뇨병(diabetes) 데이터셋으로 수행하는 탐색적 분석(EDA)과 회귀 모델링 실습. 취업 준비생 대상 멘토링용으로, **연습용(practice)** 과 **답지용(solution)** 을 구분해 구성.

> 본 저장소의 모든 설명 문장은 존댓말이 아닌 **명사형 어미**로 작성됨.

## 프로젝트 구성

```
scikit_diabetes/
├── README.md              # 프로젝트 안내
├── requirements.txt       # 의존성 목록
├── .gitignore
├── notebooks/             # EDA+모델링 실습 노트북 (눈으로 보며 진행)
│   ├── eda_modeling_practice.ipynb   # 연습용 (TODO 빈칸 포함)
│   └── eda_modeling_solution.ipynb   # 답지용 (완성본)
├── practice/
│   └── app_practice.py    # 연습용 Streamlit 앱 (TODO 빈칸 포함)
└── solution/
    └── app_solution.py    # 답지용 Streamlit 앱 (완성본)
```

> **노트북 vs Streamlit 앱**
> - `notebooks/` — 셀 단위로 결과를 눈으로 확인하며 EDA·모델링을 따라가는 실습용
> - `practice/`, `solution/` — 완성된 분석을 인터랙티브 리포트로 보여주는 Streamlit 앱

## 데이터 개요

- **샘플 수**: 442명
- **피처 수**: 10개 (나이, 성별, 체질량지수, 혈압, 혈청 지표 6종)
- **타깃**: 기준 시점 1년 후 당뇨병 진행도 (연속형, 25~346)
- 모든 피처는 평균 0으로 표준화되어 제공됨

| 변수 | 의미 |
|------|------|
| age | 나이 |
| sex | 성별 |
| bmi | 체질량지수 |
| bp | 평균 혈압 |
| s1 | 총 콜레스테롤(TC) |
| s2 | LDL |
| s3 | HDL |
| s4 | TC/HDL 비율 |
| s5 | 혈청 중성지방 로그값(ltg) |
| s6 | 혈당(glu) |

## 실습 흐름

1. **데이터 로드 및 개요 확인** — 형태, 요약 통계, 변수 의미 파악
2. **탐색적 분석(EDA)** — 타깃 분포, 상관관계 히트맵, 상위 변수 산점도
3. **회귀 모델링** — 5종 모델 학습 및 5-fold 교차검증 비교
4. **변수 중요도 분석** — Ridge 계수와 Random Forest 중요도 비교
5. **예측 결과 해석** — 예측값 대 실제값 비교

## 멘토링 진행 방법

- 멘토는 `solution/app_solution.py`로 전체 흐름과 결과를 먼저 시연
- 멘티는 `practice/app_practice.py`의 `# TODO` 부분을 직접 채우며 실습
- 각 TODO 옆 주석의 힌트를 참고해 코드 완성
- 완성 후 답지용과 비교하며 차이 확인

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 주피터 노트북으로 EDA+모델링 실습 (눈으로 보며 진행)
jupyter lab notebooks/eda_modeling_practice.ipynb   # 연습용
jupyter lab notebooks/eda_modeling_solution.ipynb   # 답지용

# 3. Streamlit 리포트 앱 실행
streamlit run solution/app_solution.py   # 답지용(시연)
streamlit run practice/app_practice.py   # 연습용(실습)
```

## 핵심 결론 (미리 보기)

- 체질량지수(bmi)와 혈청 중성지방(s5)이 진행도와 가장 강한 양의 상관을 보임
- HDL 콜레스테롤(s3)은 진행도와 음의 상관 — 보호 인자로 해석됨
- 정규화 선형 모델(Lasso/Ridge)이 트리 앙상블보다 우수한 성능 — 데이터가 선형적이기 때문
- 최고 모델도 분산의 약 47%만 설명 — 10개 지표만으로는 한계 존재

> 본 실습은 교육용 데이터셋 기반이며 실제 임상 진단에는 사용 불가.
