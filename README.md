# scikit-learn 당뇨병 데이터 분석 — 멘토링 실습 프로젝트

scikit-learn 당뇨병(diabetes) 데이터셋을 현업 데이터 사이언티스트 관점에서 다루는 심화 분석 실습. 취업 준비생 대상 멘토링용으로 **연습용(practice)** 과 **답지용(solution)** 을 구분해 구성.

> 본 저장소의 모든 설명 문장은 존댓말이 아닌 **명사형 어미**로 작성됨.

## 프로젝트 구성

```
scikit_diabetes/
├── README.md              # 프로젝트 안내
├── requirements.txt       # 의존성 목록
├── .gitignore
├── notebooks/
│   ├── diabetes_advanced_practice.ipynb   # 연습용 (TODO 빈칸 포함)
│   └── diabetes_advanced_solution.ipynb   # 답지용 (완성본)
└── dashboard/
    ├── analysis.py        # 대시보드용 분석 로직 (캐시 가능한 순수 함수)
    └── app.py             # Streamlit 대시보드 UI
```

- `notebooks/` — 셀 단위로 결과를 확인하며 분석을 따라가는 실습용
- `dashboard/` — EDA·모델링 결과를 한눈에 시각화한 Streamlit 대시보드(결과 발표·공유용)

## 데이터 개요

- **샘플 수**: 442명 (적은 표본 → 증강 실험 대상)
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

## 분석 파이프라인 (고도화)

1. **데이터 품질 점검** — 결측·중복 점검, 요약 통계
2. **심화 EDA** — 왜도·첨도, 타깃 사분위별 바이올린, 상관 클러스터맵, VIF 다중공선성, 상호정보량(비선형), PCA, IsolationForest 이상치 탐지
3. **피처 엔지니어링** — 시각화 기반 파생변수(상호작용·비선형·비율 항) 생성 및 효과 검증
4. **모델링** — 7종 모델(ElasticNet·Ridge·Lasso·SVR·KNN·RandomForest·HistGBM)을 RepeatedKFold(5×3)로 비교, 원본 vs 파생 피처 검증
5. **데이터 증강(뻥튀기)** — 가우시안 노이즈·GMM 합성으로 학습 데이터 증강, **학습 폴드 한정·테스트 원본 유지**로 누수 없는 엄격 비교
6. **하이퍼파라미터 튜닝** — RandomizedSearchCV
7. **모델 해석** — 순열 중요도, 잔차 분석, 부분의존도(PDP), 학습곡선

## 멘토링 진행 방법

- 멘토는 `diabetes_advanced_solution.ipynb`로 전체 흐름과 결과를 먼저 시연
- 멘티는 `diabetes_advanced_practice.ipynb`의 `# TODO` 빈칸(`______`)을 직접 채우며 실습
- 각 TODO 옆 주석의 힌트를 참고해 코드 완성
- 완성 후 답지용과 비교하며 차이 확인

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 주피터로 실습 (셀 단위로 결과 확인)
jupyter lab notebooks/diabetes_advanced_practice.ipynb   # 연습용
jupyter lab notebooks/diabetes_advanced_solution.ipynb   # 답지용

# 3. Streamlit 대시보드 실행 (결과 시각화)
streamlit run dashboard/app.py
```

## 핵심 결론 (미리 보기)

- bmi와 s5가 선형·비선형·중요도 분석 전반에서 일관되게 핵심 인자로 확인됨
- 혈청 지표 간 강한 공선성 존재(VIF 높음) → 정규화 선형 모델·트리 계열이 안정적
- 파생변수는 모델에 따라 소폭의 성능 향상에 기여
- 데이터 증강은 누수 없는 비교에서 뚜렷한 개선을 주지 못함 → 표 형식 회귀에서 합성 증강의 한계 확인
- 학습곡선의 검증 성능이 평탄 → 표본 수보다 피처 정보량이 성능의 병목

> 본 실습은 교육용 데이터셋 기반이며 실제 임상 진단에는 사용 불가.
