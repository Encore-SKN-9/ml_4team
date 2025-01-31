# [SKN09-ML-4Team]
✅ SKN AI FAMILY CAMP 9기<br>
✅ 개발 기간: 2025.01.25 - 2025.02.03 

---

# 📍Introduction Team(팀 소개)
### 👩‍👧‍👦팀명:<br> jobis, 취뽀

### 👩‍💻팀원
 
| 김우중👨‍💻 | 임수연👩‍💻 | 조민훈👨‍💻 |
|--|--|--|
|<a href="https://github.com/kwj9942">@kwj9942</a>|<a href="https://github.com/ohback">@ohback</a>|<a href="https://github.com/alche22">@alche22</a>|
<br>

# 📍프로젝트 개요
### 주제<br>
취업훈련을 받은 대졸자들의 취업 현황 예측

### 배경<br>
날로 높아지는 청년 실업에 취업에 도움되는 정보를 제공하기 위하여 직업훈련을 받은 대졸자들이 실제로 취업에 성공하는지, 고용보험 가입 여부를 통해 분석하고 머신러닝을 활용하여 취업 가능성을 예측함으로써, 효과적인 취업 지원 전략을 수립하기 위한 프로그램 제작

<img src="images/news.headline.jpg" width="300" height="150" />
<img src="images/news.article.jpg" width="300" height="150" />
<img src="images/news.main.jpg" width="300" height="150" />
<br>

# 📊데이터 시각화를 통한 탐색(EDA)
- Heatmap
<img src="images/heatmap.png" width="500" height="500" />

- Boxplot
<img src="images/box_plot.png" width="500" height="500" />

- 상관관계를 보이지 않는 b코드 제거
<img src="images/proBlem.png" width="500" height="500" />

- 유의미한 변수 그래프
<img src="images/.png" width="500" height="500" />
<br>

# 📍Data Pre-Processing(데이터 전처리)
- Data Set 개요<br>
고학력화로 인한 청년실업문제가 심화되면서 학교에서 노동시장으로의 학교 (전공)별 이행현황 분석과 원활한 이행을 지원하기 위한 다각적인 정책적 수요 증대를 위한 조사인 「대졸자직업이동경로조사(GOMS: Graduates Occupational Mobility Survey) 4년치(2016-19년도) 데이터를 사용하였음<br>
출처: [https://www.kli.re.kr/klips](https://www.kli.re.kr/klips)
<br>

- 전처리 과정
1) 숫자형 데이터는 결측값을 0, "모름" 을 최빈값으로 채움
```python
# 부모님의 자산규모 nan -> 0 and 모름 -> 평균값 3.01로 대체
use_data['p036'].replace(-1, 3.01, inplace=True)
use_data['p036'] = use_data['p036'].fillna(0)
```

2) 범주형 데이터는 설문조사지의 응답과 비교하여 유, 무로 간소화
```python
# 군 복무 경험 있으면 1, 없으면 0
use_data['p045'] = use_data['p045'].replace({val:0 for val in [-1,1,6,7]})
use_data['p045'] = use_data['p045'].apply(lambda x: 0 if x == 0 else 1)
use_data['p045'] = use_data['p045'].fillna(0)
```
<br>

# 📍Machine Learning (머신 러닝)
- Validation 
<img src="images/validation.png" width="500" height="500" />
<br>

# 📍실제 예측 결과
<br>

# 🎯프로젝트 기대 효과
- 청년 실업 문제 해결을 위한 데이터 기반 인사이트 제공
- 구직자 맞춤형 취업 전략 수립 지원
- 청년층의 취업률 향상 기여
<br>

---
# 📌한줄회고
김우중 :
임수연 : 
조민훈 : 
