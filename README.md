# 모여봐요_나뭇잎숲

# 주제: DACON_주차수요 예측 AI경진대회

parking_demand_competion
대회 이름 : 주차수요 예측 AI 경진대회
웹 사이트 : https://dacon.io/competitions/official/235745/data
기간 : 2021.06.10 ~ 2021.07.30 18:00
데이터 셋
train.csv
test.csv
sample_submission.csv
age_gender_info.csv
배경- 대회 참조
아파트 단지 내 필요한 주차대수는 ①법정주차대수 ②장래주차수요 중 큰 값에 따라 결정하게되어 있어, 정확한 ②장래주차수요의 산정을 필요로 합니다. 현재 ②장래주차수요는 ‘주차원단위’와 ‘건축연면적’을 기초로하여 산출되고 있으며, ‘주차원단위’는 신규 건축예정 부지 인근의 유사 단지를 피크 시간대 방문하여 주차된 차량대수를 세는 방법으로 조사하고 있습니다. 이 경우 인력조사로 인한 오차발생, 현장조사 시점과 실제 건축시점과의 시간차 등의 문제로 과대 또는 과소 산정의 가능성을 배제할 수 없습니다.

[주제]
유형별 임대주택 설계 시 단지 내 적정 주차 수요를 예측

## 🔖 목차

[🌻 팀원소개](https://www.notion.so/caac7b127e5547d39234c067ceca5d21)

[ 📑 사용 Module ](https://www.notion.so/54557a7597f644c6ba8a636ed915e35a)

## 🧐 베이스 라인 데이터 셋 & 릿지 모델 선정 이유

### 1. 베이스 라인 데이터 셋 구축 과정에서 발견한 포인트

- 트레인 셋의 행의 개수 2952개 & 단지코드 기준으로 분류하면 423개 단지 → 개별 행의 독립성



- 각 행의 데이터 개체의 정보를 최대한 활용하는 방향으로 데이터 셋 구축

    ex) 전용면적, 전용면적 별 세대수 특성 VS 나머지 공통된 특성들

- 파생변수 생성 (세대수 비율 주차면수, 세대수 규모 당 총 전용면적, 전용면적별세대수)

```python
# 세대수 별 주차면수 피처 생성
train_df["세대수비율주차면수"] = (train_df["전용면적별세대수"] / train_df["실거주세대수"]) * train_df["단지내주차면수"]
test_df["세대수비율주차면수"] = (test_df["전용면적별세대수"] / test_df["실거주세대수"]) * test_df["단지내주차면수"]

# 세대수 규모 당 총 전용 면적 피처 생성
train_df["세대수 규모 당 총 전용면적"] = train_df["전용면적"] * train_df["전용면적별세대수"]
test_df["세대수 규모 당 총 전용면적"] = test_df["전용면적"] * test_df["전용면적별세대수"]

# 전용면적 당 차량등록수 피처 생성
train_df["전용면적 당 차량등록수"] = (train_df["세대수 규모 당 총 전용면적"] / train_df["단지별 총 전용면적"]) * train_df["등록차량수"]
```

- 다중공선성으로 예상되는 피처들 제외 (단지내주차면수, 총세대수, 전용면적 등등)
- 타깃 레이블 변환 (총 등록차량 수 → 전용면적 당 규모별 등록차량수)
- 피처들과 타깃 레이블간의 상관관계 분석 및 선형성 검증


### 2. 릿지 알고리즘을 활용하게 된 이유

- 설계한 핵심 피처들의 선형성을 검증 → 단순 선형 회귀 모델의 적합도 가능성 판단
- 작은 데이터 셋의 규모 → GBDT 계열 알고리즘의 과대적합성이 높을 것으로 판단

```python
# 랜덤포레스트 모델 예측 성능 평가 -> MAE 14.1685

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

rf_reg = RandomForestRegressor(n_jobs=-1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25, random_state=42)

rf_reg.fit(X_train, y_train)
pred = rf_reg.predict(X_test)

mean_absolute_error(np.expm1(y_test), np.expm1(pred))
```

```python
# 라쏘 모델 예측 성능 평가 -> MAE 16.8551

lasso_model = Lasso(alpha=0.01)
X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25, random_state=42)

lasso_model.fit(X_train, y_train)
pred = lasso_model.predict(X_test)

mean_absolute_error(np.expm1(y_test), np.expm1(pred))
```

```python
# 릿지 모델 예측 성능 평가 -> MAE 14.9751

ridge_model = Ridge(alpha=20)
X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25, random_state=42)

ridge_model.fit(X_train, y_train)
pred = ridge_model.predict(X_test)

mean_absolute_error(np.expm1(y_test), np.expm1(pred))
```

- train_test_split 및 gridsearchcv를 이용해 최적 파라미터를 찾고, MAE 예측 성능 평가

```python
# 선형회귀 모델 최적의 alpha 값 탐색

params = {"alpha": [0.01, 0.1, 0.3, 0.5, 1, 3, 5, 10, 20]}

elastic_params = {"alpha": [0.01, 0.1, 0.3, 0.5, 1, 3, 5, 10, 20],
                 "l1_ratio": [0.1, 0.3, 0.5, 0.7, 1]}

ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()

grid_ridge = GridSearchCV(ridge, param_grid=params, cv=5, scoring="neg_mean_absolute_error")
grid_lasso = GridSearchCV(lasso, param_grid=params, cv=5, scoring="neg_mean_absolute_error")
grid_elastic = GridSearchCV(elastic, param_grid=elastic_params, cv=5, scoring="neg_mean_absolute_error")

grid_ridge.fit(X_train_df_log, y_train_df_log)
grid_lasso.fit(X_train_df_log, y_train_df_log)
grid_elastic.fit(X_train_df_log, y_train_df_log)

grid_ridge.best_params_
```

```python
importance = ridge_model.coef_
feature_importance = pd.Series(data=importance, index=X_train_df_log.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(20, 12))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.show()
```

- 데이터 셋의 규모와 피처 다양성을 고려 → 데이터 셋의 모든 정보를 활용하기 위해 "릿지" 적합

![%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/5.png](%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/5.png)

![%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/6.png](%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/6.png)

## 💻 Catboost 하이퍼 파라미터 튜닝

- 1. 최적의 파라미터 값 찾기(GridSearchCV)

    > **GridSearchCV 란?**                   
    하이퍼 파라미터 그리드에 기술된 모든 파라미터를 편리하게 찾게 해주는 것입니다.
    하지만 동시에 순차적으로 파라미터를 다양하게 테스트 하므로 수행시간이 상대적으로 오래걸립니다.

    ```python
    params = {'iterations': [200,300,400],
              'depth': [4,5,6],
              'learning_rate': [0.2,0.4,0.5],
              'l2_leaf_reg': [7,8,9]}

    cat_reg = CatBoostRegressor(random_seed=42)
    grid_model = GridSearchCV(cat_reg, param_grid=params, cv=3, scoring="neg_mean_absolute_error")
    gridmodel.fit(X_train_df_log, y_train_df_log)

    grid_model.best_params_
    {'depth': 4, 'iterations': 200, 'l2_leaf_reg': 9, 'learning_rate': 0.2}
    ```

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25,
                                                        random_state=42)

    pred = cat_boost_best.predict(X_test)
    mean_absolute_error(np.expm1(y_test), np.expm1(pred))
    >>> 9.962978529209792
    ```

    - 결과 : 104.6389221828
- 2. 최적의 파라미터 값 찾기

    그리드 안에 다양한 수를 넣으며 비교 해본 결과,
    Best parameter를 찾기에 너무 오랜시간이 걸려 값을 한개씩 대입해 보았습니다.

    ```python
    params = {'iterations': [300],
              'depth': [3],
              'learning_rate': [0.1],
              'l2_leaf_reg': [1]}

    cat_reg = CatBoostRegressor(random_seed=42)
    grid_model = GridSearchCV(cat_reg, param_grid=params, cv=3, scoring="neg_mean_absolute_error")
    grid_model.fit(X_train_df_log, y_train_df_log)

    grid_model.best_params_
    >>> {'depth': 3, 'iterations': 300, 'l2_leaf_reg': 1, 'learning_rate': 0.1}
    ```

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25,
                                                        random_state=42)

    pred = cat_boost_best.predict(X_test)
    mean_absolute_error(np.expm1(y_test), np.expm1(pred))
    >>> 10.731237926130666
    ```

    - 결과: 92.0333993119

    **MAE=10.95 > 88점**

    **MAE=10.82 > 87점** 

      - MAE값을 10.82보다 미세하게 낮추기 위해 하이퍼 파라미터 조정

    **MAE=10.79 > 90점**

    **MAE=10.73 > 89점**

    **MAE=10.42 > 89점**

    **MAE=9.96 > 104점**

    ```python
    params = {'iterations': [100,200,300],
              'depth': [2,3,4],
              'learning_rate': [0.1,0.2,0.3],
              'l2_leaf_reg': [1,2,3]}

    cat_reg = CatBoostRegressor(random_seed=42)
    grid_model = GridSearchCV(cat_reg, param_grid=params, cv=3, scoring="neg_mean_absolute_error")
    grid_model.fit(X_train_df_log, y_train_df_log)

    grid_model.best_params_
    >>> {'depth': 4, 'iterations': 100, 'l2_leaf_reg': 2, 'learning_rate': 0.2}
    ```

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_train_df_log, y_train_df_log, test_size=0.25,
                                                        random_state=42)

    pred = cat_boost_best.predict(X_test)
    mean_absolute_error(np.expm1(y_test), np.expm1(pred))
    >>> 10.802095654274565
    ```

    - 결과: 89.461210929

    하이퍼 파라미터 튜닝을 하면 정확도가 오히려 낮아지기 때문에 기본 Catboost 모델을 사용하기로 결정했습니다.

---

![%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/Untitled.png](%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/Untitled.png)

> 이 말고도 다양한 값을 대입하여
하이퍼 파라미터를 측정하였지만,,, 
기록의 미숙함으로 다양한 값의 결과를 
보여드리지 못한 점 죄송합니다... 

개선할점: 모든 시행 착오들을 기록해 
가면서 어떠한 점이 부족했는지 상세히
기록해 놓아야 겠다고 다짐했습니다,,^^

## 🏡 임대보증금 예측

```python
from sklearn.model_selection import train_test_split
sel = [ '전용면적', '자격유형','임대료', '단지내주차면수','공급유형_lbl',
				'지역_lbl','도보 10분거리 내 버스정류장 수','qcut_총세대수']
X = all_df_1[sel]
y = all_df_1['임대보증금']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=0,)
```

![%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/image.png](%E1%84%86%E1%85%A9%E1%84%8B%E1%85%A7%E1%84%87%E1%85%AA%E1%84%8B%E1%85%AD_%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE%E1%86%BA%E1%84%8B%E1%85%B5%E1%87%81%E1%84%89%E1%85%AE%E1%87%81%20f4667dbda1884d4e84347a2eaec25569/image.png)

### 임대보증금 히트맵

- 임대보증금과 최소한 0.1이상의 상관관계가 있는 Features 선택
- 임대건물구분_lbl 오류? 발생

```python
all_df["임대보증금"] = all_df["임대보증금"].fillna(0)
all_df.loc[all_df["임대보증금"] == "-", "임대보증금"] = 0
all_df["임대보증금"] = all_df["임대보증금"].astype(int)

all_df_0=all_df.loc[all_df['임대보증금']==0]
all_df_1=all_df.loc[all_df['임대보증금']!=0]
all_df_1['임대건물구분_lbl'].value_counts()

1    3110
Name: 임대건물구분_lbl, dtype: int64
```

- 오류 이유는 임대보증금 NULL값&(-)을 모두 0으로 변환후 0을 기준으로 나누는 과정에서  all_df_1['임대건물구분_lbl'] 속에 value 종류가 1(아파트)만 존재!  단일값의 Feature는 다른 변수와 상관관계 분석 불가

## 🤖 추후 개선 사항

- 문제점 - 원핫인코딩 처리 후, 데이터 피쳐들의 개수가 많아졌다.

```python
train_df.columns
Index(['단지코드', '총세대수', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '자격유형',
       '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수',
       '단지내주차면수', '등록차량수', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
       'K', 'L', 'M', 'N', 'O', '실거주세대수', '세대수비율주차면수', '강원도', '경기도', '경상남도',
       '경상북도', '광주광역시', '대구광역시', '대전광역시', '부산광역시', '세종특별자치시', '울산광역시', '전라남도',
       '전라북도', '제주특별자치도', '충청남도', '충청북도', '상가', '아파트', '세대수 규모 당 총 전용면적',
       '단지별 총 전용면적', '전용면적 당 차량등록수'],
      dtype='object')

test_df.columns
Index(['단지코드', '총세대수', '지역', '공급유형', '전용면적', '전용면적별세대수', '공가수', '자격유형',
       '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)', '도보 10분거리 내 버스정류장 수',
       '단지내주차면수', 'A', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
       'B', 'F', 'O', '실거주세대수', '세대수비율주차면수', '강원도', '경기도', '경상남도', '경상북도',
       '광주광역시', '대구광역시', '대전광역시', '부산광역시', '세종특별자치시', '울산광역시', '전라남도', '전라북도',
       '제주특별자치도', '충청남도', '충청북도', '상가', '아파트', '세대수 규모 당 총 전용면적',
       '단지별 총 전용면적'],
      dtype='object')
```

- 그러나, 모델 예측에 사용하는 피쳐들의 개수는 정해져 있고,  이러한 모델 예측에 필요한 데이터의 피쳐의 개수는 줄이되, 최대한 데이터 손실을 없애는 방향으로 개선이 필요하다.
- 개선사항 - 개별적인 피쳐들의 상관관계가 높은 변수들끼리 피처를 결합하여 모델을 개선해 볼 수 있다.

---

- 문제점 - 카테고리 변수들을 다양하게 활용해 보지 못했다.
- 개선사항 - 시간이 더 있다면  향후에는 지역별(수도권, 비수도권), age 데이터의 다양한 변수 등과 같은 다른 데이터들을 활용하여 최적의 주차수요 예측을 위한 모델 개발을 고안해 볼 것이다.

[🙌 결론 및 소감](https://www.notion.so/516b8117251b4aa8a5d03321b03bf4dc)

[ComeOn_LeafForest](https://github.com/comeon-leaf-forest)
