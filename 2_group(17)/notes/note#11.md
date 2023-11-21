# 비지도 학습
레이블이 없는 데이터를 통해서 학습을 진행하는 기법 - 가장 잠재력이 큰 학습 방법

## 군집(Clustering)
비슷한 샘플을 클러스터로 모음.

**군집을 이용한 이상치 탐지, 밀도 추정 -** 밀도가 높은 곳은 정상으로 보고 낮은 곳은 이상치라고 봄.

군집을 만드는 방법 (Clustering) 

1. 각 샘플을 하나의 그룹에 할당
2. 비슷한 샘플을 구별해 하나의 클러스터 또는 비슷한 샘플의 그룹으로 할당

**K-평균(로이드-포지 알고리즘)** - 각 **군집의 중심(센트로이드)을 찾고** 가장 가까운 군집에 샘플을 할당하는 방법

문제점 1. 센트로이드를 어떻게 결정할 것인가? 

방법 1. 랜덤 센트로이드 결정

1. 처음에는 센트로이드를 랜덤하게 설정
2. 각 샘플을 가장 가까운 센트로이드에 할당
3. 군집별로 샘플의 평균을 계산하여 센트로이드를 지정
4. 수렴할 때 까지 2, 3번 반복

→ 센트로이드 초기화가 운이 나쁘면 최적의 솔루션이 아니게 됨

방법 2. 관성(inertia, 이너셔) - 센트로이드 초기화 방법 

정의 : 샘플과 가장 가까운 센트로이드와의 거리의 제곱의 합

→ 관성의 값이 가장 낮은 모델이 좋은 모델

*K-평균++: 센트로이드를 무작위로 초기화하는 대신 특정 확률분포를 이용(== 다중 초기화)

K-평균 속도 개선: elkan알고리즘→불필요한 거리 계산을 피함으로 학습 속도 향상됨

미니배치 K-평균: 각 반복마다 미니배치를 사용하여 센트로이드를 조금씩 이동

→K-평균 알고리즘보다 훨씬 빠름, 이너셔는 일반적으로 조금 더 나쁨, 군집수가 증가할 때 이니셔는 더 나쁨

문제점 2. 클러스터 갯수를 얼마로 설정할 것인가?

**관성과 클러스터** - 클러스터 개수 k가 증가할 수록 관성이 작아지므로, 좋은 성능 지표가 아님 

**실루엣 점수와 클러스터 개수** - 모든 샘플에 대한 **실루엣 계수의 평균을 계산**하는 방법

- 실루엣 계수 : -1과 +1사이의 값
    
    +1에 가까운 값 : 자신의 클러스트 안에 포함되고, 다른 클러스트와 멀어짐
    
    0에 가까운 값 : 클러스트 경계에 위치
    
    -1에 가까운 값 : 샘플이 잘못된 클러스터에 할당됨
    
- 실루엣 다이어그램과 클러스트 개수
    - 실루엣 다이어그램 : 클러스터별 실루엣 계수 모음. 칼 모양의 그래프
        - 칼 두께 : 클러스트에 포함된 샘플의 개수
        - 칼 길이 : 클러스터에 포함된 샘플의 실루엣 계수(길 수록 좋음)
    - 빨간 파선 : 클러스터 계수에 해당하는 실루엣 점수
    
    → 칼의 두께가 모두 비슷하고, 칼의 길이가 모두 빨간 파선보다 길어야 좋은 모델

장점 - 속도가 빠르고 확장이 용이함

단점

- 클러스터의 크기, 밀집도가 서로 많이 다르거나 원형이 아닌 경우(ex 타원형) 잘 작동하지 않음(샘플과 센트로이드의 거리만 고려되기 때문)
- 최적이 아닌 솔루션을 피하려면 알고리즘을 여러 번 실행해야 함
- 클러스터 개수를 미리 지정해야 함

하드 군집 : 각 샘플에 대한 가장 가까운 클러스터 선택

소프트 군집  : 각 클러스터가 샘플 별로 센트로이드와의 거리를 측정해서 샘플에 점수를 부여하는 방법

### 군집을 사용한 이미지 분할
이미지 분할: 이미지를 세그 먼트 여러 개로 분할 → 다영한 클러스터 개수로 k-평균ㅇㄹ 사용해 만든 분할

시맨틱 분할: 동일한 종류의 모든 픽셀을 같은 세그 먼트에 할당

색상 분할: k-평균을 이용하여 색상분할 실
### 군집을 사용한 전처리
K-평균을 전처리 단계에 사용한 경우 데이터 셋의 차원이 감소

클러스터 갯수를 GridSearchCV를 사용해 최적의 클러스터 갯수를 찾음.
### 군집을 사용한 준지도 학습
준지도 학습 - 레이블이 없거나 적을 때 사용하는 학습 방법

예제로 미니 Mnist 데이터 셋을 이용한 경우 훈련 세트를 샘플 수 만큼의 클러스터로 모아서 수동으로 레이블을 할당하는 경우 성능이 향상

레이블 전파 - 동일한 클러스터에 있는 모든 샘플로 전파 → 센트로이드에 가까운 20%정도만 레이블 전파 후 학습
### DBSCAN
집된 연속적 지역을 클러스터로 정의

주요 하이퍼 파라미터 2개

1. eps : 이웃 범위
2. min_samples : eps 반경 내에 위치하는 이웃의 수

핵심샘플과 군집

- 핵심 샘플 : eps 반경 내에 자신을 포함해서 min-samples개의 이웃을 갖는 샘플
- 군집 : 핵심샘플로 이루어진 이웃들로 구성된 그룹

이상치 - 핵심샘플이 아니면서 동시에 핵심샘플의 이웃도 아닌 샘플

예측 - predict()지원x → fit_predict()지원

장단점 - 매우 간단하고 강력한 알고리즘으로 군집의 모양과 개수 상관없음, BUT 군집 간의 밀집도가 크게 다르면 모든 군집 파악 불가능
# 가우시안 혼합
## 가우시안 혼합 모델
샘플이 파라미터가 알려지지 않은 여러 개의 혼합된 가우시안 분포에서 생성되었다고 가정하는 확률 모델

가우시안 분포 = 정규 분포
**M 알고리즘** - 라벨을 얻기 위해서는 분포가 필요하고, 분포를 얻기 위해서는 라벨이 필요한 경우를 해결하기 위한 알고리즘

1. 초기값: 분포도를 랜덤하게 설정함
2. E_Step: 분포도를 통해 라벨링
3. M-Step: 라벨링 한 것을 토대로 평균 값을 구해서 분포도 업데이트
4. 1~3번을 반복

→ EM알고리즘을 이용한 사우시안 혼합 과정(정규분포의 변화 과정)

Start                               반복

랜덤하게 분포 제안 → [likelihood비교로 라벨링 → 각 그룹별 모수 추정 → 추정된 모수를 이용한 각 그룹별 분포 도시]

GMM 모델 규제 - 특성 수가 크거나, 군집 수가 많거나, 샘플이 적은 경우 최적 모델 학습이 어려워서 규제를 가해서 학습에 도움을 줌

**가우시안 혼합을 사용한 이상치 탐지** - 밀도가 임곗값보다 낮은 지역에 있는 샘플을 이상치로 간주, 정밀도/재현율 트레이드 오프

**클러스터 개수 선택**

군집이 타원형일 때 값이 일정하지 않기 때문에 관성, 실루엣 점수 사용 불가

이론적 정보 기준을 최소화 하는 모델 선택 가능(기준 = BIC, AIC)

**베이즈 가우시안 혼합 모델**

BayesianGaussianMixture 모델 - 최적의 군집수를 자동으로 찾아줌, 자동으로 불필요한 군집 제거

사전믿음 - 군집수가 어느 정도일까를 나타내는 지(weight_concentration_prior 하이퍼파라미터)

가우시안혼합 모델 장단점

장점 - 타원형 클러스터에 잘 작동

단점 - 다른 모양을 가진 데이터셋에서는 성능 좋지 않음(억지로 타원을 찾으려고 시도함)