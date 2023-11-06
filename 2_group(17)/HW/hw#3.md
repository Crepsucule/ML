# 1. 기계 학습에서 학습이란 무엇인가?
기계 학습에서의 학습이란 훈련 데이터로부터 가중치 매개변수를 조정하고 손실 함수를 최소화하는 것을 의미한다.
이 때 가중치와 손실함수란 다음과 같다.
1. 가중치 : 각 신호의 영향력을 조절하는 매개변수이며, 이를 조정해서 모델의 성능을 최적화한다.
2. 손실 함수 : 손실 함수란 모델의 예측값과 실제 값의 차이를 측정하는 함수로 이를 최소화하여 모델의 정확도를 향상시킬 수 있다. 대표적인 손실함수로는 평균제곱오차(MSE)가 있다.

# 2. 획률적 경사 하강법 소스코드 분석
```Python
  n_epochs = 50
  t0, t1 = 5, 50 # 학습 스케쥴 파라미터

  def learning_schedule(t):
    return t0 / (t + t1)

  theta = np.random.randn(2, 1) # 무작위 초기화

  for epoch in range(n_epochs):
    for i in range(m):
      random_index = np.random.randint(m)
      xi = X_b[random_index:random_index+1]
      yi = y[random_index:random_index+1]
      gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
      eta = learning_scheduel(epoch * m + i)
      theta = theta - eta * gradients
```
1. 'n_epochs'는 전체 데이터셋에 대해 학습하는 횟수를 나타낸다.
2. 't0'와 't1'은 학습 스케쥴 파라미터로 학습률을 조정한다.
3. 'learning_schedule' 함수는 학습률을 계산하는 함수이다.
4. 'theta'는 임의의 값으로 지정한다.
5. for문을 이용해서 각 스텝마다 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플(random_index)에 대해서 'gradients'를 계산한다.
6. 'eta'는 계산된 학습률을 뜻한다.
7. 'theta'에서 학습률과 기울기르르 곱한 값을 빼면서 매개변수를 업데이트한다.

즉, 이 코드는 매 스텝마다 한 개의 샘플을 랜덤으로 선택하고 그 하나의 샘플 대한 'gradients'를 계산하고 매개변수인 'theta' 값을 조정하는 소스코드이다.
