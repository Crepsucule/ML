## 인공신경망(Artificial Neural Network)

뇌에 있는 생물학적 뉴런의 네트워크에서 영감을 받는 머신러닝 모델

생물학적 뉴런 → 인공 뉴런

**인공뉴런** - 생물학적 뉴런에 착안한 매우 단순한 신경망 모델→하나 이상의 이진(on/off)입력, 이진 출력 하나를 가짐

**퍼셉트론** - 가장 간단한 인공신경망 구조(TLU or LTU라 불리는 인공 뉴런을 활용)

TLU(Threadhold Logic Unit) - 입력값과 가중치를 곱한 값들의 합에 계단 함수 적용

TLU 모델 학습 = 최적의 가중치 𝑤i찾기

![12_1](https://github.com/Crepsucule/ML/assets/86185735/c624ef7a-7905-450b-9830-7c8ee8c2615b)

하나의 TLU를 이용해서 이진분류기로 활용가능 - 모든 입력값(특성)의 선형 조합을 계산한 후 **임계값을 기준으로 양성/음성 분류**

퍼셉트론 학습 알고리즘 

- 오차가 감소되도록 가중치를 조절하며 뉴런 사이의 관계를 강화시킴.
- 하나의 샘플이 입력될 때마다 예측한 후 오차를 계산하여 오차가 줄어드는 방향으로 가중치를 조절(경사하강법과 비슷)
- 결정경계가 선형이며 복잡한 패턴을 학습하지 못함.

**다층 퍼셉트론** - 퍼셉트론을 여러개 쌓아올린 인공신경망

![12_2](https://github.com/Crepsucule/ML/assets/86185735/8506b662-3e42-43af-bbac-65199c5e424c)

다층 퍼셉트론의 모습

완전 연결층 - 층에 속한 각각의 뉴런이 이전 층의 모든 뉴런과 연결되어 있을 때

**심층신경망(DNN)** - 여러 개의 은닉층을 쌓아올린 인공신경망 


역전파 훈련 알고리즘 - 다층 퍼셉트론의 층이 많아질수록 훈련시키는 과정이 어려워지는 것을 해결하기 위해 나타난 훈련 알고리즘

1. 정방향 : 예측을 만든 후 오차 측정(입력층 → 출력층)
2. 역방향 : 역방향으로 각 층을 거치면서 각 출력연결이 오차에 기여한 정도 측정(미분의 연쇄법칙, 출력층 → 입력층)
3. 가중치 조정 : 오차가 감소하도록 모든 가중치 조정

MLP특징

- 랜덤 설정
- 활성화 함수: 로지스틱(시그모이드), 하이퍼볼릭 탄젠트 함수(쌍곡 탄젠트 함수), ReLU함수

→활성화 함수 대체 필요성: 선형성(계단 함수라)을 벗어나기 위해, 비선형 활성화 함수를 충분히 많은 층에서 사용하면 매우 강력한 모델 학습 가능

**회귀를 위한 다층 퍼셉트론**

출력 뉴런 수 - 예측해야 하는 값의 수에 따라 출력 뉴런 설정

활성화 함수 지정 - 출력값에 특별한 제한이 없다면 사용하지 않음(출력이 양수인 경우엔 주로 ReLU함수 사용)

손실 함수 - 평균제곱오차(MSE) 활용, 이상치가 많은 경우 → 평균절댓값오차(MAE) 사용 가능

**분류를 위한 다층 퍼셉트론**

이진분류 - 하나의 출력 뉴런 사용 → 활성함수는 로지스틱 함수

다중레이블 이진분류 - 다층 퍼셉트론 활용, 출력층 활성화 함수 → 소프트맥스 함수

손실함수 - 크로스 엔트로피

### 케라스로 다층 퍼셉트론 구현

**시퀀셜 API 활용한 이미지 분류**

Sequential 클래스 내에 층을 쌓아 순차적 학습 지원

```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation="softmax"))
```

층의 리스트를 전달함으로써 모델을 만드는 것도 가능

**모델 컴파일**

손실함수, 옵티마이저, 평가기준 등을 지정함

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

**모델 훈련**

```python
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
```

모델 학습 곡선-훈련된 모델이 반환하는 History객체의 history속성에 학습된 에포크의 손실과 정확도 기록됨

**케라스 시퀀셜 API 활용 : 회귀**

Sequential 클래스를 이용한 회귀용 MLP 구축은 분류용과 기본적으로 동일.

차이점 1. 출력층에 활성화함수를 사용하지 않는 하나의 뉴런만 사용(하나의 값만 예측하기 때문

차이점 2. 손실함수 : 평균제곱오차(MSE)

**모델 생성, 훈련 및 평가**

```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)
```

**함수형 API를 사용해 복잡한 모델 만들기**

기존의 모델 → 레이블을 순차적 처리,  함수형 API 활용 → 순차적으로 처리하지 않는 다양한 신경망 구축 가능

**와이드 & 딥 구조**

깊은 경로 → 복잡한 패턴 학습,  짧은 경로 → 간단한 규칙 학습

![12_3](https://github.com/Crepsucule/ML/assets/86185735/23d58957-f4c0-446a-aeef-b1d53790d2ec)

**활용방법 1**

**모델 층 정의**

```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])
```

**모델 생성, 학습 및 평가**

```python
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)
```

**활용방법 2 - 다중 입력 사용(특성을 나누어 짧은 경로, 깊은 경로에 특성을 나누어 보냄)**

입력 뉴런마다 입력값 지정하여 학습해야 함

![12_4](https://github.com/Crepsucule/ML/assets/86185735/8998c91e-f274-4672-b577-b544354da43b)

**********************활용방법 3 - 다중 출력이 필요한 경우(보조 출력 층을 이용)→출력이 2개라 손실함수도 2**********************

동일한 데이터에서 독립적인 여러 작업 수

![12_5](https://github.com/Crepsucule/ML/assets/86185735/d50bc6ea-92a3-4d4a-b182-b7b4492a218d)

**서브클래싱 API로 동적 모델 생성**

Sequential 클래스, 함수형 API → 선언적 방식으로 정적임

Model 클래스 상속

- 초기설정 메소드를 이용하여 은닉층, 출력층 설정
- call() 메소드로 층을 동적으로 구성

동적 모델의 단점

- 모델 구조가 call() 메소드 안에 숨겨져 있어 케라스가 분석하기 어려움
- 모델 저장 및 복사 불가능
- 층간의 연결정보 확인 불가능

**모델 저장과 복원**

Sequential모델과 함수형 API를 사용하여 훈련된 모델 저장

→ 케라스(모델 구조와 층의 모든 파라미터 저장 ), 옵티마이저도 저장(하이퍼파라미터와현재상태 포함 )

복원 - load_model()함수 활용

주의사항

- 서브클래스 API에서 사용불가
- save/load_weights()메서드를 활용하여 모델 파라미터만 저장/복원 가능
- 나머지는 수동처리

**콜백 사용하기**

체크 포인트 저장 시 사용 - 대규모 데이터 셋에서 훈련 시에 훈련 도중 일정 간격으로 체크 포인트를 저장해야함. (최적의 값을 알아내는데도 활용)

훈련의 시작/끝/중간에 호출할 객체를 콜백함수를 이용하여 지정 가능

조기 종료 구현 - 일정 epoch 동안 검증세트에 대한 점수가 향상되지 않으면 자동 종료

사용자 정의 콜백함수 - 상속 과정에서 이미 선언된 콜백함수 메서드 중에서 필요한 메서드를 재정의하면 됨

**텐서보드** - 훈련하는 동안의 학습곡선, 여러 실 간의 학습곡선 비교, 계산 그래프 시각화…

### 신경망 하이퍼 파라미터 튜닝

**그리드탐색/랜덤탐색 활용 - 오래 걸림**

하이퍼 파라미터 최적화를 위한 다양한 라이브러리 존재

더 좋은 기법 - AutoML : **자동**으로 최적의 하이퍼파라미터를 찾고 최적의 아키텍쳐을 만듦
