---
layout: post
title: "[데이터분석 정리] black box interpretation.  Individual Conditional Expectation 개인적 정리"
categories:
  - 머신러닝
tags:
  - black box
  - Individual Conditional Expectation
  - ICE
  - Centered ICE
comment: true
---

앞선 글에서 black box 모델에서 변수의 영향을 근사적으로 파악하기 위한 방법을 PDP를 다뤘다. 이번엔 pdp의 단점들을 보완하기 위한 방법들에 대해 다뤄본다. 구체적으로 Individual Conditional Expectation (ICE)와 Centered ICE Plot이다.

# Individual Conditional Expectation (ICE)

앞서 다룬 Partial dependency plot은 모든 train data에 대해 예측값을 평균 내어 관심변수의 영향력을 근사하였다. 그러나 이 평균, 즉 통합에서 오는 문제가 있다. 예를 들어 분포가 매우 skewed 되어 있는 경우, mean은 오히려 직관적이지 못한 대표값을 만들어낼 수 있다. 

'통합'으로 인해 생기는 문제를 보완하기 위한것이 individual conditional expectation curves이므로, 해결책은 간단하다. 각각의 데이터에 대해서 관심변수 $$X_s$$ 가 변화될때 어떻게 예측값이 변하는지를, **모든 train데이터에 대해 보여주는 것**. 즉, PDP는 ICE의 각 line을 average취한 하나의 선이다.  식으로 표현하면 엄청 간단한데, 특정 데이터i에 대해서 ICE line은 다음과 같다. 

$$
f(X_s)^{(i)}=f(X_s,X_c^{(i)}), i=1,2,..,n
$$

여기서 $$X_s$$값을 이리저리 넣는것.

과정을 좀더 상세하게 설명해보자면 다음과 같다.

크기 (n X q)의 데이터 matrix 에서, 다른 변수들 $$X_c$$에 관련된 컬럼들은 그대로 두고, 관심변수 $$X_s$$를 feature space에 lower bound에서부터 upper bound까지 값을 바꾼다. (각 $$x_s$$의 값 하나당 (n X q)의 matrix가 만들어지는 셈) 이를 mean을 취하면 그게 pdp의 $$f(X_s)$$, 이를 그냥 다 그림에 때려박으면 ICE의 $$f(X_s)^{(i)}$$이다. 

그림으로 보면 바로 이해되는데, 다음과 같다. (아래의 선들을 average한게 PDP다) 

<img width="545" alt="ICE plot" src="https://user-images.githubusercontent.com/31824102/55941019-4a137500-5c7c-11e9-8f3d-4f35c2d06712.PNG">

이를 보면 대부분의 데이터들에 대해 결과값 (predicted cancer probability)가 0.0~0.1 정도에 몰려 있음을 알 수 있다. 즉, 상당히 right skewed된 형태이다. 그러나 이를 mean을 취하면, 0.4를 넘는 몇몇의 극단값들에 의해 의미가 왜곡될 수 있다. (예를 들어 평균값이 0.2정도 나온다던지) 따라서 단순하게, 모든 train data에 대해 line을 그려본다. 이를 통해 PDP에선 발견하지 못했던 상세한 관계를 발견할수 있게 된다. 

< 단점 >

그러나 여전히 같은 주의할점이 있다. (앞선 pdp와 같이,)실제 데이터 분포를 유의해야하고, 관심밖의 변수가 correlated되어 있을 경우 잘못된 결과가 나올 수 있다.

그리고 추가적인 단점은 다음과 같다. 

- 1차원밖에 못본다. 선을 여러개 그리는건 가능하지만, 면을 여러개 그리면 알아볼 수 없다. (사실상 50겹 크레이프)
- 데이터가 많으면, 당근 보기 힘들어진다. 그냥  검은도화지..

# Centered ICE Plot

또다른 plot 방식이다. ICE가 너무 선들이 난잡해서, 그래서 $$X_s$$에 따라 결국 예측값이 올라가는건지 아닌지 등 trend를 보기 힘들때가 있다. (위 그림에서 age 25~40정도가 그런 느낌이다.) 그래서 모든 ICE의 선을 (0,0)에서 시작하도록 만든것이 c-ICE plot 이다. 식은 다음과 같다.

$$
f_{cent}(X_s)^{(i)}=f(X_s)^{(i)}-f(x^*,X_{c}^{(i)})=f(X_s,X_c^{(i)})-f(x^*,X_{c}^{(i)})
$$

예를 들어 위의 예시에서 보면 나이의 minimum값이 13인데, $$x^*=13$$으로 해서 모든 i에 대해 $$f_{cent}(13)^{(i)}=0$$이 되도록 하는 것이다. 이를 통해 평행선들의 고도차이를 없애주고 차이의 trend에 집중하게 된다. 

> 편의를 위해 minimum값으로 고정한다고 했지만 그냥 한점에서 만나면 되므로,$$X_s$$의  maximum값등 암거나 $$x^*$$가 될 수 있다. 그러나 (0,0)지나가는게 이쁘니까 minimum하자

그림으로 보면 다음과 같다.

<img width="548" alt="ICE plot2" src="https://user-images.githubusercontent.com/31824102/55941018-4a137500-5c7c-11e9-9564-abc782a3b9a3.PNG">

< 주의할점 >

평행선들의 level을 모두 맞춰주고 그림을 그린것이기에, absolute prediction이 아닌, prediction의 trend에 관심이 있을때 활용해야한다는것이 주의하자.