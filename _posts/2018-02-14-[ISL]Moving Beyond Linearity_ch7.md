---
layout: post
title: "[ISL] 7장 -비선형모델(Local regression, Smoothing splines, GAM) 이해하기"
categories:
  - 머신러닝
tags:
  - An Introduction to Statistical Learning
  - Machine Learning
comment: true
---



{:toc}



지금까지는 linear model들에 대해 다뤄왔다. standard linear model은 모델의 단순성으로 인해 해석과 추론이 쉽다는 장점이 있으나, 예측력이라는 중요한 부분에서 한계를 가진다. 선형모델은 현실의 문제에 선형성 가정을 하는 것이고, 쉽게 예상 가능하다시피 이는 몇몇의 경우 터무니 없는 가정이 되기도 하기 때문이다. 6장에서의 Lasso와 Ridge, PCR등을 통해 선형모델의 variance를 줄이는 방법을 다루웠으나, 이 역시 '선형' 모델이라는 것에는 변함이 없다. 따라서 이번에는, 해석력은 가능한 잃지 않으며 **선형가정을 완화시킬 수 있는 방법들**에 대해 다뤄보겠다. 이 장에 다룰 내용들을 미리 간략하게 소개하자면 다음과 같다.

- Polynomial regression은 기존의 변수의 다차항($$X^2,X^3$$등)을 추가하여 non-linear data에 적합을 할 수 있도록 선형모델을 확장한다
- Step function은 변수를 $$K$$개의 부분으로 나누어, 질적변수(즉, constant)를 만들어낸다. 이는 piecewise constant function(후에 자세히 설명)을 적합하는 효과를 같는다.
- Regression splines은 위의 두 방식의 확장으로, 전체 $$X$$를 똑같이 $$K$$개의 범주로 나누되 각 범주내에서 다항적합을 하는 것이다. 이때 다항적합은 양 옆의 범주의 다항함수와 매끄럽게(smoothly) 연결되도록 한다는 제약이 있다. 적당한 범주로 나눌 경우, regression spline은 매우 유연한 적합을 가능하게 한다.
- Smoothing splines는 regression splines과 비슷하지만, smooth penalty를 포함한(뒤에 자세히 설명한다) SSE식을 최소화하는 방식으로 적합을 한다.
- Local regresson은 spline방식과 유사하지만 각 범주가 **겹칠수 있다**는 점에서 다르다. 이러한 방식으로 더욱 유연한 적합을 가능하게 한다.
- Generalized additive model은 위의 방식들을 **여러개의 예측변수들**에 적용할 수 있게 하는 방식이다.

위의 순서대로 앞에선 우선 예측변수가 1개인 경우($$X_1$$)에 대해 다룰 것이고, 이를 뒷부분에 가서 여러 변수들($$X_1,..,X_p$$)로 확장할 것이다.

## 7.1 Polynomial Regression

non linear한 데이터에 적합하는 가장 기본적인 방식은 다항식을 추가하는 것이었다. 다항회귀를 식으로 나타내면 다음과 같다
$$
y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+..\beta_dx_i^d+\epsilon_i
$$
> $$\epsilon$$에도 i가 들어가 있다. 선형회귀에서는 모든 $$\epsilon_i$$가 $$N(0,1)$$로 동일하다 가정했기에 없던 것이고, 원래는 이렇게 표현. 물론 이경우도 mean0의 가정은 (주로)유효하다. 평균이 0인 수많은 다른 분포를 가질 수 있기에 i로 구분해서 쓴다.

각 계수들은 단순히 least square로 적합을 하면 된다. 이는 3장에서도 다뤘었지만, 이는 새로운 예측변수로써 다차항을 넣은것과 같다.

> 두번째 예측변수$$X_2$$로써 단순히 $$X_1$$을 제곱한 $$X_1^2(=X_2)$$를 넣어주었다 보면 된다.

일반적으로 3,4차항 이상까지는 포함시키지 않는다.

![nonlin1](https://user-images.githubusercontent.com/31824102/36205902-62ddbbda-1188-11e8-9af7-b41190172a16.PNG)

왼쪽 그림의 파란 선은 Wage~Age의 자료에 대해 4차적합을 한것이고, 점선은 예측값의 분산을 통해 구한 신뢰띄(여기선 $$\pm2*SE$$로 하였다)이다.  예측값, 즉 $$\hat f(x_0)$$의 분산은 단순회귀에서 와 동일한 방식으로 구할 수 있다.

$$\hat f(x_0)=\hat\beta_0+\hat\beta_1x_i+\hat\beta_2x_i^2+\hat\beta_3x_i^3$$라고 하면 각 $$\hat\beta_j$$의 variance와  각 쌍들간의 covariance를 구하면 된다. 이를 각 점에 대해 구하고 선으로 이은것이다. 

그림에서 wage이 소득 250을 기준으로 2개의 다른 집단, 즉 '고소득층'과 '저소득층'으로 유의미하게 나뉘어진다는 것을 알았다고 하자. 이 경우 역시나 다음과 같은 식으로 다항로지스틱 적합을 할수도 있다. 
$$
Pr(y_i>250\lvert x_i)=\frac{exp(\beta_0+\beta_1x_i+\beta_2x_i^2+..\beta_dx_i^d)}{1+exp(\beta_0+\beta_1x_i+\beta_2x_i^2+..\beta_dx_i^d)}
$$
이를 통해 적합된 확률(로지스틱의 경우 확률을 반환하였다는 것을 상기)이 오른쪽 그래프에 나와있다. 고소득층에 대한 데이터가 79개로 적었기에 95%신뢰구간이 매우 넓게 나왔다.

## 7.2 Step Functions

다항함수를 사용하는 것은 예측변수의 **전체** 구간에 non-linear형태를 부여하는 것이다. 반면 step function에서는 전체 X를 **몇개의 구간으로** 나눈다. 그리고, 그 구간마다 일정한 **상수**를 부여하는 것이다. 이는 사실 연속형인 반응변수Y를 몇개의 **범주형 변수로** 바꿔서 부여하는것과 같다.

사실상 구간을 나눠서 계산하는것과 같지만, 이를 식으로 나타내면 다음과 같다.

![nonlin2](https://user-images.githubusercontent.com/31824102/36486473-6c31826e-1716-11e8-8267-ecf2eef8dcd6.PNG)

전체 X를 K개의 범주로 나누고 각 범주에 대해 상수를 부여한것이다. 각 범주에 들어가면 1이고 나머지가 0이된다는 점에서 질적변수때 다루었던 dummy variable과 동일하게 이해해도 된다. intercept인 $$\beta_0$$을 설정하였으므로 $$C_0(x_i)$$가 들어가지 않았다. 이때 $$\beta_0$$는 $$(X<c_1)$$인 데이터의 Y값의 평균이되고 나머지 $$\beta_j$$는 ($$c_j<X<c_{j+1}$$)에 속한 데이터의 평균과 $$\beta_0$$과의 차를 의미하게 된다.(질적변수 적합과 같은 맥락으로 이해하면 된다.) 이 식의 계수들 역시 **least square로 적합하여** 구한다. 또한, 역시나 다음과 같은 식으로 logistic regressoin model에 적합을 할수도 있다.

![nonlin3](https://user-images.githubusercontent.com/31824102/36486467-6ba3d734-1716-11e8-9607-65ce3cf44892.PNG)

이를 통해 적합한 두 그림은 다음과 같다. 3장에서 더미변수에 대한 variance를 구할 수 있듯이 같은 방식으로 신뢰구간을 구할 수 있다.

![nonlin3](https://user-images.githubusercontent.com/31824102/36486470-6bdab43e-1716-11e8-9d22-1b5b4a728843.PNG)

그러나, 왼쪽 그림을 보면 구간을 **어떻게 자르느냐**에 따라 제대로 작동하지 못할 수 있음을 알 수 있다. 첫번째 구간의 경우 명확한 증가추세인데 이를 제대로 구간지어주지 않아 해당 추세를 반영해주지 못한 것이다. 한편 step function은 생물통계학이나 전염병학에서 자주 사용되는데, 주로 5살로 그룹을 지어 나눈다고 한다.

## 7.3 Basis Functions

사실, 앞서 다룬 polynomial과 piecewise-constant regression은 **basis function**의 특별한 형태라고 할 수 있다. basis function는 다음과 같은 식으로 나타낼 수 있다.  
$$
y_i=\beta0+\beta_1b_1(x_i)+\beta_2b_2(x_i)+..+\beta_Kb_K(x_i)+\epsilon_i
$$
Step function과 비슷한 형태인데, $$x_i$$에 대한 **여러 함수들**로 식을 표현하는 것이다. 

이때 각 함수 $$b_k(x_i)$$가 어떠한 형태일지(1차항만인지 2차항도 포함한 형태인지 등)는 물론 분석자가 미리 정해놓는다. 다양한 $$b_k(x_i)$$의 형태에 따라 basis function은 여러가지 개념을 포함할 수 있다. $$b_j(x_i)=x_i^j$$라고 설정하면 polynomial regression($$y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+..\beta_dx_i^d+\epsilon_i$$)이 되는것이고, $$b_j(x_i)=I(c_j<x_i<c{j+1})$$라면 piecewise constant function($$y_i=\beta_0+\beta_1C_1(x_i)+\beta_2C_2(x_i)+..\beta_KC_K(x_i)+\epsilon_i$$)이 되는 것이다. 또한 이 $$b_i(x_i)$$를 각각의 예측변수라고 생각하면, standard linear model로도 생각할 수 있다. 따라서 회귀계수의 standard error, F-통계량 등 3장에서 다루었던 모든것을 역시나 적용할 수 있다. 다음으로는 basis function에서 대표적인, regression splines를 보겠다.

## 7.4 Regression Spines

앞서 살펴본 polynomial regression과 piecewise constant regression을 확장한 유연한 형태의 basis function을 살펴보자

### 7.4-1 Piecewise Polynomials

전체 X의 범위에 고차항의 다항적합을 하는 것이 아닌, 몇개의 범주의 범위에 (비교적) 낮은 차수의 적합을 따로따로 하는 것이 piece wise polynomial regression이다. 예를 들어, 각 범주마다 $$y_i=\beta_{0,k}+\beta_{1,k}x_i+\beta_{2,k}x_i^2+\beta_{3,k}x_i^3+\epsilon_i$$를 적합하는 것이다. (이때 물론 각 회귀계수는 각 범주마다 다르다. 이를 명시하고자 0,k라고 표현했다) 각 범주가 바뀌는 지점을 knots라고 부른다.

예를 들어 knot가 상수c라는 지점에서의 단하나 뿐이라면, piecewise cubic polynomial식은 다음과 같다. (cubic은 3차라는 뜻이다.)

![nonlin4](https://user-images.githubusercontent.com/31824102/36486465-6b6fe564-1716-11e8-8f1f-3b634091f6f4.PNG)

$$(x<c)$$와 $$(x\ge c)$$의 2개의 subset에 두번의 다항적합을 한것이라 보면 된다. 역시 각 계수들은 least sqaure로 적합한다. 이때, 모델의 자유도는 추정할 계수들의 수, 즉 4개씩 2범주이므로 df=8이 된다.

그러나, 해당 조건만으로는 다음과 같은 모델이 만들어진다.

![nonlin5](https://user-images.githubusercontent.com/31824102/36486460-6ae9f17a-1716-11e8-8c02-8ce4b3531244.PNG)

Age=50이라는 knot에서 선이 전혀 연결되지 않는 비합리적인 모델이 만들어져버린다. 49.999..세까지는 Wage가 110달러이다가, 50세가 된 직후 160달러가 될수는 없기 때문이다. 이에 따라 추가적인 방법이 사용된다.

### 7.4-2 Constraints and Splines

Age=50에서 wage가 갑자기 50가까지 뛰는 것은 비합리적이다. 위의 그림처럼 비합리적인 모델이 나오지 않게 하기 위해, 각 적합된 곡선이 서로 연결(continuous)되어야 한다는 제약을 추가한다. 이에 따라 50부근의 선들을 연결해주었다. 

> 어떻게 해당 제약을 줄 수 있을까? 뒤에서 truncated power basis function부분에 나온다. 여기서는 '연속'만의 제약이기에, $$h(x,50)=(x-50)_{+}$$의 제약이 들어갈 것이다. 뒤에 나올 truncated power basis function부분을 읽고 돌아와보면 이해가 가능하다.

![nonlin6](https://user-images.githubusercontent.com/31824102/36486456-6aaebe84-1716-11e8-8d93-9e9ae6b9838c.PNG)

그러나 여전히 V자형의 꺾인 선이 비합리적으로 보인다. (지금은 완만한 V자이지만, 해당 제약만을 가지곤 급변하는 V자형태의 함수가 나올 수도 있다.)

이에따라, 2개의 새로운 제약을 추가한다. Age=50이라는 knots에서, **1차미분, 2차미분이 가능해야** 한다는 것이다. 

> (왜 2차까지 일까? 사람의 눈에 매끄럽게 보이기 위해선 d-1차까지 continuous derivative, 즉 연속이고 미분가능해야 한다고 한다.) 

이는 Age=50이라는 점에서 continous뿐 아니라 매끄러워야 한다는 제약을 추가한 셈이다. 이의 결과는 다음과 같이 매끄러운 연결된 선이 나온다.

![nonlin7](https://user-images.githubusercontent.com/31824102/36486453-6a70c4a8-1716-11e8-8295-6d69ee02806c.PNG)

각각의 제약은, 자유도(degree of freedom)을 하나 잃게 된다는 것을 의미한다(!) 따라서 맨 위의 piecewise cubic에서는 자유도가 8(추정하고자 하는 회귀계수의 수)이었지만, 3개의 제약(knots에서 1.continous여야 한다, 2. 1차미분 가능하다, 3. 2차미분 가능하다.)으로 인해, Cubic Spline의 함수는 자유도가 5이다. 해당 경우 K개의 knot에 따른 자유도는 (4+K)일 것이다. ($$\because 4*(K+1)-3K$$)

> $$d$$차 적합을 하며 $$K$$개의 knots가 있는 regressoin spline의 경우 $$(d+1)*(K+1)-(d)K$$의 자유도를 갖는다 보면 된다. K개의 knot가 있으니 구간은 K+1개로 나뉘고, 각 knot마다 '연속 + d-1차까지 미분가능' 이라는 제약이 들어간다.

위의 결과를 일반화하면, d차의 spline적합은 **1)** 각 piecewise에서 d차 다항적합을 하고, **2)** 각 knot에서 **(d-1)차까지의 미분이 가능**, 즉 매끄럽게 연속적이어야 한다는 제약이 붙는다.

### 7.4-3 The Splines Basis Representation

그럼, 대체 어떻게 d차 적합을 하면서도 d-1차까지의 미분이 가능하다는 제약을 달성할 수 있을까?

이는 앞에서 다루었던 Basis function의 형태를 통해 알 수 있다. K개의 knots가 있는 cubic spline도 역시 적절한  $$b_1,..,b_{K+3}$$가 선택된다면 다음과 같이 basis function의 형태로 표현할 수 있다.
$$
y_i=\beta0+\beta_1b_1(x_i)+\beta_2b_2(x_i)+..+\beta_{K+3}b_{K+3}(x_i)+\epsilon_i
$$
그럼 cubic(3차) spline을 어떻게 저렇게 표현할 수 있을까? 가장 대표적이고 직접적인 방법으로는 $$x,x^2,x^3$$만을 가지고 있는 삼차 다항식에서 각 knot마다 truncated power basis function을 추가하는 것이다.

truncated power basis function(절단 멱 기저함수...그냥 영어로 표현하자)는 다음과 같이 정의된다.

![nonlin8](https://user-images.githubusercontent.com/31824102/36486452-6a3dbbda-1716-11e8-849c-932761aa4ee1.PNG)

$$\xi$$는 knot를 의미한다. 즉 $$\xi$$에서 2차 미분('d-1'차 미분) 까지 미분이 가능함을 의미한다. 따라서 K개의 knots가 있을때 이를 다음과 같이 표현하게 된다. 조금 전에 나왔던 식과 비교해서 이해하면 된다.
$$
Y=\beta0+\beta_1X+\beta_2X^2+\beta_3X^3+\beta_4h(X,\xi_1)+..+\beta_{K+3}h(X,\xi_K)+\epsilon_i
$$
> 사실 truncated power basis function의 형태를 보면 알수 있지만 단순히 knot뒤에 새로운 식(여기서는 $$\beta(x-\xi)^3$$)이 점차 추가되는 형태이다. 

따라서 K개의 knots가 있을때의 cubic spline적합은,  $$X,X^2,X^3,h(X,\xi_1),...,h(X,\xi_K)$$의 **K+3개의 예측변수를 가지고 least square적합을 한 것**이 된다. 물론 절편항도 필요하니(위의 식에서 $$\beta_0$$부터 시작한거) 총 **K+4개의 회귀계수를 추정**해야 한다. 이러한 이유로 K개의 knots가 있는 cubic spline에서 자유도가 K+4개 인것이다.

> least square적합에 대해 좀더 설명해보자면, $$X,X^2,X^3,h(X,\xi_1),...,h(X,\xi_K)$$는 모두 정했으니 이제 $$\beta_j$$들의 수치를 정하면 되는데, 이는 
> $$
> \sum_{i=1}^n(y_i-\beta_0-\sum_{j=1}^{K+3}\beta_jb_j(x))^2
> $$
> 이렇게 구한다는 의미이다. [참고](http://www.stat.cmu.edu/~ryantibs/advmethods/notes/smoothspline.pdf)

그러나 이렇게 구한 cubic spline은 양 끝단 에서 예측의 신뢰구간이 넓어지게 된다. 즉, 예측의 정확도가 떨어지는 것이다. 사실 양끝단의 경우 모델의 분산이 커지는 것은 어느 모델이든 통용되지만, 이 경우 non-linear적합을 하고자 하였으므로 그 분산이 더 커지게 된다.(flexibility가 클수록 Variance도 크다는것 상기). 다항적합의 경우 양끝단에서 모형이 지나치게 급변하는 것은 고질적인 문제이다. 따라서 데이터의 양끝단, 즉 가장 왼쪽의 knot보다도 왼쪽에 있는 데이터와, 가장 오른쪽끝쪽에 있는 knot보다 오른쪽에 있는 데이터의 경우 **선형 적합**을 하여 이러한 문제를 완화하고자 하기도 한다. 이를 **Natural Cubic Spiline**이라 부른다. 아래의 그림을 보면, natural cubic spline의 경우가 양끝단에서 신뢰구간의 범위가 비교적 더 좁아졌음을 알 수 있다.

![nonlin9](https://user-images.githubusercontent.com/31824102/36486451-6a084090-1716-11e8-9a75-7bd49379e59f.PNG)

이 경우 양 끝단에선 3차적합이 아닌 1차적합을 하기에, 각각 자유도 2씩을 잃어 K+4-2X2, 즉 K의 자유도를 갖게 된다.

### 7.4-4 Choosing the Number and Location of the Knots

그렇다면 가장 중요한 문제인, 어디에, 또 몇개의 knot를 설정해야 하는지의 문제가 남았다. knot를 무수히 많이 설정하면 모델이 지나치게 flexible해지는 것은 자명한 일이다. 따라서 데이터가 빠르게 변동하는, 그래서 flexible한 형태가 필요해 보이는 지점에선 많은 knot를 배정하고, 상대적으로 안정되보이는 지점에선 적은수의 knot를 지정하는 방식을 사용할 수 있을 것이다. 

해당 방법 역시 좋은 접근이 될 수 있지만, 예측 변수가 많아질 수록 어느부분에서 flexible한 함수가 필요할지 직관적으로 알기 힘들때가 많다. 따라서 좀더 알고리즘적으로 적용될 수 있는 방식을 알아보자. 적정한 degrees of freedom을 설정하고 그에 따라 **균등한 qunatile**에 knot를 배정하는 것이다. 다음 그림은, 자유도5를 지정한 경우의 그림이다.

![nonlin10](https://user-images.githubusercontent.com/31824102/36486450-69cc894c-1716-11e8-8a56-0e81c3dac086.PNG)

자유도 4를 지정한 경우 natural cubic의 knot는 5개이다. 따라서 boundary knot 2개를 제외하고, 3개의 knot들이 quantile에 따라 자동으로 생긴것을 볼 수 있다. 즉, 25th, 50th, 75th quantile들이 knot로써 배정된 것이다. 이렇게 함으로써, 자료들이 많은 부분에선 많은 knot들을 배정하여 그 자료들이 잠재적으로 가질수 있는 non-linear형태를 반영하고, 자료들이 많지 않은 부분에선 적은 수의 knot들을 배정하려는 전략을 어느정도 달성하게 된다. [참고](https://stats.stackexchange.com/questions/7316/setting-knots-in-natural-cubic-splines-in-r)

> (이때의 자유도 4는 intercept를 제외한 것이라, 이전에 다뤘던 개념대로면 자유도5이다.)
>
> (또한, boundary knot는 우리가 관심있는 데이터의 범위를 지정해주기 위한 것을, 논의되지 않았으나, 여기선 df와 knot수의 관계보다는 df를 지정해주어 해당되는 knot수를 quantile에 맞추어 자동배정해준다는 사실에 주목하자.)

그럼 결국 몇개의 knot들을 설정할지, 즉 자유도를 몇으로 설정할지를 정해야 하는데, 이는 역시나 앞에서 나왔던 Cross-validation으로 결정한다. CV를 통해 본 CV error값이 가장 낮은 자유도의 갯수를 사용하는 것이다.

### 7.4-5 Comparison to Polynomial Regression

많은 경우 regression spline이 다항회귀보다 좋은 결과를 가져오는데, flexible하기 위해선 지나치게 차수를 높여가야 하는 반면에($$X^{15}$$차까지 있는 다항회귀를 생각해보자) regression spline은 차수는 유지하면서 knot의 수를 늘림으로써 flexibility를 늘릴 수 있기 때문이다. 따라서 대부분의 경우 regressoin spline이 더욱 **안정된 추정**을 가능하게 한다. 다음은natural cubic spline과 15차항 적합을 한 다항회귀를 비교한 그림이다. 15차항 적합은 지나친 flexibility로 인해 끝쪽에서 회귀선이 비합리적으로 요동치는 것을 볼 수 있다.

![nonlin11](https://user-images.githubusercontent.com/31824102/36486449-6991b966-1716-11e8-9be7-9deee4766b52.PNG)

## 7.5 Smoothing Splines

### 7.5-1 An Overview of Smoothing Splines

smoothing spline은 좀더 근본적인 개념에서 접근을 한 방식이다. 우리가 원하는 함수는, 관측된 데이터에 잘 맞는, 즉 $$RSS=\sum_{i=1}^n(y_i-g(x_i))^2$$를 최소화 하는 함수 $$g(x_i)$$일 것이다. 그러나 만약 함수g에 아무런 제약이 없다면, RSS를 최소화하자는 목표만으로는 모든 관측된 데이터 완벽하게 적합하는 n-1차 식이 만들어져 버릴 것이다.(선형회귀와 헷갈리지 말자. 선형회귀는 함수g의 형태를 $$(\beta_0+\beta_1x_i)$$로 제한하였다.) 따라서 우리는, **1)** RSS를 작게 만들면서도 **2)** 어느정도의 smooth를 가지는 함수 $$g(x_i)$$를 찾는 것을 목표로 해야한다. 

함수 g의 smooth를 보장하는 방식은 여러개가 있지만, 대표적으로 다음의 식을 최소화함으로써 찾는 방법이 있다.
$$
\sum_{i=1}^n(y_i-g(x_i))^2+\lambda \int g''(t)^2dt
$$
위 식의 앞부분은 우리가 아는 RSS이며 이때 $$\lambda$$는 양수인 하이퍼파라미터이다. 위 식을 최소화하도록 하는 함수g가 바로 smoothing spline인 것이다.

식을 보면서 느낌이 왔겠지만, 이는 **"Loss+Penalty"**의 형태를 가진 식이다. 6장의 Ridge와 Lasso에서도 이미 한번 접하였다. **데이터에 잘 적합하도록 하는 Loss**와, **지나치게 변동이 크지 않도록 하는 Penalty**가 합해진 것이다.

그럼 왜 $$g''(t)^2$$일까? 일차도함수의 경우 t에서의 기울기를 의미하고, 이차도함수의 경우 그 기울기의 변화를 의미한다. 따라서 2차도함수는 얼마나 급격하게 변하는지(roughness)를 의미하는 것으로 받아들일 수 있는 것이다. (직선의 경우 2차도함수값은 0이라는 것이 위의 내용을 이해하는데 도움을 준다.) 이차도함수값의 '절대값'이 크다면(기울기가 급격하게 줄어드는것도 roughness니까, '-'도 고려해야 한다) 이는 t지점에서 함수가 꾸불꾸불하다는 것을 의미한다. 따라서 이차도함수에 **제곱**을 해준다.

이를 모든 t의 범위에 대해 $$\int$$를 해줌으로써, 전체 t의 범위에서의 $$g'(t)$$의 변화량을 의미할 수 있다. 즉 식 $$\lambda \int g''(t)^2dt$$은 함수g를 smooth하게 되도록 만드는 역할을 한다. 당연히 $$\lambda$$값이 커질수록 g는 더욱 smooth하게 될것이다.

$$\lambda$$가 0이라면 모든 데이터를 지나가는 함수g가 만들어질것이고, $$\lambda$$가 무한히 크다면 g는 굴곡이 전혀 없는, 즉 선형이 될것이고, 이 경우 g는 **선형회귀와 같아진다**. 또 적당한 $$\lambda$$값에 대해서는 적당히 train data에도 적합되면서 적당한 smooth정도를 가지고 있는 함수가 만들어질 것이다. 따라서 $$\lambda$$는 bias-variance trade-off의 정도를 조절하는 값이라 볼 수 있다.

사실, 해당 값을 최소화하는 함수는 $$x_1,..,x_n​$$에서 2차미분이 가능한 knot를 갖으며 양 끝단에선 linear한, 즉 natural cubic spline과 같게 된다.(받아들이자) 위의 조건을 가진 natural cubic spline과 완전히 값까지 똑같은 함수가 나오는 것이 아니고, 이의 **shrink된 버전의 함수**가 나온다. 얼마나 shrinkage될 것인지를 $$\lambda​$$로써 정하는 것이다. 즉, Smoothing spline은 **모든 input에 대해 knot를 가지고 있는 natural cubic spline**을 **shrinkage penalty**를 hyperparameter $$\lambda​$$를 통해 조절해서 variance를 줄여주는 모델로도 볼 수 있는 것이다.(!)

### 7.5-2 Choosing the Smoothing Paramter $\lambda$

smoothing spline이 '모든' $$x_i$$에서 knots를 갖는 natural cubic spline이라면, 이는 너무 큰 degrees of freedom, 즉 너무 큰 flexibility를 갖는 것이 아닐까? 그러나 여기서 새로운 개념인 effective degrees of freedom, 즉 실질적인 자유도가 등장한다. smoothing spline에서는 $$\lambda$$가 smooth 정도를 조절해주므로, $$\lambda$$에 따라 실질적인 **effective degrees of freedom**이 변화한다. 구체적으로, $$\lambda$$가 0에서 $$\infty$$까지 변할때 effective degrees of freedom($$df_\lambda$$로 표현한다.)은 n에서 2까지 변한다.

#### 갑자기 왠 effective degrees of freedom?

왜 기존의 자유도가 아닌 effective degrees of freedom을 사용하는것일까?

기존의 자유도는 자유로운 parameter의 갯수로써, 계수들의 수를 의미했었다. smoothing spline은 n개의 parameter를 가지고 있으므로 명목상 n의 자유도를 가지고 있지만, 이 n개의 parameter들이 이미 shrink되도록 **constrain**을 받고 있다. 따라서 같은 n개의 parameter이더라도 shink의 정도에 따라 flexibility가 다르기에, 이를 따로 표현해주고자 하는 것이다.

effective degrees of freedom은 어떻게 구할까? 이는 선형대수적인 개념이 들어가게 된다. 간략하겐 다음과 같은 과정을 통해 구할 수 있다.
$$
\hat g_\lambda=\boldsymbol S_\lambda \boldsymbol y
$$
여기서 $$\hat g$$는 $$x_1,..,x_n$$까지에 대해 적합된 **n개의 적합값**이다.  $$\boldsymbol S_\lambda$$는 y를 적합값으로 표현하는 n X n의 matrix이다.  이때, $$df_\lambda=tr(\boldsymbol S_\lambda)$$이다. 즉 대각원소들의 합이다.

> matrix notation으로 표현하면 선형회귀에서 처럼 적합값을 y에 대한 식으로 나타낼 수 있는데, 이때
> $$
> \hat \beta=(G^TG+\lambda\Omega)^{-1}G^Ty
> $$
> 로 나타낼 수 있다. 따라서 적합값은 $$g(x)^T\hat \beta=g(x)^T(G^TG+\lambda\Omega)^{-1}G^Ty$$로 나타낼 수 잇고, 이를 정리하여 $$S_{\lambda}$$라고 notation한 것이라 볼 수 있다. 보다 자세한 설명은 [참고](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/splines.pdf)의 54쪽.

#### $\lambda$를 정해보자

regression spline가 다르게 smoothing spline은 모든 관측치 $$x_i$$에 knot를 두기에 knot의 갯수나 위치를 지정할 문제가 사라진다. 그러나, 여기선 shrink의 정도인  $$\lambda$$를 몇으로 할것인지의 문제가 남아있다. 이는 역시나, Cross-validation을 통해 구할 수 있다. 놀랍게도, smoothing spline에서 LOOCV는 다음과 같은 한번의 적합을 통해서 구할 수 있다. 식의 맨 오른쪽 등식을 보자.

![nonlin12](https://user-images.githubusercontent.com/31824102/36486448-695c0e2e-1716-11e8-8e66-47d147911bce.PNG)

이는  마치 least square에서 LOOCV가 leverage statistics를 통해 한번에 구할 수 있던 식과 매우 유사하다. least square에서의 LOOCV식 상기

![nonlin13](https://user-images.githubusercontent.com/31824102/36486446-691fc89c-1716-11e8-9626-7111fccd111f.PNG)

참고로 위 식은 least square에서 통용될 수 있기에, 앞서 다루었던 regression spline이나 least square적합을 이용하는 다른 basis function의 LOOCV에도 사용될 수 있다.

다시 smoothing spline으로 돌아와, LOOCV를 통해 적정한 $$\lambda$$값을 결정할 수 있다. ![nnonlin1](https://user-images.githubusercontent.com/31824102/36486445-68adb23e-1716-11e8-88eb-6949dedc59ab.PNG)

위 그림은 임의로 지정한 큰effective degrees of freedom, 즉 df=16의 선과 LOOCV를 통해 정한 선의 차이를 보여주고 있다.(smoothing spline에서는 degrees of freedom이 아닌 effeictive degrees of freedom을 쓴다! 기존의 df는 어차피 둘다 같다.) 이때, LOOCV를 통해 결정된 6.8df가 거의 차이가 없으면서도 덜 꼬불꼬불한 선을 나타냄을 알 수 있다.

## 7.6 Local Regression

Local regression은 말그대로 '지역적인' regression이다. 즉, 각 특정 target point $$x_0$$에서 그 **근처의 관측자료**들만을 토대로 적합을 시켜 flexible한 적합을 하고자 하는 접근방식이다.

특정 target point $$x_0$$에서 Local Regression의 대표적인 적합방식은 다음과 같다.

1. 전체 데이터중 $$x_0$$에서 가까운 k개의 자료 $$x_i$$들을 모은다. ($$s=k/n$$으로 전체의 $$s$$%만큼 모은다고 표현하기도 한다.)

2. 가까운 k개의 자료(nearest neighbor)들만이 가중치를 갖고, 나머지는 가중치가 0이 되도록 가중치 $$K_{i0}=K(x_i,x_0)$$을 설정한다. 

3. 앞서 언급한 가중치를 곱해주는, weighted least square regression을 한다. 이를 식으로 나타내면 다음의 식을 최소화하는 $$\hat \beta_0$$과 $$\hat \beta_1$$을 구하는 것이다.
   $$
   \sum_{i=1}^nK_{i0}(y_i-\beta_0-\beta_1x_i)^2
   $$

4. 최종적으로 $$x_0$$에서의 적합값은 $$\hat f(x_0)=\hat \beta_0+\hat \beta_1x_0$$이 된다.

이를 모든 point에 대해 반복하는 것이다. 다음의 그림은 Local regression을 통해 적합한 그림이다. 

![nnonlin2](https://user-images.githubusercontent.com/31824102/36486444-68763e58-1716-11e8-9082-f707b793c96f.PNG)

각각의 x수준에서의 적합값을 찾기위해, 전체의 s부분 만큼의 점들을 골라내고,(그림에서 빨간점으로 표시) 이들을 가까운 순서대로 weight를 부여해서, (그림에서 노란색 종모양 그림으로 표시) 해당 x수준에서의 적합값을 만드는 것이다. 그림에선 x=0.05와 x=0.45의 두 값만을 계산하였지만, 이를 가능한 모든 x수준에 대해 계산하는 것이다. 각 적합값을 구할때마다 traing data가 필요하기에, memory based 방식이라고도 지칭된다.

이때 위의 알고리즘에서, 2번에서의 가중치 $$K_{i0}$$는 각 $$x_0$$에 따라 달라질 수도 있고, 3번에서의 least square식이 linear regressoin식이 아니라 다항식일수도 있는 등 여러 변화를 줄 수 있다. 그러나 이중 가장 중요한 차이를 불러오는 변화는, 역시나 얼만큼의 nearest neighbor를 볼것인지, 즉 s가 몇%인지에 따른 변화이다. 작은 수준의 s는 전체 중 작은 부분만을 보는것, 즉 더욱더 지역적인 regression이 될것이고 더욱 꾸불꾸불한 선이 나올 것이다. 반대로 큰 s값은 거의 전체의 데이터를 다 보는 선을 만들어 낼것이다. 역시나 적절한 s는 Cross-validation을 통해 찾는다.

local regression은 여러 방식으로 사용되는데, 여러개의 변수가 있는 경우, 특히 다른 변수에서는 전역적이지만, 시간변수는 국소적인 경우 최근에 들어온 데이터를 처리할때 사용된다.(시간에 한해서만 local 적합을 하여 적합값을 만든다) 또한 여러 변수가 있을때는 단순히 다차원에서의 nearest neighbor를 모아서 그 데이터로 다중회귀를 하면된다. 그러나 3장의 Nearest-neighbor regression에서와 마찬가지로 차수가 **3,4차 이상으로 갔을때**는 $$x_0$$에 유의미하게 가까운 nearest neighbor를 찾기가 힘들기에 성능이 안좋아지게 된다.

## 7.7 Generalized Additive Models

지금까지는 예측변수가 하나인 경우의 non linear한 적합을 해보는, 단순 선형회귀의 확장을 다루었다. 그러나 해당 논의는 변수가 여러개일 때도 가능하다. 이번에는 변수가 여러개, 즉 다중선형회귀에 대한 확장을 다뤄보자.

Generalized Additive Models(GAM)은 기존의 선형모델에서 **가법성은 유지하면서도** 각 변수에 non-linear한 적합을 가능하게 하는 방법이다. 이는 기존의 선형모델이 그렇듯이 역시나 질적변수와 양저견수에 모두 적합이 가능하다. 우선 양적변수일때의 GAM에 대해 알아보자.

### 7.7-1 GAMs for Regression Problems

다항회귀를 확장하는 방식은 다음과 같은 원래의 식
$$
y_i=\beta_0+\beta_1x_{i1}+..+\beta_px_{ip}+\epsilon_i
$$
에서 단순한 선형결합 $$\beta_jx_{ij}$$가 아닌 비선형(그러나 동시에 smooth한) 함수 $$f_j(x_{ij})$$를 넣는 것이다. 즉 다음과 같이 변한다. 
$$
y_i=\beta_0+f_1(x_{i1})+..+f_p(x_{ip})+\epsilon_i
$$
($$\beta_j$$가 왜 사라졌을까 할 수 있지만, 함수형태이므로 상수배인 $$\beta$$ 역시 흡수할 수 있다.)

여전히, 각 변수 $$x_{ij}$$을 각각의 함수 $$f_j$$에 부과하고 있으므로, 가법적인 모형(additive model)은 유지하고 있다. (쉽게 말해 $$x_1x_2$$ 와 같은 항이 존재하지 않는다는 것이다.) 따라서 각각의 Y에 대한 각각의 예측변수들의 기여를 '더하여' 표현했다는 점에서 가법적인 모형이라 표현한다. 각 변수 $$x_{ij}$$에 대해 별도로 $$f_j$$가 계산되고, 이들의 기여를 더하는 것이다. 이때 $$f_j$$는 모수적으로 어떠한 형태를 지정하고 분석을 진행할 수도 있고, 비모수적으로 특정형태를 가정하지 않고 진행을 할 수도 있다.

GAM은 지금까지 1차원 변수에 대해 다룬 여러 방법들을 building block으로써 활용하여 가법적인 모델을 만들수 있게 해준다. 예를 들어 다음과 같은 식을 만들 수 있다.

![nnonlin3](https://user-images.githubusercontent.com/31824102/36486442-6813e230-1716-11e8-87e4-8d805c123973.PNG)

 여기서 year, age는 양적변수이고 education은 질적변수이다. 앞의 두 변수는 natural spline을 이용하여 적합을 하고, education은 더미화를 시켜 constant로 적합을 하였다. 그 결과는 다음과 같다.

![nnonlin4](https://user-images.githubusercontent.com/31824102/36486441-67db6478-1716-11e8-9d33-42c2c4f39ac0.PNG)

이 역시 적절한 basis function에서 basis represntation으로 나타낼 수 있는데, 이렇게 $$f_j$$을 basis function으로 표현이 가능한 형태로 설계한 경우 least square로 적합하여 값을 찾을 수 있다. 기존의 가법적인 선형모델과 마찬가지로 해석 또한 가능한데, 왼쪽 그래프는 우상향이므로, age와 education이 고정된 상태에서 year의 증가는 wage의 증가와 관계가 있다고 할 수 있다.

![nnonlin5](https://user-images.githubusercontent.com/31824102/36486440-677b4962-1716-11e8-92c0-7df0b1099d76.PNG)

해당 그림은 natural spline이 아닌 smoothing spline으로 적합한 그림인데, 이 경우는 least square적합을 할 수 없기에 좀더 까다롭다.(penalty term으로 인해 basis function으로 표현이 불가능하다) 이때는 다른 예측 변수들을 고정시킨 채 순서대로 변수들을 업데이트 시키는 backfitting이라는 방법을 사용한다. 이는 각 변수를 적합할때 해당 변수를 제외한 나머지 변수들의 partial residual에 적합을 통해서 가능하다. backfitting에 대한 자세한 설명은 [참고1](https://en.wikipedia.org/wiki/Backfitting_algorithm) [참고18쪽](https://web.stanford.edu/class/stats202/content/lec17.pdf).

> $$X_3$$에서의 partial residual이란 $$r_i=y_i-f_1(x_{i1})-f_2(x_{i2})$$이다. 만약 $$f_1$$과 $$f_2$$를 알고 있다면 남은 잔차, 즉 partial residual에 대해 $$X_3$$를 비선형회귀 적합을 하여 $$f_3$$을 구할 수 있을 것이다. 이를 각 $$f_j$$가 수렴할때까지 반복하여 업데이트한다.

위의 두 그림은 거의 차이가 없다. 사실 대부분의 경우, natural spline과 smoothing spline은 거의 결과차이가 없다. 또한, GAMs에서 위의 두 방식이 아닌 다른 방식을 사용해도 물론 된다.

#### Pros and Cons of GAMs

GAM의 장단점에 대해 요약해보자.

- 기존의 선형적합에서는 할 수 없던 비선형적합을 자동적으로 할 수 있게 한다. 고로 각 변수를 일일이 변형해줄 필요 없다. 
- 비선형적합인 만큼 예측력이 높다.
- 가법적인 모델이므로 다른 변수들이 고정되어 있을때의 한 변수의 영향을 알수도 있다. 고로 추론의 측면에서도 강점이 있다.
- $$X_j$$에 대한 함수, 즉 $$f_j$$의 smooth정도가 어느정도인지를 degrees of freedom으로 간단하게 나타낼 수 있다.
- 그러나 가법적인 모델이라는 점에서 중요한 상호작용을 잡아낼 수 없다는 단점이 있다. ([가법성이란](https://stats.stackexchange.com/questions/260175/what-does-the-additive-assumption-mean)) 그러나 선형회귀에서 가법성을 덜어내고 확장하였듯이, 인위적으로 interaction항을 넣어주어 $$f_{jk}(X_j,X_k)$$와 같은 상호작용 함수를 적합할 수도 있다. 그러나 이에 대한 적합벙법은 다루지 않겠다.

### 7.7-2 GAMs for Classificatino Problems

GAMs는 반응변수Y가 질적변수일때도 사용이 가능하다. 편의를 위해 Y가 0과 1의 값을 가진다 하자. 4장에서와 마찬가지로 특정 X가 주어졌을때 Y가 1일 확률을 $$p(X)=Pr(Y=1\lvert X)$$라고 표현할 것이다. Logistic regression에서 다음과 같은 식으로 모델을 설계했던 것을 상기하자.
$$
log({\frac{p(X)}{1-p(X)}})=\beta_0+\beta_1X_1+...+\beta_pX_p
$$
좌변의 '로짓', 즉 $$Pr(Y=0\lvert X)$$분의 $$Pr(Y=1\lvert X)$$인 오즈의 로그형태가 예측변수 X들과 선형결합의 관계를 가지고 있다. 이를 non-linear관계로 확장하는 자연스런 방법은 다음과 같을 것이다.
$$
log({\frac{p(X)}{1-p(X)}})=\beta_0+\beta_1f_1(X_1)+...+\beta_pf_p(X_p)
$$
이것의 logistic regression의 GAM버젼이다. 질적변수일때도 이전에 논의했던 GAM의 장점과 단점을 동일하게 갖는다.

예시로 나왔던 Wage 데이터에, 수입이 250넘을지로 다시한면 분류문제를 적합시켜보았다. 이때, GAM은 다음과 같은 적합이 될것이다.

![nnonlin6](https://user-images.githubusercontent.com/31824102/36486481-6d718bd8-1716-11e8-99c2-ae9f6b6c91d4.PNG)

이때 $$p(X)$$는 특정 year, age, education이 주어졌을때 wage>250일 확률, 즉 다음과 같다.

![nnonlin7](https://user-images.githubusercontent.com/31824102/36486480-6d3776b4-1716-11e8-9760-a0f1501874ea.PNG)

식을 보면 알 수 있지만 year의 경우 단순 선형적합을 하였다. 추가로, age의 $$f_2$$는 자유도5의 sommthing spline을, $$f_3$$은 setp function을 적용하였다. 각각의 적합결과는 다음과 같다.

![nnonlin8](https://user-images.githubusercontent.com/31824102/36486478-6cdb5122-1716-11e8-8b79-93d05e22e4bb.PNG)

additive model의 특성으로 역시나 각 변수들의 영향을 개별적으로 볼 수 있는데, 구체적으로는 year가 다른 변수들에 비해 income에 영향을 거의 미치지 않는다는 것을 볼 수 있다.

---

참고

전반적인 이론 : http://www.stat.cmu.edu/~ryantibs/advmethods/notes/smoothspline.pdf

이론2: https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/splines.pdf

backfitting참고 : https://web.stanford.edu/class/stats202/content/lec17.pdf