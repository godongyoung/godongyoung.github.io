---
layout: post
title: "[ISL] 4장 - 분류(로지스틱, LDA, QDA) 이해하기"
categories:
  - 머신러닝
tags:
  - An Introduction to Statistical Learning
  - Machine Learning
comment: true
---
{:toc}

이 장에서는 가장 많이 쓰이는 분류모델 3가지, **로지스틱 회귀, 선형 판별 분석, KNN**을공부한다.

> 여기 안에 들어간것은 개인적인 참고를 위한 지엽적인 부분입니다.



숫자형 변수를 예측했던 3장의 회귀문제와는 다르게, 질적 변수를 예측해야 하는 경우도 있다. 질적 변수(혹은 범주형 변수)를 예측하는 문제를 classification이라 부른다. classification 방법들은 보통 **각 범주에 속할 '확률'을 예측하는 형태로 분류**를 한다. (사실 이미 여기서 부터 선형회귀와 다르다)

이 전과 똑같이, $$(x_{1},y_{1}),...,(x_{n},y_{n})​$$의 주어진 데이터를 가지고 (여기서 y는 어느클래스에 속하는지, 즉 답.) 학습을 한 후 새로운 데이터 input $$x_{0}​$$이 주어졌을때 그 데이터가 어디에 속하는지를 분류하는 것이다. 예를 들면 환자의 현재 상태를 체크한 데이터들을 보고 이 환자의 증상이 무었인지를 분류하는 문제를 들 수 있다. 

이 경우 왜 회귀분석으로 접근할 수 는 없을까? 범주형 변수를 임의로(강제로) 숫자형 변수로 만들어주면 회귀분석을 '할 수는' 있다. 예를 들어 감기라면 $$Y=1$$, 폐렴이라면 $$Y=2$$, 독감이라면 $$Y=3$$ 과같은 식으로. 이런 식으로 회귀분석을 하면 값은 나온다. 그러나 이는 잘못된 접근이다. 앞장에서도 다루었지만 임의로 숫자를 부여하게 되면, 감기와 폐렴의 차이(위의 encoding에선 1)이 폐렴과 독감의 차이(1)와 같다는 의미가 되버린다. 

또 변수가 2개일때는 encoding, 즉 0과 1등의 숫자를 부여해서 마치 '1이 나올 확률'처럼 해석하는 시도가 어느정도 가능하지만(이때는 True, False의 역할을 하기 때문) 이 역시 아래 그림과 같이 회귀분석으로는 한계가 있다. 확률값으로 볼 수가 없음. (???뒤에서 나오는데 이는 LDA와 같다) (찾아보기!!!)

![class-linear](https://user-images.githubusercontent.com/31824102/35482733-48cf66e4-0431-11e8-9550-d9d310ef8782.PNG)

따라서 범주형 자료를 예측하기 위해선, **분류의 목적에 맞게 고안된** 다른 방법을 써야 한다.

## 4.1 로지스틱 회귀 (Logistic Regression)

로지스틱 회귀는 반응변수 Y가 '미납자가 될것인가 or 아닌가'와 같은 두개의 범주를 나눌때 주로 쓰인다. 그러나 로지스틱회귀는 Y를 (앞의 잘못된 회귀분석 예시에서 그랫듯이) 바로 계산의 대상으로 두지 않는다. 앞에서도 언급하였듯이, 분류문제에서는 **Y가 특정 카테고리에 속할 확률**을 목표로 두고 모델링을 한다.(bayes classifier를 상기하자) 위의 예시에서 독립변수(X)로 카드 부채(balance)가 주어졌을때 확률은 이렇다.
$$
Pr(미납자=yes|balance)
$$
줄여서 $$p(balance)$$로 표현하기도 한다(즉, $$p(X)$$). 특정 input에 대해 이 확률을 구한다면, 0.5이상일 경우 미납자로 치부해버리거나, 좀더 위험을 피하고자 하는 회사라면  이 확률이 0.1 이상일 경우 미납자로 판단하는 등의 결정을 내릴 수 있을 것이다.

#### 어떻게 이런 모델링을 할까

그럼 어떻게 'Y가 특정 카테고리에 속할 확률'을 목표로 두고 모델링을 할 수 있을까? 첫부분의 그림에서 보았듯이 $$p(X)=\beta_{0}+\beta_{1}X$$식의 단순한 선형 적합으로는 '확률'의 의미를 띌 수 없다. 그럼 낮은 balance의 input에 대해선 음수를, 반대의 경우 1을 넘는 확률을 반환해 버리기 때문이다. 넘는 확률들을 0과 1로 치부해버려도, 특정 input에 대해 확률이 1, 즉 100%라는 현실적이지 않은 결과가 나오게 된다. 직관적으로 이해 할 수 있지만 **binary classification(범주2개를 분류하는 문제)에서 선형적합은 한계가 있을 수 밖에** 없다.

이 문제를 boundary problem이라 하는데($$-\infty<x<\infty$$ 일때도 특정 boundary를 넘어가지 않는 함수를 만드는 것), 이를 해결하기 위한 많은 함수형태가 있지만, 로지스틱회귀에선 **로지스틱 함수**라 불리는 다음 식을 쓴다.
$$
p(X)=\frac {e^{\beta_{0}+\beta_{1}X}}{1+e^{\beta_{0}+\beta_{1}X}}
$$
(혹은 간략화하여 $$\frac {1}{1+e^{-f(x)}}$$라고 쓰기도 한다. 머신러닝에서는 이 식을 더 많이 보았을 것이다.)

![class-linear-log](https://user-images.githubusercontent.com/31824102/35482732-489fd2ee-0431-11e8-83ca-e14500caa76c.PNG)

위 식을 이용하면 다음과 같이, 0과 1에 가까워지지만 절대 넘지는 않는 '확률'의 특성에 딱 맞아 떨어지는 함수형태가 만들어 진다. 책의 예시에서 미납자의 비율은 3%였는데, 로지스틱 적합의 미납자 비율은 0.033으로 엄청 잘 맞았다.(물론 이는 과장된 가짜 데이터지만..) 위의 식의 계수들은 **maximum likelihood로** 구한다. (뒤에 설명)

#### 오즈의 등장

위의 식을 정리하면, 다음과 같은 식을 얻을 수 있다.
$$
\frac{p(X)}{1-p(X)}=e^{\beta_{0}+\beta_{1}X}
$$
여기서 $$\frac{p(X)}{1-p(X)}$$에 주목하자. 이는 $\frac{(해당클래스에 속할 확률)}{(속하지 않을 확률)}$        을 의미한다(!) 이를*odds*라고 부르는데, $$0\sim\infty$$값을 가질 수 있고 각각 **0일 수록** ($$p(X)$$가) 매우 낮은 확률, **$$\infty$$일수록** 매우 큰 확률을 의미한다.($$p(X)$$에 값을 몇개 넣어보면 바로 느낌온다.)

#### 로짓의 등장

이제 거의 다왔다. 최종적으로 우리에게 익숙한 선형회귀(느낌의) 식을 만들기 위해 양쪽에 log를 취해주자
$$
log(\frac{p(X)}{1-p(X)})=\beta_{0}+\beta_{1}X
$$
좌변을 '오즈'에 '로그'를 씌워줬다 하여 '*log-odds*' 혹은 '*logit*'이라고 부른다. 로지스틱 회귀 모델은, 바로 이 **'로짓'과 $$X$$가 선형관계에 있는 모델**이다(!). 

이때, 해석에 주의해야 한다. 앞장의 선형 회귀모델에서 $$\beta_{1}$$은 X가 한 단위 증가할때 Y의 변화량이었지만, 여기에선 **'로짓'의 변화량**이다. 바꿔 말하면, **'오즈', 즉 $$\frac{p(X)}{1-p(X)}$$가 $$e^{\beta_{1}}$$배 만큼 증가하는 것**이다 (!!) 이때 $$\beta_{1}$$이 양수면 X와 Y의 변동도 양의 관계, 음수면 변동도 음의 관계이다. ($$\beta_{1}$$의 부호에 따라 함수 개형이 위아래로 뒤집힌다.)

> '증가(혹은 변동)'에 대한 개념이 선형회귀의 $$Y_{i}+\beta_{1}=Y_{i+1}$$만큼 직관적이지 않다.
>
> $$\frac{p(X_{i})}{1-p(X_{i})}*e^{\beta_{1}}=\frac{p(X_{i+1})}{1-p(X_{i+1})}$$을 풀면 $$p(X)$$가 얼마나 변동했는지를 구할 수는 잇으나.. 식이 굉장히 지저분해져서, 그냥 딱 ''오즈가 $$e^\beta_{1}$$만큼 증가했다''로 받아들이는게 좋다. 즉 $$Y_i$$, 그러니까 $$p(X)$$가 $$X$$의 한 단위 증가에 변화하는 구체적인 량은 현재 $$X$$의 수준에 따라 다르다.

#### Maximum likelihood를 이용한 계수 추정

앞에서 살짝 얘기하였으나, 로지스틱회귀의 경우 mle로 추정을 한다.(least square로도 추정'할 수'는 있으나 mle의 성질이 여기서는 더 좋기에 mle로 추정한다.) Maximum likelihood는 많은 non-linear model을 적합하는데에 사용된다.

maximum likelihood란 간단하게 관측값들을 토대로 결정을 내리는 방법이라 보면 된다. 실제 함수의 parameter $$\theta$$를 추정할때, 해당 관측값들을 토대로 $$\theta$$가 취할 수 있는 여러 값 중 **'그 관측값을 만들어 냈을 가능성이 가장 큰 값'**을  $$\theta$$로 추정하는 방법이다. 

더욱 일차원 적으로 표현하자면, 미납자class의 데이터는 1을 많이, 미납자가 아닌 class는 0을 많이 반환하는 모델을 만드는것이다. binary classification의 경우 이 식은 다음과 같이 표현된다. 

![logi-mle](https://user-images.githubusercontent.com/31824102/35482731-48311638-0431-11e8-8d74-657a69d5876b.PNG)

이를 *likelihood function*이라 하는데, 이를 최대화 해주는 parameter $$\beta_{0},\beta_{1}$$을 찾는 것이다.(여기서 $${p(x_i)=Pr(Y_i=y_i|x_i)}$$)

> maximum likelihood는 주로 log를 씌워 log-likelihood로 계산을 한다. 그렇게 함으로써 곱이 합의 형태로 바뀌며 계산이 쉬워진다.(미분을 할 경우 이게 훨씬 더 쉽다) 위의 식에 log를 씌우고 '-'를 붙이면 그 유명한 NLL(Negative log likelihood)이 된다. 즉, 둘은 사실 같은 것이다. ([참고](https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80))

다음은 R을 통해 예시문제, 카드 부채(balance)와 미납자인지 아닌지에서 로지스틱을 적합한 결과이다.

![logi-table](https://user-images.githubusercontent.com/31824102/35482730-47f97ab6-0431-11e8-99da-28dddb6d97f7.PNG)

카드 부채의 계수가 0.0055이므로 balance가 1증가할때 미납자에 대한 '오즈'가 $$e^{0.0055}$$배 증가한다고 해석 할 수 있다. 또한 이때의 p-value는 선형회귀에서 t분포를 사용한것과 다르게 **z-분포**를 사용하였고, 마찬가지로 standard error를 통해 추정의 정확도 역시 구할 수 있다.

좀더 상세히 말하자면 $$\frac{b_{1}-0}{SE[b_{1}]}\sim Z$$이고 p-value는 전과 같게 $$H_{0}:\beta_{1}=0$$에 대한 검정에 사용된다.

실제 확률은 구해진 계수를 $$p(X)=\frac {e^{\beta_{0}+\beta_{1}X}}{1+e^{\beta_{0}+\beta_{1}X}}$$에 대입하고 X값을 넣으면 된다. 또 선형회귀에서 처럼, 질적변수는 더미변수(0,1)를 만들어서 같은 방식으로 적합하면 된다.

#### 다중 로지스틱 회귀와 confounding

이름이 더 어렵게 생겼지만, 선형회귀에서와 같이 변수가 한개가 아니라 여러개가 된것이다. 역시나, 식을 다음과 같이 확장만 하면 된다
$$
p(X)=\frac {e^{\beta_{0}+\beta_{1}X+...\beta_{p}X_{p}}}{1+e^{\beta_{0}+\beta_{1}X+...\beta_{p}X_{p}}}
$$
혹은 
$$
log(\frac{p(X)}{1-p(X)})=\beta_{0}+\beta_{1}X+...\beta_{p}X_{p}
$$
같은 방식으로 $$\beta_{0},\beta_{1},...,\beta_{p}$$를 추정하기 위해 maximum likelihood를 사용한다.

그러나 신기한 사실! 책의 예제에서 단순히 미납자~학생(인지 아닌지)으로 적합하였을때는 계수가 양수, 즉 '학생이면 미납자일 확률이 더 크다'로 나왔었는데, 학생($$X_{1}$$)과 부채($$X_{2}$$)로 적합하였더니 학생에 대한 계수가 음수, 즉 '학생이면 미납자일 확률이 더 작다'를 의미하는 결과가 나왔다. 심지어 p-value가 둘다 유의하게 나왔는데도!

이는, 다중회귀에서의 계수가 다른 변수들이 '고정'된 상태에서의 의미, 즉 동일한 부채(balance)에서 학생들이 미납자일 확률이 작다를 의미하기 때문이다. 이는 그림을 통해 알 수 있다.

![stud-logi](https://user-images.githubusercontent.com/31824102/35482729-47bfd608-0431-11e8-860b-9a13222d29a1.PNG)

왼쪽 그림에서 빨간 곡선은 부채에 따른 학생의 미납자 비율, 파란색은 비학생이다. 그리고 그림 아래부분의 빨간직선, 파란직선은 전체의 미납자 비율이다. **같은 balance 수준(X) 내에서는** 학생의 미납자 비율이 더 낮지만, **전체적으로 보았을때**는 학생의 미납자 비율이 더 높다. 왜 그럴까? 같은 부채의 정도에서는 학생이 미납자가 아닐 확률이 더 높지만, '부채가 높은 사람은 미납자일 확률이 크다->그런데 학생들은 부채가 높은 사람이 non-student에 비해 더 많다!(이는 오른쪽 그림을 통해서 알 수 있다)'. 이에 따라 balance 정보가 없으면 전체로 보아 학생이 더 미납자가 될 확률이 크다고 판단할 수 있지만 balance 정보가 있을 경우 학생이 미납자가 될 확률이 더 작다라고 판단할 수 있게 되는것이다. 이렇게, 변수 자체에 correlated된 관계가 있을 경우 하나의 변수만을 이용하는 것은 다른 결과를 가져 올 수 있다. 이를 *confounding*문제 라고 부른다.

추가로, 2개 이상의 범주에서의 로지스틱 회귀을 하는 것이 '가능'은 하다. 그러나 이 경우 성능이 다른 모델(바로 뒤에 나옴!)에 비해 안좋기에 실제에서 그렇게 자주 쓰이진 않는다. 

## 4.1 Linear Discriminant Analysis

직접적으로 $$P(Y=k|X=x)$$를 구했던 로지스틱 회귀와는 조금 다르게, LDA에서는 조금 덜 직접적인 방법을 쓴다. 각각의 Y가 주어졌을때의 X의 분포, **즉 $$P(X|Y)$$를  통하여 _'Bayes' theorem'_으로 $$P(Y=k|X=x)$$를 추정하고자** 한다. 이 때 X의 분포,  $$P(X|Y)$$에 정규가정이 생긴다면, 로지스틱 회귀와 매우 유사한 형태가 된다.

매우 유사한 형태가 된다면, 왜 굳이 로지스틱 회귀가 아닌 LDA를 사용할까? 다음의 이유가 있다.

- 2개 이상의 범주가 있을 시 성능이 더 뛰어나다
- 범주가 명확하게 구분되 있을 경우, 로지스틱 회귀는 굉장히 불안정한 결과를 낸다(계수가 쉽게 변동한다). LDA는 그렇지 않다(범주가 명확하게 구분되어 있다는 것은, 0,1의 분류에서 보통은 한 x 수준에서도 0인 data도 1인 data도 어느정도 겹쳐있기 마련인데 0인 x의 수준과 1인x의 수준이 극단적으로 나뉘어져 있는 경우를 말한다. [참고](https://stats.stackexchange.com/questions/254124/why-does-logistic-regression-become-unstable-when-classes-are-well-separated))
- 자료의 개수 n이 적을때에도 각 클래스에 대한 X의 분포가 정규분포와 유사하다면, 역시나 로지스틱 회귀보다 안정적인 성능을 보인다.

고로, 예상 가능했겟지만 상당한 경우에 LDA가 장점이 있기에 쓰인다.

### Using Bayes’ Theorem for Classification

$$\pi_k$$를 임의의 관측치가 k번째 클래스에서 왔을 확률이라고 하자. 쉽게 말해 주어진 자료 중 몇개의 라벨이 k클래스인 자료인지를 보면 된다. (즉, $$P(Y_k)$$를 의미한다. 이를 **사전확률**이라고도 한다.) 그리고 $$f_k(x)$$를 Y=k클래스 일때 특정 $$x$$가 나타나는 확률 이라고 하자. 즉, $$f_k(x)=P(X=x|Y=k)$$이다. 베이즈의 정리는 다음과 같은 형식으로 **사후 확률**을 구하는 방법이다.
$$
P(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum^{K}_{l=1}\pi_lf_l(x)}
$$
사실 이 책의 notation이 좀 독특한것이지 (notation 정리 : $$f_k(x)=P(X=x|Y=k)$$이고 $$\pi_k=P(Y_k)$$이다.)$$P(Y=k|X=x)=\frac{P(Y_k)P(X| Y_k)}{P(Y_1)P(X| Y_1)+P(Y_2)P(X| Y_2)+..+P(Y_K)P(X| Y_K)}=\frac{P(X\cap Y_k)}{P(X\cap Y_1)+P(X\cap Y_2)+..+P(X\cap Y_K)}$$와 같은 말이다. 따라서 $$P(Y=k|X=x)$$를 바로 예측하는 것이 아닌 $$P(Y_k)$$와 $$P(X=x|Y=k)$$를 추정하여 이로써 계산하는 것이다. 이 때 $$P(Y=k|X=x)$$, 즉 우리의 목표는 **사후확률**이라 부른다. 사실 $$P(Y_k)$$는 **전체 데이터에서 몇개가 k클래스 라벨을 가지고 있는지** 보면 추정완료이지만, $$f_k(x) =P(X=x|Y=k)$$는 추정하기가 좀 까다로워 **특정 분포를 가정을 하고** 분석을 진행한다. 2장의 Bayes Classifier에서 배웠듯이, $$P(Y=k|X=x)$$를 올바르게 추정할 수 있다면 오류율이 최소인 이상적인 분류를 할 수 있을 것이다. 따라서 이제, $$P(Y=k|X=x)$$를 추정하기 위해 필요한 $$f_k(x)$$를 	잘 추정할 수 있는 방법에 대해 다룰 것이다. 

### Linear Discriminant Analysis for p = 1

$$P(X=x|Y=k)=f_k(x)$$를 	추정하기 위해, 위에서도 언급하였듯이 **분포를 가정**한다. 가장 대표적으론 **정규가정**을 하는데, 수식으로 표현하자면 이렇다. 

![LDA1](https://user-images.githubusercontent.com/31824102/35482726-47330d54-0431-11e8-88d5-c1abe5b737a0.PNG)

즉, 각각의 평균과 분산 $$k=1,..,K$$인 K개의 각기 다른 정규분포가 있는 것이다. 추가로 모든 k에 대해 분산이 $$\sigma$$로 동일하다는 가정, 즉 등분산 가정을 한다. (선형회귀에서 이미 한번 접했다.) 이 경우 우리의 최종 목표는 다음과 같은 수식으로 변화할 수 있다. (어려워 보이지만 단순히 대입한 것이다)
$$
p_k(x)=P(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum^{K}_{l=1}\pi_lf_l(x)}
$$
![LDA2](https://user-images.githubusercontent.com/31824102/35482725-470538f2-0431-11e8-931b-bc42e685a2a9.PNG)

그 후 저 확률이 가장 큰 $$k$$클래스에게로 분류를 해주는 것이다(Bayes classifier 처럼). 또, 위의 확률 $$p_k(x)$$에 log를 씌우고, 어떤 k에 대해서든 변하지 않는 중복된 term을 없애면 결국 다음 수식이 가장 큰 클래스$$k$$를 고르는 문제로 귀결된다. (log는 1:1함수이므로 정보의 손실이 없다.)

![LDA3](https://user-images.githubusercontent.com/31824102/35482724-46d8c10a-0431-11e8-91ae-e85653bf41b8.PNG)

> $p_k(x)=\frac{\pi_k\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2\sigma^2}(x-\mu_k)^2)}{same_1}$
>
> $log(p_k(x))=log(\pi_k)-\frac{1}{2\sigma^2}x^2+\frac{1}{\sigma^2}x\mu_k-\frac{1}{2\sigma^2}\mu_k^2-same_1$ 이 중 $$-\frac{1}{2\sigma^2}x^2$$도 모든 $$k$$에 대해 동일하니까,
>
> $=\frac{x\mu_k}{\sigma^2}-\frac{1}{2\sigma^2}\mu_k^2+log(\pi_k)+same_2$

이 경우(정규가정과 등분산 가정) 공통분산$$\sigma^2$$와 각 클래스에 대한 parameter $$\mu_k$$를 알고 있다면, (정규 가정과 등분산 가정 하에서) 정확한 Bayes classifier를 계산으로 도출할 수 있다. LDA는, 바로 이 공통분산$$\sigma^2$$와 각 클래스에 대한 parameter $$\mu_k$$를 추정하여 Bayes classifier에 근사하고자 하는 것이다.

parameter는 다음과 같이 추정한다.

![LDA4](https://user-images.githubusercontent.com/31824102/35482723-46aa18c8-0431-11e8-8149-69ddb0380717.PNG)

간단하다. $$\hat \mu_k$$는 단순히 sample mean을 구한것이고, 공통 분산 $$\hat \sigma^2$$는 각 편차제곱에 자유도(K개의 $$\hat \mu$$를 사용하였으니 $$df=n-K$$)로 나누어 분산을 추정한 것이다.

> 이렇게 분산을 추정하는 것을 pooled variance라고 한다. k개의 $$\hat \mu$$를 사용하여 자유도를 K만큼 잃었으나 각 클래스의 평균이 모두 동일하다는 가정이 없으므로 overall variance를 구하지 못하고 해당 식을 쓰게 된다.

이제 $$f_k(x)$$는 추정이 끝났고(parameter를 다 구했으니), 이제 $$\pi_k$$를 추정해야한다. (notation 정리 : $$f_k(x)=P(X=x|Y=k)$$이고 $$\pi_k=P(Y_k)$$이다.)

앞에서도 언급하였지만 $$\pi_k$$는 다음과 같이 간단하게 추정될 수 있다.
$$
\hat \pi_k=\frac{n_k}{n}
$$
즉 (k클래스 데이터의 수/전체 데이터의 수)이다. 앞서 구한 여러 추정량들을 토대로, 우리의 최종 목적은 다음의 $$\hat \delta_k(x)$$가 최대가 되는 클래스k를 찾는것으로 귀결된다는. (단순히 추정한 값들로 바꿔 넣어 준 것이다.)
$$
\hat\delta_k(x)=x\frac{\hat \mu_k}{\hat \sigma^2}-\frac{\hat \mu_k^2}{2\hat \sigma^2}+log(\hat \pi_k)
$$
Linear Discriminant라는 명칭은 최종 function인 $$\hat \delta_k(x)$$가, 즉 최종 결정을 내리는 **decision boundary가 $$x$$에 대한 선형식**으로 나오기 때문이다. (훨씬 복잡한 식으로 나오는 다른 모델들이 많다.)

![LDA5](https://user-images.githubusercontent.com/31824102/35482722-467e93e2-0431-11e8-93c3-24d6c4d16d9a.PNG)

위와 같이 초록과 빨강 각각 20개의 simulated 된 데이터가 있다면, 각 클래스(초록, 빨강)에 대해 $$\hat \pi_k,\mu_k,\sigma$$를 추정하고 이를 통해 $$\hat \delta_k(x)$$가 큰 클래스로 분류를 하는 것이다. ($$p_k(x)=P(Y=k|X=x)$$의 추정확률이 제일 큰 클래스로 분류하는거랑 같은 말이다.)

해당 그림에서 실선은 decision boundary, 즉 $$\hat p_{초록}(x)=\hat p_{빨강}(x)$$인 경우의 경계선이다. simulated 데이터인 만큼 엄청난 데이터를 만들어 내어 구한 실제 boundary(Bayes classifier)가 점선으로 나와 있는데, simulated test data에 대하여 Bayes는 error rate가 10.6%, LDA는 11.1%로 상당한 성능을 보였음을 알 수 있다.

최종 정리하자면, LDA는 '$$f_k(x)=P(X=x|Y=k)$$에 대한 정규 가정과 각 클래스마다 다른 평균, 동일한 분산'을 가정하여 Bayes classifier의 확률을 추정하는 방법이다. 뒷장에서 '등분산'에 대한 가정을 떼어내는(즉 클래스마다 $$\sigma_k^2$$를 가질 수 있게 하는) 방법을 배울 것이다.

### Linear Discriminant Analysis for p >1

이제 한걸음 더 나아가, 변수가 여러개인 경우의 LDA에 대해 생각해보자. 기존에 $$P(X=x|Y=k)=f_k(x)$$엔 정규분포를 가정하였지만, 이젠 $$X=(X_1,..,X_p)$$가 다변량 정규분포(*multivariate Gaussian*)를 따른다고 가정한다. 이번에도, 각 class마다의 평균은 다르고, 분산은 공통의 분산을 가정한다.

#### 다변량 정규분포란? 

이름에서부터 부담스런 포스를 팍팍 풍기는데, 그 의미는 간단하다. 다변량 정규분포는 각각의 변수들이 1차원의(기존의 알고 있는) 정규분포 하나를 따르고 각각의 변수 pair가 어떠한 correlation를 가지고 관계를 갖는 분포이다. 쉽게말해 다차원 정규분포라고 받아들이면 편하다.

![LDA-multi](https://user-images.githubusercontent.com/31824102/35482721-464849f4-0431-11e8-9f9d-db0ad21c5c1e.PNG)

가장 간단하게 X가 2개인 경우부터 접근해보면, 위의 그림과 같은 그래프가 다변량 정규분포이다.(한 축에서만 바라볼 경우 우리가 아는 1차원의 정규분포가 나타남에 주목하라) 왼쪽그림은 가장 기본적인 그림으로, $$Var(X_1)=Var(X_2)$$, $$Cor(X_1,X_2)=0$$인 경우의 그림이다. 그러나 많은 경우 각 **변수들의 pair가 correlated되어 있는 경우**나 **분산이 같지 않은 경우**가 있는데, 그 경우 오른쪽 그림처럼 약간 길게 늘어진 형태의 그래프가 그려진다.

다변량 정규분포는 수식으로 나타낼 경우, p개의 정규분포를 일일이 쓰는 것이 아니라, 또 그들간의 correlation관계도 동시에 포함해주고자 다음과 같이 matrix로써 나타낸다.
$$
X\sim N(\boldsymbol \mu,\boldsymbol \sum)
$$
식에서 $$\boldsymbol \mu$$는 p개 변수의 분포의 평균을 나타낸 p차원 **vector**, $$\boldsymbol \sum$$는 p개 변수의 covariance를 나타낸 p*p크기의 **covariance matrix**이다.

> 2변량 정규분포일때 풀어써보면 다음과 같다.
> $$
> X\sim N({\begin{pmatrix}\ \mu_1\\ \mu_2 \end{pmatrix},\begin{pmatrix}\ \sigma_1^2&\rho\sigma_1\sigma_2 \\\rho\sigma_1\sigma_2& \sigma_2^2 \end{pmatrix}}), \rho\sigma_1\sigma_2=Cov(X_1,X_2)
> $$
>

이를 수식으로 나타내면 다음과 같다. 엄청 부담스러 보이지만, $$\boldsymbol \sum$$을 $$\sigma^2$$라고 생각한다면 기존에 알던 정규분포식의 생김새랑 크게 다르진 않다. 

![LDA-multi2](https://user-images.githubusercontent.com/31824102/35482720-461cfc04-0431-11e8-9953-c84e85aa60a7.PNG)

> 저기서 ||는 Determinant. 행렬을 root씌울 수 없으니 행렬은 이렇게 표현한다.

#### 다시 LDA

다시 돌아와서, 여러개의 변수를 포함한 LDA는 k클래스에 속한 관측치들의 분포 $$P(X=(x_1,..,x_p)|Y=k)=f_k(x)$$가 다변량 정규분포, 즉 $$N(\boldsymbol {\mu_k},\boldsymbol \sum)$$를 따른다고 가정한다. 이를 이용하여 다시 bayes 정리, 즉 $$P(Y=k|X=x)=\frac{\pi_kf_k(x)}{\sum^{K}_{l=1}\pi_lf_l(x)}$$를 통해 Bayes classifier의 확률을 구하면 결국 다음의 식을 최대화하는 k클래스를 고르는 문제로 귀결된다.

![LDA-multi3](https://user-images.githubusercontent.com/31824102/35482719-45f15b08-0431-11e8-9229-cb4941825931.PNG)

역시 단순LDA 경우의 matrix버전이라고 보면 된다. 이 경우 역시 결정에 관여하는 함수식 $$\delta_k(x)$$가 $$x$$의 선형함수이기에, LDA(Linear Discriminant Analysis)라고 부른다. 구체적인 예시로, 변수가 2개이고 클래스K가 총 3개일때 다음과 같이 결과가 나온다.

![LDA-multi4](https://user-images.githubusercontent.com/31824102/35482718-45c2a42a-0431-11e8-9db3-7c29f6a7705d.PNG)

바로 전 그림의 3차원 확률그림을 위에서 바라본것이라 이해하면 된다. 왼쪽그림은 클래스k가 $$k=1,2,3$$인 각 경우에 $$P(X=(x_1,x_2)|Y=k)=f_k(x)$$를 추정하여 boundary를 그은 것이다. 저기서 각 원들은 각 분포의 95%신뢰 구간이다. 오른쪽 그림은 실제분포에서 20개씩 관측치를 뽑아 만든 LDA와 Bayes classifier를 비교해본 그림이다.(다시한번, simulated data이기에 Bayes classifier를 구할 수 있다.) error rate가 각각 0.0746, 0.0770으로 비슷한 수준의 성능을 보였다.

#### null classifier의 함정(!)

그러나, 실제문제 예를 들면 책의 예시에서 default(미납자)를 예측하는 모델을 만들고자 할때 좀 더 흥미로운 문제가 발생한다. 10000개의 training data에 대해 적합한 LDA는 2.75%의 낮은 오류율을 보였다. 그러나, 모델을 적합시킬때, 다음과 같은 점을 고려해야 한다.

1. training data에만 지나치게 잘 적합되어 있는게 아닐까? 즉, overfitting된게 아닐까? 

보통, data의 갯수n에 비해 parameter의 갯수 p가 많아지면 overfitting의 우려가 생긴다. 그러나 지금은 10000개의 data에 대해 2개의 parameter를 추정하고자 하였기에, 여기에선 overfitting에 대한 걱정은 주 관심사가 아니다.

2. 전체 training data 중 default의 비율이 몇인가? 즉, 아무것도 하지 않은 null classifier에 비해 성능이 어느정도 좋은가?

여기선 이게, 진짜 문제다. 실제로 training data 중 default(미납자)의 비율은 3.33% 밖에 되지 않았다. (현실적으로 납득이 갈만한 분포) 즉, LDA의 train error 2.75%는 그 자체로 평가되어야 하는게 아니라, 위의 3.33%를 고려하여 평가해야 하는것이다. 아무것도 하지 않고 모두 non-default로 평가하는 null classifier라도 error rate 는 3.33%밖에 되지 않으니 말이다. 이를 더욱 상세하게 볼 수 있게 한 표가 다음에 나와 있다. 

![LDA-table](https://user-images.githubusercontent.com/31824102/35482717-458cfece-0431-11e8-8eb2-ff6cc075e7a1.PNG)

LDA가 default라고 평가한 104명중 81명이 실제 default라서 **default라 평가한 사람들에 한해서는** 잘한것 같다. 그러나, 전체 10000개의 data 중 non-default 9,667명 중에서는 9,664명을 non-default라고 옳게 평가하였지만, 실제 미납자 333명이 있는데, 그중 81명만을 default라고 제대로 평가하였다! 이는 무려 **default인 사람 중 252/333=75.7%를 놓친 것**이다. non-default를 잘못 잡아내는 것보다 default를 잘못 잡아내는 것이 주된 관심사일 credit-card 회사에게 이는 결코 좋은 결과가 아니다. 의학분야에서는 이를 sensitivity와 specificity로 명명하는데, **sensitivity**는 '목표 class를 제대로 잡아내었는지', (여기선 81/333=24.3%) **specificity**는 '반대 class는 제대로 잡아내었는지', (여기선 9,644/9,667=99.8%)이다. 

그렇다면, 왜 이런 low sensitivity가 발생하게 되었을까? 그 답은, 우리가 따라하려 목표로한 Bayes Classifier가 **'어떤 class를 틀리던간에 상관 없이'**, 전체 **_'total_ error rate를 줄이고자'** 목표하였기 때문이다. 쉽게 말해 non-default라고 평가하면 '주로' 맞으니까(train error가 적으니까), default라고 평가하는 것에 대해 매우 신중해진 것이다.

이러한 문제를 해결하기 위해, Bayes classifier는 약간의 수정을 한다. 구체적으로는 '역치'를 수정을 한다. 기존의 Bayes classifier는 해당 class에 속할 확률이 가장 크면 배정을 하는, 즉 default/ non-default의 경우 50%가 넘으면 배정을 하였다. (식으로는 $$P(default=yes|X=x)>0.5$$)그러나 위의 표에서 보았듯이, default는 실제 발생이 매우 드물다. 즉, 50%의 역치로는 이를 제대로 잡아주지 못하는 것이다. (실제로 default를 104명밖에 판결내리지 않았다.) 따라서, 이 역치를 줄여주는 것이다. 예를 들면 특정 input이 default일 확률이 0.2정도만 되어도 default로 평가하는 것이다.(식으로는 $$P(default=yes|X=x)>0.2$$) 역치 0.2의 LDA에 대한 결과는 다음과 같다.

![LDA-table2](https://user-images.githubusercontent.com/31824102/35482716-455c1a16-0431-11e8-9747-9feaa80b3239.PNG)

이제 sensitivity의 관점에서, 전체 333명 중 138명, 즉 41.4%를 놓치게 되어 75.7%였던 지난 모델보다 훨씬 나은 결과를 보여주었다. 물론 non-default를 그만큼 더 못 판별하여 전체 total error-rate는 3.73%으로 늘어나게 되었다. 그러나 default를 잘 잡아내고자 하는 credit-card회사의 관점에서는 sensitivity가 낮은 이 모델이 더욱 좋은 모델인 것이다.

default의 역치(threshold)와 non-default의 오류율이 반비례관계에 있다는 것을 눈치 챘을 것이다. 이는 그래프로 나타내면, 다음과 같다.

![LDA6](https://user-images.githubusercontent.com/31824102/35482713-44c7e6c0-0431-11e8-81e5-8f0028fd4ed0.PNG)

파란선이 default에 대한 오류율, 노란점선이 non-default에 대한 오류율, 검은선이 전체 오류율이다.

역치가 1/K(여기서는 1/2이므로 0.5)에서 0까지 변동할 동안, default에 대한 오류율은 계속 감소하지만 non-default에 대한 오류율은 계속 증가한다. 그럼, 어느정도 역치가 좋을까? 역시나 이것도 **그때그때 다르다**. 이 경우 default를 잘못 분류하는 것에 대한 위험비용을 계산을 고려하는 등의 도메인 지식이 요구될 것이다.

#### 비교를 위한 ROC curve

여러 threshold의 오류율을 비교하고자 할때, AUC(area under ROC curve)가 주로 이용된다. ![LDA7](https://user-images.githubusercontent.com/31824102/35482712-4495a6a6-0431-11e8-8533-5ac7dc6217e5.PNG)

위의 선은 각 threshold에 따른 sensitivity와 1-specificity이다. y축은 default를 제대로 판별한 경우, x축은 non-default를 default로 잘못 판별한 경우이므로 왼쪽위 모서리에 가까운 점이 좋은 threshold이고, 저 점에 가까운 ROC를 보이는 모델이 더 좋은 모델이라 할 수 있다. (이는 단순히 비교할때 이렇게 비교하는게  좋다는 것이다. 왜 여기서 나왔는지..)

### Quadratic Discriminant Analysis

Quadratic Discriminant Analysis , 즉 QDA는 이름에서 느껴지다시피 LDA와 약간 다르다. $$P(X=(x_1,..,x_p)|Y=k)=f_k(x)$$에 대하여 multivariate normal 분포를 가정하여 Bayes' theorem을 이용한 분류를 한다는 점에서는 똑같지만, 모든 클래스k에 대하여 **동일한 covariance matrix를 가정했던 LDA**와 달리 QDA는 k클래스 마다 **각각의 covariance matrix를** 가지게 한다. 

즉, $$X\sim N(\boldsymbol {\mu_k},\boldsymbol {\sum_k})$$를 갖게 하는 것이다. 훨씬 어려워 보이지만, 최종 결정 함수에서는 기존에는 모두 동일하여 제외되었던  $$-\frac{1}{2}x_k^T\sum^{-1}x_k$$가 추가되기만 하면 된다. 식으로 나타내면 다음과 같다.

![LDA8](https://user-images.githubusercontent.com/31824102/35482711-4464ee12-0431-11e8-9362-7fdbcb7a9126.PNG)

위의 식이 가장 크게 나오는 클래스k로 input을 분류해주는 것이다.  이 결정함수가 x에 대한 선형식이 아니기 때문에 QDA(quadratic)로 부른다.  자연스럽게 decision boundary도 LDA와는 다르게 비선형의 형태를 띄게 된다. 

왜 굳이, 분산이 동일하다 가정하는 LDA와 다르다 가정하는 QDA로 두개의 방법이 있을까? 답은 간단하게 bias-variance trade-off에 있다. p개의 예측변수가 있다면 그들의 covariance matrix를 추정하기 위해선 p(p+1)/2개의 parameter가 필요하다. (p개의 분산과 그들간의 covariance $$_pC_2$$개) 이를 k개의 클래스에 대해 다른 분산으로 추정하려면 K*p(p+1)/2개의 parameter를 추정해야 하는 것이다. 작은 수의 parameter를 추정하는 LDA의 경우는 훨씬 덜 flexible하고 variance가 적은 모델이 된다. 그러나, 기본 가정인 공통분산이 아닐 경우, (많은 경우 아닐 것이다. 마치 선형 가정처럼) 높은 bias를 갖게 될 것이다.

따라서, **training data 수가 적어**서 variance를 줄이는 것이 중요할 경우 **LDA를**, **데이터 수가 많아**서 variance(데이터 셋이 달라지는 것에 따라 모델이 변동하는것)에 대한 우려가 적을때, 혹은 공분산에 대한 가정이 비현실적으로 판단될 때에는 **QDA를** 사용한다.

### A Comparison of Classification Methods

지금까지 논의한 classification방법들, KNN, 로지스틱 회귀, LDA, QDA에 대해 비교해보자(!). 

#### LDA와 로지스틱

LDA와 로지스틱 회귀는, 접근 방식은 달랐지만 굉장히 유사한 밀접한 관계를 갖는다. 1개의 예측변수에 대해 2-클래스로 분류하는 문제를 보았을때, LDA로 로짓(log odds)를 계산해 보면 다음과 같다. 

![LDA9](https://user-images.githubusercontent.com/31824102/35482710-44395982-0431-11e8-9dab-44d8cea280bf.PNG)

> 엄밀히 말하자면 해당 식에서 $$c_0,c_1$$은 $$\mu_1,\mu_2,\sigma^2$$의 함수일 것이다.

로지스틱에서 로짓이 $$log(\frac{p_1}{1-p_1(x)})=\beta_0+\beta_1X$$로 X에 대한 1차식으로 나타내어 지는것과 같다. 물론 로지스틱은 계수를 maximum likelihood로, LDA는 정규가정하에서 mean 과 variance의 추정으로 계수를 구하였기에, 이 둘의 계수 자체가 일치하지는 않지만, 선형 decision boundary를 만들어 낸다는 점에서 둘은 유사한 관계를 갖는다. 이러한 관계는 변수가 여러개일 때에도 마찬가지로 유지된다. 결국 적합방법이 다른 같은 계열의 방법이라고 이해하는 것이 좋다. 성능은 물론 LDA의 가정(정규가정)이 어느정도 맞을 경우 LDA가 좋고, 틀릴 경우 로지스틱회귀가 좋게 된다.

> binary classification에서 LDA의 로짓을 구하면 다음과 같다.
>
> ![LDA10](https://user-images.githubusercontent.com/31824102/35482709-43f15330-0431-11e8-8c21-2d71cfc8997c.PNG)
>
> $\therefore \frac {p_1(x)}{p_2(x)}=\frac{\pi_1exp(-\frac{1}{2\sigma^2}(x-\mu_1)^2)}{\pi_2exp(-\frac{1}{2\sigma^2}(x-\mu_2)^2)}$
>
> $log( \frac {p_1(x)}{p_2(x)})=log(\frac{\pi_1}{\pi_2})+(x-\mu_1)^2-(x-\mu_2)^2=c_1x+log(c)+c$
>
> $=c_1x+c_0$

#### KNN

KNN은 가장 가까운 k개의 관측치를 보고 그들의 특성에 따라 분류를 하는 것으로, **비모수적 방법**이라고 볼 수 있다. 따라서 실제 decision boundary가 non-linear일 때 좋은 성능을 보일 수 있다. 그러나 어떤 변수가 중요했는지 등에 대한 해석력은 잃게 된다. 애초에 데이터의 현황을 파악하여 반영하는 것이기에. 

#### QDA

QDA는 KNN과 LDA, 로지스틱회귀를 합친 특성을 가지고 있다. 이는 KNN보다는 flexible하지 않지만 decision boundary에 non-linear가정을 하였기에 LDA보다 flexible하고, 분포를 가정하였기에 KNN과 다르게 비교적 적은 데이터에서도 잘 적합할 수 있다. (LDA에 비해선 많이 필요하다. parameter수가 더 많으니.)

#### 실험을 통한 성능비교

6개의 유형에 대한 시나리오 데이터를 가지고 비교해보았다. 3개는 Bayes decision boundary가 linear, 나머지는 non-linear이다. 각 시나리오 마다 100개의 random noise가 포함된 data를 만들고, 이에 대해 충분히 큰 test-set에 대해 각 모델의 test error를 비교해보았다. 이때, 예측변수(X)는 2개이다.  이를 모두 설명하는 것은 낭비이고, 각각의 시나리오에 대한 결과를 요약하면 다음과 같다.

시나리오 1: 평균이 다른 서로 uncorrelated된 normal분포. 이때는 LDA가 제일 잘했다. LDA의 가정사항이 normal이니까. KNN이나 QDA는 이 경우 지나친 flexible을 가졌기에 bias측면에서의 강점은 별로 부각되지 않으며(정규가정이 딱 부합한 경우니까) Variance가 커졌기에, 잘 못했다. 로지스틱도 boundary가 선형이라는 점에서 LDA보다 약간 못했다.

시나리오 2 : 평균이 다르고 correlation이 -0.5인 normal분포.(찌그러진 multivariate normal dist) 시나리오 1과 결과가 다르지 않았다

시나리오 3: 이번엔 normal이 아닌 t-분포에서 뽑은거. LDA의 가정이 무너졌기에 로지스틱보다 조금 못했다. 그러나 다른애들보단 잘했다.(정규분포와비슷하니까)

시나리오 4: 첫번째 클래스에 속하는 자료는 0.5 correlation의 normal dist, 두번째 클래스의 자료는 -0.5 correlation. 이 경우는 QDA의 가정을 만족하는 경우니, QDA가 다른 방법보다 잘했다. 

시나리오 5: 이차 decision boundary.($$X-1^2,X_2^2,X_1X_2$$가 들어간 로지스틱함수에서 반응변수를 뽑음) 이땐 예상대로 QDA가 제일 잘했고 그다음이 KNN-CV였다.

시나리오 6: 이차보다도 더욱 non-linear한 function에서 뽑음. 이땐 KNN-CV, QDA, linear method순. KNN-1은 더욱 못했다. (1은 너무나도 flexible, 즉 너무나도 쉽게 variable) 고로 KNN에서 k를 잘 뽑아야한다는걸 시사한다.

따라서, 어느상황에나 잘 적합되는 모델은 없다. 실제 decision boundary가 linear이면 LDA, 적당히 non-linear이면 QDA, 훨씬 non-linear(꼬불꾸불)하면 KNN. 이때 KNN의 경우 k를 몇으로 설정하느냐에 따라 결과는 천지차이인데, 다음 장(Cross-validation)에서 이를 다룰 것이다.

> 또한, 3장에서 다루었던것 처럼 linear-method인 LDA나 로지스틱에도 다항식, 즉 $$X^2,X^3$$등을 추가하여 non-linear한 결과를 낼 수 있다. 물론 이때는 bias는 줄것이나 Variance는 늘어날 것. 그리고 LDA에서 가능한 모든 2차항과 상호작용항을 추가 할 경우 이는 추정된 계수는 다르겠지만 식은 QDA와 같아진다.