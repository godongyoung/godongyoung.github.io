---
layout: post
title: "[데이터분석 정리] black box interpretation.  Partial dependency plot 개인적 정리"
categories:
  - 머신러닝
tags:
  - black box
  - Partial dependency plot
  - PDP
comment: true
---

### 서론. 모델의 해석에 관하여

우리가 아는 모델은 2가지 관점, 즉 예측력과 해석력 측면에서 접근할 수 있다. 이 중, 예측력이 높은 복잡한 모델일 수록 해석력은 떨어진다고 할 수 있다. 대표적으로 딥러닝 모델들은 해당 변수가 어떻게 결과값에 영향을 주었는지 파악하고 활용하기가 상당히 힘들어진다. 이러한 특성에서 흔히 예측력이 높으나 해석이 힘들 모델을 **black box**라고 부른다. 반면, 선형회귀 같이 전통적인 모델들은 예측성능은 비교적 떨어지지만, '**변수 $$X_1$$이 1 증가할때 결과값 $$y$$가 몇 증가한다**' 등 직관적인 해석이 가능하는 강점이 있다. 이에 따라 다양한 모델에(즉, 딥러닝이던 어떤 모델이던!) 범용적으로 적용이 가능하고 모델의 예측값과 input변수가 어떤 관계가 있었는지를 해석하려는 다양한 시도들이 있다. Partial dependence plot은 그중 대표적인 접근방법이다.

## Partial Depedence Plot (PDP)

### PDP란?

학습된 모델을 해석하는 방법중 하나. **1개 혹은 2개의 변수**에 대해서 사용하여, 해당 변수가 target과 (학습된 모델하에서) 선형관계인지, 단조증가관계인지 등을 **plot을 통해** 확인할 수 있다. 

### 수식과 함께 더 자세한 설명 

여러개의 피쳐들 $$x_1,..,x_p$$중 우리가 effect를 보고 싶은 피쳐를 $$x_s$$(selected), 그 외 선택되지 않은 피쳐를 $$x_c$$(complement)라고 하자. 

즉 $$X^T=(X_1,..,X_p)$$이고, $$X=X_S\cup X_c$$일것이다. 

>  왜 굳이 얘넬 나눌까? 지금 우리는 1개, 혹은 2개의 변수들에 대해 집중해서, 선택된 피쳐만의 함수로 만드는 중이다. 왜 1,2개의 변수에 집중할까? 변수를 1,2개로 집중할 경우 plot으로 그리기 수월해지고, 결과적으로 직관적으로 관계파악을 할 수 있게 된다. 선택된 변수 $$X_s$$와 선택되지 않은 변수 $$X_c$$간에 강한 상호작용이 없다면 이러한 접근은 충분히 좋은 근사를 보여줄 수 있다.
>
> 실제 분석에선 주로 feature importance를 통해 유의하게 나온 소수의 변수들이나 domain지식을 통해 중요할것이라 나온 변수들에 대해서 확인한다.)

이때 모델이 학습한 관계는 $$f(X)$$, 혹은 $$f(X_s,X_c)$$라고 하면

이중 우리의 관심 $$X_s$$에 집중하는 방법은 여러가지가 있을 수 있지만, 그중 partial dependece가 취한 방법은 다음과 같다.
$$
f_{X_s}(X_s)=E_{X_c}[f(X_s,X_c)]
$$
즉, 고정된 $$X_s$$에 대해 모든 $$X_c$$의 feature space 에서 $$f(X_s,X_c)$$를 mean을 취하는 것이다. 

좀더 정확히 Expectation을 풀어써보면 다음과 같다. (단순히 expectation의 성질에 따라 식을 전개한것)

$$E_{X_c}[f(X_s,X_c)]=\int f(X_s,X_c)f_{X_c}(x_c)dx_c$$, 혹은 $$E_{X_c}[f(X_s,X_c)]=\sum f(X_s,X_c)f_{X_c}(x_c)$$ 

(이때 $$f(X_s,X_c)$$는 '예측값을 내뱉는 함수'이고, $$f_{X_c}(x_c)$$는 해당 $$X_c$$가 발현될 prob_density를 나타내는 pdf임을 주의하자)

예를 들어, 몸무게와 키로 연봉을 예측하는 함수가 있다고 해보자. 즉 다음과 같다. 'f(몸무게,키)=연봉'. 이때 몸무게만의 영향을 보기 위해, 모든 모집단 데이터에 대해 키의 분포를 알 고 있다면, (**주어진 몸무게**, 키) 에 대한 모든 값들을 키의 pdf에 따라 평균 내는것. 만약 대한민국사람들의 키의 모집단 분포가 다음과 같고, 주어진 몸무게(예를들어 60)에서의 예측값이 다음과 같다면, 

| x                                        | 160  | 170  | 180  | 190  |
| ---------------------------------------- | ---- | ---- | ---- | ---- |
| p(키=x)                                  | 4/8  | 2/8  | 1/8  | 1/8  |
| f(주어진 몸무게60,키x)의 값 (연봉예측값) | 150  | 200  | 300  | 250  |

$$f(몸무게60)=150*4/8+200*2/8+300*1/8+250*1/8=193.75$$가 된다.

> 해당 예시는 discrete한 변수들에 대해서의 예시이고, continuous에 대해선 당근 $$\sum xf(x)$$가 아니라 $$\int xf(x)dx$$가 된다. 

그러나 우리는 실제 데이터 $$X_s,X_c$$들을 다 가지고 있지 않다. 또한 $$X_c$$가 어떤 분포를 가지고 나타나는지, 즉 $$f_{X_c}(x_c)$$도 알지 못한다. 그래서 실제로 우리가 $$f(X_s)$$를 예측할때는 **train data안에 있는 데이터들을 그대로 이용해서 $$\hat {f_{X_c}}(x_c)$$을 구하고**, 이에 대해서 mean을 취해 partial effect를 구한다. 말이 어려운데, 결국 n개의 train data의 X_c를 각각 넣어보고 1/n으로 mean을 취하는 것이다. 이를 식으로 나타내보자.

n개의 데이터가 있을 경우, 그들 데이터에 대해 $$X_c$$값인 $$x_{1c},..,x_{nc}$$가 있을 것이다. 이때, 추정된 partial dependence function은 다음과 같다.
$$
\hat f_{X_s}(X_s)=\frac{1}{n}\sum_i^nf(X_s,x_{ic})
$$
input으로 들어가는 $$(X_s,x_{ic})$$는 함수가 만들어 졌으면(즉 모델이 적합되었으면) 어차피 우리가 암거나 넣을 수 있는 input자리이다. 정해진 $$X_s$$에 대해서(예를들면 몸두게 60), 이 자리에 $$(X_s,x_{1c}),..,(X_s,x_{nc})$$를 넣고 1/n한게 $$\hat f_{X_s}(X_s)$$인것이다. 이를 $$X_s$$가 존재할 수 있는 parameter space에 대해 (예를들면 몸무게 50kg~130kg)쭉 반복하면 최종적으로 $$\hat f_{X_s}(X_s)$$를 추정하고, 그림을 그릴 수 있다. (앞서 언급했듯이 **plot으로 확인**을 해보는 것이므로 $$x_s$$는 주로 1개의 변수, 혹은 2개의 변수로 그 수가 정해진다. ) 이를 통해, $$X_s$$가 어떻게 변하는지에 따라 average prediction이 어떻게 변화하는지를 눈으로 확인할 수 있게 된다.

< 사용법 >

위의 알고리즘을 읽으면서 예상했다시피, 모든 $$X_s$$의 feature space 에 대해 $$X_c$$를 돌면서 예측을 해야하므로, 계산량이 엄청나다. 그래서 그런지 실제 구현도 그닥 예쁘게 되어 있는게 없다. (R은 랜포에만 있고, python은 xgb에만 있다.) python을 통해 그린 그림은 다음과 같다. ([사용법](https://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html)은 매우간단). 짜여진 효율적인 함수가 없으니, 필요할때 마다 내가 만들어 쓰자. 

![sphx_glr_plot_partial_dependence_0011](https://user-images.githubusercontent.com/31824102/55940980-3405b480-5c7c-11e9-97e3-4012a12620d3.png)

< 이상적인 사용 >

1,2개의 소수개의 변수에 대해서 관계를 확인하는 것이기에, 선택된 $$X_s$$가 $$f(X)$$에서 **dominate한 경우** 관계를 잘 approximate할 수 있다. (그렇기에 주로 **중요할것이라 생각되는 변수에 대해서** 하게 된다) 혹은 $$f(X)$$가 단순한 관계, 즉 low-order interaction을 가지고 있을때 판단하기 좋다. 즉, 단순히 nuisance feature $$X_c$$에 대해 simple mean을 취하는 것이기에, $$X_s$$가 $$X_c$$와 **강한 상관관계를 가지고 있지 않은 경우**에 이상적인 결과를 낼 수 있다.

<주의 할 점>

1. 전체의 train data에서 pdf $$f(X_c)$$를 추정하고 그것에 대해 테스트한것이기에, 실제 데이터들이 어떻게 분포되었는지, 즉 **feature distribution**에 대해선 고려를 하지 않았다. 예를들어 2차원 pdp로 키,몸무게~연봉을 보고자 할때 f(키=190,몸무게=21) 과 같이, 어린애의 몸무게도 test를 해보게 된다. (모델에 키=190,몸무게=21을 넣으면 예측값이 당근 반환되긴 할것이다. 그러나 이 예측값은 현실적으로 의미가 없고, 제대로 학습되엇을리도 없다. ) 즉, 실제 그 value들이 어떻게 분포되어있는지를 무시한것이고, 그림을 잘못해석 할 수있게 된다. 이를 위해 2차원 plot에서 **실제 데이터의 분포**(혹은 rug)도 확인해야한다. 
2. 또한, $$X_s$$와 $$X_c$$가 highly correlated되어 있다면, 잘못된 해석을 낳을 수 있다. 같은 예시에서 키=190인데 전체 몸무게의 distn으로 평가를 하여 몸무게=21의 확률도 현실보다 커지게 반영하는것. 이를 보완한것이 [Accumulated Local Effect plots](https://christophm.github.io/interpretable-ml-book/ale.html#ale) 이다.
3. 이는 pdp보다는 mean(평균)의 문제이기도 한데, mean으로 인해 미처 보지못하는 효과가 있을 수도 있다. 예를 들어 데이터의 상위 50%는 positive한값, 하위 50%는 negative한 값이 나오면, 평균으로 나오는 $$f(X_s)$$는 0에 가깝게 나와 효과가 없는것으로 보인다. 이처럼 '통합'으로 인해 생기는 문제를 보완하기 위한것이 [individual conditional expectation curves](https://christophm.github.io/interpretable-ml-book/ice.html#ice) 이다. 


덧. 아래의 내용은 이론적 측면에서의 주의점이다. 

> 이때 PDP를 통해 나오는 $$\hat f_{X_s}(X_s)$$는 $$\tilde f_{X_s}(X_s)=E_{X_c|x_s}[f(X_s,X_c)|x_s]$$가 아님에 주의하자. (구분하기 위해 물결표를 위해 그렷다.) 미묘한 차이인데, $$E_{X_c}[f(X_s,X_c)]=\sum f(X_s,X_c)f_{X_c}(x_c)$$ 여기서 곱해지는 pdf가 $$X_c$$만의 pdf $$f_{X_c}(x_c)$$인지, 아님 주어진 $$X_s$$에 대해 condition이 걸어진 conditional pdf $$f(X_c|x_s)$$인지의 차이이다. 
>
> 후자(Conditional pdf를 사용)의 경우 $$X_c$$의 영향력을 무시한 (marginalize out한) $$X_s$$의 함수라고 할 수 있고, 전자(PDP)의 경우 $$X_c$$의 effect를 (평균값으로라도) 고려해준  $$X_s$$의 함수라고 할 수 있다. 또한 후자의 경우,  $$f(X)$$와 $$X_s$$와의 관계뿐 아니라 $$X_s$$와 $$X_c$$의 **결합으로 생긴 관계도** 반영하게 되기 때문에, $$X_s$$가 결과값과 **크게 관련이 없는 변수들일 경우에도 strong effect가 잡힐 수** 있다고 한다. (ESL370p) 혹은 [논문](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451) 자세한 이해를 위해 예시를 첨부한다.

> 원문:  However, averaging over the conditional density , rather than the marginal density , causes $$\tilde f_{X_s}(X_s)$$ to reflect not only the dependence of $$f(X)$$ on the selected variable subset $$X_s$$, but in addition, apparent dependencies induced solely by the associations between $$X_s$$and the complement variables $$X_c$$.
>
> <예시 1>
>
> 예를들어 실제 목적함수가 만약 선택된 변수 $$X_s$$와 선택되지 않은 변수 $$X_c$$가 서로 가법적(더하기)관계로 얽혀있었다고 해보자. 즉, $$f(X)=h_1(X_s)+h_2(X_c)$$이다. 이경우 marginal pdf를 사용하는 PDP는 additive constant를 제외하고는, 실제 알고픈 함수 $$h_1(X_s)$$를 복원할 수 있다.
>
> 즉, $$E_{X_c}[f(X)]=E_{X_c}[f(X_s,X_c)]=\int (h_1(X_s)+h_2(X_c))f_{X_c}(x_c)dx_c$$
>
> $$=h_1(X_s)*\int f_{X_c}(x_c)dx_c+\int h_2(X_c)f_{X_c}(x_c)dx_c=h_1(X_s)+const$$
>
> 위의 constant는 $$X_s$$와 관계 없는 constant이기에, 모든 $$X_s$$에 대해 동일하고, 따라서 특정 $$X_s$$가 미치는 상대적 영향을 비교하는데에는 전혀 방해가 없다.
>
> 반면, conditional pdf를 이용하는 방법의 경우, $$h_1(X_s)$$를 복원하지 못한다. 
>
> 즉 $$E_{X_c|x_s}[f(X_s,X_c)|x_s]=\int (h_1(X_s)+h_2(X_c))f_{X_c|X_s}(x_c|X_s)dx_c$$
>
> $$=h_1(X_s)+\int h_2(X_c)f_{X_c|X_s}(x_c|X_s)dx_c$$
>
> 여기서 뒤의 항은 $$X_s$$에 따라 값이 달라지는 값이고 즉 ,constant가 아니다. 이에 따라 각 $$X_s$$마다 $$h_1(X_s)$$를 복원하지 못하게 되고, **뒤의 항때문에** 결과 해석에서 **의도치 않은 영향을 해석하게 될수도** 있다.
>
> <예시 2>
>
> 다음은 실제 목적함수가 만약 선택된 변수 $$X_s$$와 선택되지 않은 변수 $$X_c$$가 서로 승법적(곱하기)관계로 얽혀있었다고 해보자. 즉, $$f(X)=h_1(X_s)*h_2(X_c)$$이다. 이경우 marginal pdf를 사용하는 PDP는 multiplicative constant를 제외하고는, 실제 알고픈 함수 $$h_1(X_s)$$를 복원할 수 있다.
>
> 즉 $$E_{X_c}[f(X)]=E_{X_c}[f(X_s,X_c)]=\int (h_1(X_s)*h_2(X_c))f_{X_c}(x_c)dx_c$$
>
> $$=h_1(X_s)*\int h_2(X_c)f_{X_c}(x_c)dx_c=h_1(X_s)*const$$
>
> 이때도 역시 뒤의 constant는 모든 $$X_s$$와 관계없이 동일하게 곱해지기 때문에, $$X_s$$에 따른 변화를 잡아낼 수 있다.
>
> 반면, conditional pdf를 이용하는 방법의 경우, 이 예시에서도 $$h_1(X_s)$$를 복원하지 못한다. 
>
> 즉 $$E_{X_c|x_s}[f(X_s,X_c)|x_s]=\int (h_1(X_s)*h_2(X_c))f_{X_c|X_s}(x_c|X_s)dx_c$$
>
> $$=h_1(X_s)*\int h_2(X_c)f_{X_c|X_s}(x_c|X_s)dx_c$$
>
> 여기서 뒤의 항은 $$X_s$$에 따라 값이 달라지는 값이고 즉 ,constant가 아니다. 이에 따라 각 $$X_s$$마다 $$h_1(X_s)$$를 복원하지 못하게 되고, **뒤의 항때문에** 결과 해석에서 **의도치 않은 영향을 해석하게 될수도** 있다.

---

참조 :

원논문 : <https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451>

정리 잘하신분 : <https://christophm.github.io/interpretable-ml-book/pdp.html#fn27>

sklearn에서 사용 : <https://scikit-learn.org/stable/auto_examples/ensemble/plot_partial_dependence.html>
