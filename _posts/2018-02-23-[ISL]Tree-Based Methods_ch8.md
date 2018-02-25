---
layout: post
title: "[ISL] 8장 -Tree-Based Methods(Bagging, RF, Boosting)이해하기"
categories:
  - 머신러닝
tags:
  - An Introduction to Statistical Learning
  - Machine Learning
comment: true
---



{:toc}



드디어, regression과 classification을 위한 tree-based 방법을 다룬다. 이 방법은 예측변수의 전체 공간을 단순한 여러 영역으로 계층화(stratifying), 혹은 분할(segmenting)하는 방법이다. 후에 예측을 할때에는 해당 영역의 training date의 평균값이나 최빈값으로 값을 예측한다. 예측변수의 공간을 나누는 방식이 tree형식으로 나타내질 수 있기에 이러한 방법들을 decision tree방법이라 부른다.

tree-based 방법은 단순하고 설명력이 좋으나, 기존의 6,7장에서 다룬 supervised learning보다 예측력이 좋지 못하다. 따라서 이번 장에선 tree를 확장시킨 **bagging, random forests, boosting방법**을 소개한다.(!) 이들은 여러개의 tree를 만들어 이를 결합하는 방식인데, 이렇게 하여 예측정확도의 엄청난 상승을 거둘 수 있다.

## 8.1 The Basics of Decision Trees

Decision tree는 회귀나 분류문제 모두에 사용된다. 우선 회귀문제부터 다뤄보도록 하자.

### 8.1-1 Regression Trees

예시 데이터로, 야구선수들의 메이저리그경력(Years)과 지난 시즌 안타수(Hits)로 연봉(Salarys)를 예측하는 문제를 생각해보자. regression tree는 간단하게 다음과 같은 그림으로 작동을 한다.

![tree1](https://user-images.githubusercontent.com/31824102/36580558-a94bc34e-1860-11e8-9d46-31b0511711b9.PNG)

위에서 부터 몇개의 spliting rule에 따라 자료들을 나눈다. 예를 들어 메이저리그Year가 4.5년보다 적으면 왼쪽으로 보내는 식이다. 이때 [Years < 4.5]인 사람들의 salary의 예측값은 단순히 training data에서  [Years < 4.5]인 사람들의 평균값으로 예측한다. Year가 4.5년보다 큰 사람들은 오른쪽으로 보내져 2번째 갈림길인 Hits<117.5에 도달하여 나뉘어 진다. 

결국, 위의 tree모델은 전체 player를 3영역으로 나누게 되는 것이다. [Years < 4.5]인 player들, [Years > 4.5 & Hits < 117.5]인 player들, [Years > 4.5 & Hits > 117.5]인 player들. 이를 그림으로 표현하면 다음과 같다.

![tree2](https://user-images.githubusercontent.com/31824102/36580557-a90419ae-1860-11e8-8beb-26f10243e225.PNG)

각각의 영역의 예측값은 그곳에 속하는 data들의 salary평균이다. 이때 위의 나무그림을 기준으로, 최종적으로 나눠진 각각의 node(아래 그림에선 $$R_1,R_2,R_3$$)은 terminal nodes, 혹은 tree의 leaf라고 불리고, 그 전까지의 갈림길(node)들은 internal node라고 불린다. 

위의 결과를 통해, 메이저리그에서의 Salary는 Year에 큰 영향을 받는것을 알 수 있다. 즉, 경력이 별로 없는 선수라면 안타(Hits)를 많이 기록해도 Salary에 영향을 거의 미치지 못한 것이다. 이와 같이 Decision tree는 설명력의 측면에서 강점을 가지고 있다. 결과로 나온 나무 그림만으로도 시각적 설명이 가능하다.

#### Prediction via Stratification of Feature Space

위와 같은 regression tree는 다음 2단계를 거쳐 진행된다.

1. 모든 설명변수($$X_1,..,X_p$$)가 포함된 공간을 $$J$$개의 '겹치지 않는' 영역으로 분할한다.
2. 각 영역에서는 그곳에 속하는 training data의 평균을 통해 **일관된 예측값**을 반환한다.

그럼, 어떻게 $$J$$개의 영역을 쪼갤까? 영역은 다양한 형태로 쪼갤수 있겠지만, 해석의 편의를 위해 위의 그림과 같이 box형태로 쪼갠다. 구체적으로는, **RSS를 최소화하는 방향으로** box를 쪼갠다.

각 box($$R_1,..,R_J$$)에서의 예측으로 인한 잔차들을 줄여주는 것이다. 이는 식으로 나타내면 다음과 같다.
$$
\sum_{j=1}^J\sum_{i\in R_j}(y_i-\hat y_{R_J})^2
$$
그러나, 모든 가능한 box들의 조합을 구하고 RSS를 계산하는 것은, 불가능하다. (Year<2,Hits<102.01,..,같은 무한한 box조합이 존재할것이다. 변수가 많아질수록 더더욱.) 따라서 실제로는, top-down, greedy 방식을 사용한다. 즉, 위에서 부터 기준을 만들되, 기준을 만들때 앞일은 고려대상에서 제외하고, 당장의 RSS를 최소화하는 것을 목표로(greedy) 기준을 만드는 것이다. 이를 recursive binary splitting이라고도 부른다. (물론, 미래의 관점에서 이는 최선의 수가 아니었을 수도 있다. 그러나 이것이 실현가능한 수이다.)

첫번째 기준, 즉 아직 모든 data가 분할되지 않고 한 영역에 있을때를 생각해보자. 그때 우리는, 각각의 $$X_1,..,X_p$$에 대해서 분할점을 고려하고, 그 중 RSS를 최소화하는 특정 변수$$X_j$$의 분할점을 고를 수 있을 것이다. 첫번째 분할만을 고려한다는 점에서 이는 훨씬 수월한 작업이 된다. 이를 식으로 나타내면, 다음과 같다.

![tree3](https://user-images.githubusercontent.com/31824102/36580556-a8bbc154-1860-11e8-9706-3d75fda4d2d8.PNG)

특정 분할점 s를 찾는다. 어떤 기준으로?

![tree4](https://user-images.githubusercontent.com/31824102/36580555-a87d0ebe-1860-11e8-893f-2a19145310eb.PNG)

그렇게 분할하였을때 RSS가 최소인 분할점으로.

그 후 만들어진 2개의 영역 중 하나를 다시 RSS를 최소화하는 기준으로 recursive binary로 쪼개고, 이를 통해 만들어진 3개의 영역중 하나를 쪼개는 식이다. 이는 미리 설정해둔 stopping criterion을 충족할때까지 반복될 수 있다. stop criterion은 각 leaf안에 최소 5개의 data는 있어야 한다는 것 등이 있다.

최소 기준만을 가지고 적합을 한 큰 tree의 예시는 다음과 같다. 그림을 잘 보면 알겠지만, [Years < 4.5]가 나왔어도 같은 변수기준인 [Years < 3.5]가 아래에 다시 등장하기도 한다. 

![tree4.1](https://user-images.githubusercontent.com/31824102/36580563-aa4a1566-1860-11e8-935b-1c27a964762a.PNG)

#### Tree Pruning

위의 방식은 training data에는 잘 적합할 수 있으나, overfit의 위험이 있다. 지나치게 세분화된 tree는 train set에서만의 특징마저 반영 해버릴 수 있기 때문이다. 이는 결과적으로 test set에서는 좋지 못한 성능을 냄을 나타낸다. 따라서 split을 제한하여 약간의 bias가 생기더라도 variance가 높지 않은, '지나치게 세분화되지 않은 tree'를 만들고자 한다. 

이를 위한 한가지 방법은, 각 split을 통해 줄어드는 RSS가 특정 임계치를 넘을때에만 split을 해주는 전략이 될 수 있다. 그러나, 간혹 이전 split에서는 RSS가 많이 줄지 않았어도 다음 split에서는 RSS가 대폭 주는 경우도 있기에, 이는 근시안적인 방법으로 좋지 못하다. (예를들어, node2에서는 RSS가 10%줄었지만 node2,node3까지 하였을 경우 RSS가 40%주는 경우)

따라서 더 나은 방법은, 아주 큰 tree $$T_0$$을 만든 후 이를 적절한 subtree를 얻기위해 **잘라내는**(prune) 방식이다. 이를 pruning이라 한다. 그렇다면 어떤 기준으로 prune을 해야할까? 우리의 목표는 낮은 test error를 갖는 subtree를 찾는 것이지만, 이를 위해 모든 subtree를 CV를 해보는 것은 불가능하다. 역시나, 너무나 많은 subtree가 가능하다. 따라서 몇개의 subtree들만을 추려 고려대상으로 삼는 방법을 쓴다.

Cost complexity pruning이란 방법이 이를 가능하게 하는 방법인데, 각 $$\alpha$$에 대해 다음의 식을 최소화하는 subtree를 구하는 것이다.
$$
\sum_{m=1}^{\lvert T\lvert}\sum_{x_i\in R_m}(y_i-\hat y_{R_m})^2-\alpha\lvert T\vert
$$
여기서 $$\lvert T\vert$$는 terminal node의 수를 말하는 것이다. 식을 보면, **1)** 각 terminal node안에서의 RSS를 줄여주는 loss term과 **2)**지나치게 많은 terminal node들이 있지 않도록, 즉 complexity를 줄여주는 penalty term이 있는 형태임을 알 수 있다. 이는 Lasso의 형태와 비슷하다. 둘의 비율은 $$\alpha$$로써 조절 할 수 있는데, 이 $$\alpha$$를 cross-validation을 통해 구하면 되는 것이다.

> 위의 식을 좀더 직관적으로 이해해보자면, 새로운 split을 함으로써 생기는 RSS의 '이득'이 $$\alpha \lvert T\lvert$$라는 penalty보다 더 크지 못하다면 split을 안하는 형태라고 생각할 수 있다. 

이를 최종적으로 정리하면, 다음과 같다.

1. recursive binary splitting(즉 top-down, greedy방식)을 통해 큰 tree를 적합한다. 이때의 유일한 stop기준은 node안에 최소 기준보다 적은 수의 data가 남는 것이다.
2. 앞의 과정에서 만들어진 큰 tree에 cost complexity pruning을 한다. 이때의 $$\alpha$$는 K fold cross-validation을 통해 이루어 진다.
3.  cross-validation은 다음의 과정을 통해 이루어 진다. training data를 K개의 fold로 나눈다
   - 한개씩 fold를 뺀 모든 K개의 data set에 대해 1번의 적합을 한다
   - 각 K개의 data set을 통해 나온 tree들에 여러 $$\alpha$$값에 따라 cost complexity pruning을 하고, 그때의 error를 평균내어 validation error로 최적의 $$\alpha$$를 정한다.
4. 3번에서 정해진 최적의 $$\alpha$$값을 통해 1번에서 전체 데이터를 통해 만든 tree에 pruning을 한다.

![tree5](https://user-images.githubusercontent.com/31824102/36580553-a7dc3f5c-1860-11e8-981f-ac215c63663c.PNG)

이는 Cross-validation을 해본 그림이다. 실제로는 $$\alpha$$로써 pruning을 하지만, $$\alpha$$와 terminal node수 간에는 1-1관계에 있기에 tree size로 나타내었다.

### 8.1-2 Classification Trees

classification tree는 반응변수가 질적변수라는 것외에 크게 다를 것이 없다. regression에서는 각 영역의 평균으로 반응변수를 예측했다면, classification문제에서는 **가장 많이 등장한 클래스**로 예측을 한다. 단순히 평균을 내는것이 아니라 가장 많이 등장한 클래스로 분류를 하고 나머지 클래스는 무시해버린 격이 되기 때문에, 분류에서는 단순히 예측값 뿐 아니라 **해당 영역에 있던 다른 클래스의 비율**에도 관심이 있다.

classification tree는 만들어질때, 역시나 recursive binary splitting을 이용하여 적합하나 기준이 RSS가 아니라 classfication error rate이다. 가장 많이 등장한 클래스로 예측을 하였으니, error rate는 다음과 같다.

![tree6](https://user-images.githubusercontent.com/31824102/36580571-abd5d6ae-1860-11e8-8090-d309368077e8.PNG)

영역m에서 여러 k클래스들의 비율 중 1-(가장 많이 등장한 것의 비율) 이다. 그러나 바로 전에 논의하였듯이, 이는 다른 클래스의 비율을 고려 안했기에 충분히 민감한 기준이 되지 못한다. 따라서 실제에서는 다음의 두 척도가 대신 사용된다. 

##### Gini index

![tree7](https://user-images.githubusercontent.com/31824102/36580570-ab98df92-1860-11e8-808b-c5f6761bc46e.PNG)

첫째로, Gini index(지니 불순도)는 전체 K클래스를 모두 고려하여, 클래스의 분산을 계산한다. 특정 k클래스의 $$\hat p_{mk}$$만이 1에 가까울 수록 gini index의 값이 작아질 것을 알 수 있다. 따라서 이는 '불순도'를 나타내는 개념이다.

##### entropy

![tree8](https://user-images.githubusercontent.com/31824102/36580568-ab48a554-1860-11e8-8bea-b1d60ab53a8b.PNG)

정보이론에서 많이 활용되는 entropy의 개념이다. $$\hat p_{mk}$$이 $$0\le\hat p_{mk}\le1$$이기에, $$0\le\hat p_{mk} log(\hat p_{mk})$$이다. 이 역시 특정 k클래스의 $$\hat p_{mk}$$만이 1에 가깝고 나머지는 0에 가까울 수록 값이 작아질 것을 알 수 있다. Gini index와 같은 특성을 판별해 내는 것이다. 사실 수학적으로 둘은 매우 유사한 지표이다.

recursive binary splitting에서, 이 둘은 단순한 classification error rate보다 더 민감한 지표이기 때문에 splitting기준으로 사용된다.

해당 기준으로 tree가 다 완성된 후, **Pruning에서는**, classification error나 gini index, entropy 3기준이 모두 활용될 수 있으나 예측의 정확도를 위해 **classificaion error**가 주로 사용된다.

다음은 classification tree의 예시이다. (변수들의 의미는 크게 중요하지 않다.)

![tree9](https://user-images.githubusercontent.com/31824102/36580566-aafe65e8-1860-11e8-9d41-4340c3a32871.PNG)

오른쪽 아래의 node, [RestECG < 1] 를 보면, 이상한점을 알 수 있다. node의 양 갈래의 예측이 모두 Yes인 것이다. 둘다 Yes 라면 왜 굳이 나눈걸까? 이는 node의 purity(순수한 정도) 때문이다.  [RestECG < 1]의 오른쪽 leaf에는, 9개의 자료가 모두 Yes이고, 왼쪽 leaf에는 7/11이 Yes였다. 따라서 실제 예측을 할때에도,  [RestECG < 1]의 오른쪽으로 배정된 test data는 더욱 확실하게 Yes라고 할 수 있고 왼쪽으로 배정된 test data는 낮은 확실성을 갖고 예측을 할 수 있게 되는 것이다.  node[RestECG < 1]는 classification error에는 차이가 없지만, GIni index나 entropy의 기준에서는 차이가 있다.(!) 이러한 이유로 후자를 사용하는 것이다.

추가로, 질적변수의 경우도 역시나 node로 활용될 수 있는데, 위의 예시의 경우 [ChestPaint : a]의 node가 그 예이다. ChestPaint가 a클래스이면 오른쪽 leaf, 아니면 왼쪽 leaf로 가는 식이다.

### 8.1-3 Tree Versus Linear Models

Tree based model은 기존의 3,4장에서 다루었던 보다 더 전통적인 방법들과는 좀 다르게 생겼음을 알 수 있다. 그렇다면, 이 들중 누가 더 좋은 모델일까? 답은 역시 **그때 그때 다르다**. 만약 진짜 관계선이 선형이라면, 선형회귀가 잘 작동할 것이고 regression tree는 그만큼 명확한 선형관계를 잡아내지 못할 것이다. (애초에 regressoin tree는 해당영역의 '평균값', 즉 x축의 평행선으로 예측을 하기에, 수많은 계단식 선이 나온다.) 그러나 실제 관계가 많은 non-linear이고 복잡한 관계를 가졌다면, decision tree가 전통적인 모델보다 뛰어날 것이다. 다음은 실제 decisoin boundary가 linear일 경우와 아닐 경우의 binary classification의 단순한 예시 그림이다.

![tree10](https://user-images.githubusercontent.com/31824102/36580564-aa987d96-1860-11e8-941f-9c05f1443e1c.PNG)

예측력 외에도, 설명력이나 시각화를 위해 tree모델이 선호되는 경우도 있다.

### 8.1-4 Advantages and Disadvantages of Trees

- Tree model은 설명력이 아주 뛰어나다. 심지어, linear regression보다도 뛰어나다.
- 전통적인 방법에 비해 좀더 사람의 의사결정과 닮았다.
- 시각화하기 좋다. (역시나 설명력의 측면)
- 더미 변수를 만들지 않고도, 양적변수와 질적변수를 모두 다룰 수 있다.
- 그러나, 이 책에서 다룬 다른 방법들에 비해 **예측 정확도가 좋지는 못하다**.(prune을 한다해도, 여전히 overfit의 한계를 완벽하게 벗어나지 못했기 때문이다.)
- 추가로 tree는 'non-robust'하다. 즉, 데이터의 작은 변화에도 tree의 최종 예측값이 크게 변동할 수 있다. 즉, variance가 크다

그러나 decision tree**들**을 모아 활용하는 방식인 bagging, random forest, boosting을 통해서는 예측 정확도가 뛰어나가 향상된다. 

## 8.2 Bagging, Random Forest, Boosting

Bagging, Random Forest, Boosting은 tree들을 building block으로 활용한 모델이다.

### 8.2-1 Bagging

5장에서 소개했던 bootstrap이 이 곳에서 활용된다. 앞에서 소개되었던 decision tree는 **high variance**로 인한 약점을 가지고 있었다. (variance가 높다는 것은 다른 데이터셋에 따라 모델이 심하게 변동한다는 것을 의미한다.) Bootstrap aggregation, 줄여서 Bagging은 이런 **Variance를 줄여주기 위한** 목적을 가지고 있다. 특히나 decision tree에서 효율적으로 쓰이기에 이곳에 같이 소개되었다.

같은 분산 $$\sigma^2$$를 가지고 있는 서로 독립인 통계량 $$Z_1,..,Z_n$$들의 평균, 즉 $$\bar Z$$의 분산은 $$\frac{\sigma^2}{n}$$이란 것을 알고있다. 물론 독립에 등분산은 현실에선 많지 않지만, 우선 간단화 해보자면, **평균을 취하는 것은 분산을 감소시킨다**. 따라서 분산을 감소시키는, 그럼으로써 궁극적으로는 예측력을 높이는 자연스런 결론은 **모집단의 많은 데이터 셋에 적합**하여 여러개의 예측 모델을 만들고, 그 모델의 **예측결과를 평균내는것**일 것이다. 

그러나 당연히 현실에서는 모집단에서 무수히 많은 데이터셋을 추출하는 것이 불가능하다. 대신, 5장에서 다루었던, 주어진 training set에서 무수히 많은 B번의 반복추출로 무수히 많은 data set을 만들어내는, Bootstrap을 진행하여 B개의 여러 모델을 만들수 있다. 이들을 평균냄으로써, 위의 논의를 따라가는 최종 결과물을 얻을 수 있을 것이다. 이것이 **B**ootstrap하여 합친다(**agg**regat**ing**한다), 즉 Bagging이다.

Bagging은 여러 regression 방법들의 variance를 줄여줄 수 있지만, 특히나 high Variance로 골치를 앓고 있던 decision tree에서 유용하게 사용된다. bootstrap을 통해 B개의 training set을 만들고, 이에 대해 B개의 tree를 만드는것이다. 이때, tree들은 **prune을 하지 않는다.** 따라서 low bias이지만, 매우 높은 variance를 가지고 있는 모델이다. 이들을 최종적으로 평균내줌으로써, variance를 줄여주는 것이다. (물론 Bootstrap을 통해 만들어진 data set들은 그 구성도 매우 비슷하여, 모델들간의 covaraince도 높아진다! 따라서 위에서 논의한 '독립인'통계량의 분산 처럼 완벽히 분산을 1/n한 결과가 나오지 않는다. 뒤에서 다룰 것이다!)

> 독립이 아닐 경우 
>
> $$Var[(Z_1+..+Z_n)/n]=\frac{1}{n^2}\left\{Var(Z_1)+...+Var(Z_n)+\sum\sum Cov(Z_i,Z_j)\right\}$$
>
> $$=\frac{1}{n^2}\left\{n\sigma^2+\sum\sum Cov(Z_i,Z_j)\right\}$$
>
> $$=\frac{\sigma^2}{n}+\frac{1}{n^2}\left\{\sum\sum Cov(Z_i,Z_j)\right\}$$
>
> 따라서 각 통계량(여기선 model)간의  공분산을 줄이는 것 역시 관건이다.

#### Out-of-Bag Error Estimation

Bagging을 통해서는 각 Bootstrap을 하며 복원추출에 뽑히지 않은 데이터들(보통 2/3정도가 뽑힌다고 한다.)이 자동으로 validation set이 되어 Cross-validation을 하지 않아도 test error를 추정할 수 있다. 이 뽑히지 않은 데이터를 Out-of-bag(OOB)이라 부른다.

특정 데이터($$x_i$$)가 뽑히지 않은, 전체 B개중 약 B/3개의 모델들이 특정데이터($$x_i$$)에 대한 예측을 하고, 이를 평균낸다. 이런 식으로 모든 데이터에 대해 예측을 하여 OOB MSE나, classification error를 구할 수 있다. 물론 cross-validation이 더욱 정확한 test error추정을 가능하게 하지만, 데이터가 충분히 많고 충분히 많은 B번의 Bootstrap의 경우 OOB 역시 비슷한 성능을 내기에, 별도의 CV를 진행하지 않는다는 점에서 이점이 있다. (그러나 왠만하면 CV를 하자...)

#### Variable Importance Measures

여러개의 나무를 합하면서, 예측 정확도는 올라갔지만 대신 기존의 decision tree가 가지고 있던 해석력을 잃었다. 더이상 하나의 나무그림으로 의사결정의 절차를 표현할 수 없어졌기 때문이다. 따라서 bagging은 해석력을 희생하여 많은 예측력을 얻은 것이라 볼 수 있다.

그러나 이전만큼의 해석력은 아니지만, RSS나 Gini index를 통하여 전반적인 **예측변수들의 중요도**를 확인할 수 있다. 예를 들어 회귀문제의 경우, **모든 B개의 나무에 대해** 각 변수에서의 split으로 인해 **RSS가 감소한 정도**를 측정하여, 이를 평균을 낸다.(!) 해당 변수로 인해 RSS가 많이 감소하였으면, 이는 중요한 변수임을 의미하게 된다. 마찬가지로 분류의 경우 Gini index의 감소량을 측정한다.

다음은 앞에서 본 classification 문제에서 bagging을 사용하였을 경우의 변수 중요도이다. 이는 각각 Gini index의 평균 감소량을 통해 정렬된 것이다.

![tree11](https://user-images.githubusercontent.com/31824102/36580562-aa0ca988-1860-11e8-92f4-4b2e57841cba.PNG)

### 8.2-2 Random Forests

앞에서도 언급하였지만, Bootstrap을 통해 만든 데이터들은 그들간에 correlated된 정도가 높다. 비슷한 데이터로 만들어진 모델들은 서로 비슷한 모델일 것이다. (split기준이 비슷하고 등등..) 이는 모델의 높은 covariance를 의미하고, 따라서 bagging은 의도한 만큼 Variance가 줄어들지 못한다. 

따라서, Random Forest는 tree들을 **decorrelate**해주고자 약간의 트릭을 사용한다. 똑같이 bootstrap된 데이터를 통해 나무들을 만들지만, 각 split을 자를때, 전체 p개의 예측변수 중 **랜덤하게 선택된 m개의 변수만을 고려**하여 그 중에서 split 기준을 만든다. 또 다음 split을 만들때는 새롭게 랜덤하게 선택된 m개의 변수만을 고려하여 선택을 한다. 이때, $$m \approx \sqrt p$$로 주로 지정한다.

이로써 random forest는 각 split에서 가능한 변수들의 반도 다 안쓰는 것이다. 얼핏 이상하게 들릴지 모르지만, 여기에는 다 이유가 있다. 만약 한가지 예측변수가 매우 강력한 예측변수이고, 나머지는 적절하게 강력한 예측변수라고 해보자. 그렇다면 bagging을 통해 만들어진 tree들은, (비록 bootstrap으로 약간씩 다름에도 불구하고) 거의 모든 tree들이 가장 강력한 그 하나의 변수를 top split으로 삼고 있을 것이다. 즉, **매우 비슷해 보이는 bagged tree들**이 만들어 질것이다. 이미 언급하였듯이, highly correlated 된 통계량들을 평균내는 것은 그렇지 않은 경우에서 평균내는것 만큼의 큰 variance의 감소를 내지 못한다. 

이를 극복하고자 random forest는 일부러 split에서 몇개(m개)만을 고려하는 것이다. 이로써 몇몇 split에는 해당 강력한 변수가 고려대상에도 들어가지 않게되고, 다른 예츠변수들이 고려될 기회를 얻는 것이다. 이로써 tree들은 decorrelated된 효과를 갖게되고, 평균을 취해서 variance를 줄여주고자 하는 의도에 더욱 부합하게 된다.

언급하였듯이 $$m \approx \sqrt p$$으로 주로 지정한다. 변수들중 많은 수의 변수가 correlated되어 있는 경우 더 작은 수의 m이 더 유효할 수 있다. (그러나 왠만하면 $$m \approx \sqrt p$$을 사용하자.) 

마지막으로 예시 데이터에 대한 Bagging과 Randomforest, 그리고 그들의 OOB error에 대한 그림이다. 예상한 대로 Bagging보다 Randomforest가 더 좋은 성능을 보였다. 그리고, tree의 갯수가 어느정도 커지면 error율이 수렴을 하는 그림을 보인다. 이는 covaraince의 존재로, 해당 방법으로 얻을 수 있는 성능향상의 한계치를 나타내준다. 즉, 나무의 갯수는 충분한 정도 이상을 늘릴 필요는 없다.

![tree12](https://user-images.githubusercontent.com/31824102/36580561-a9c519ec-1860-11e8-8110-00a10c1285a3.PNG)

> 덧. 원래 decision tree는 질적변수를 더미화하지 않아도 되지만, python의 scikit-learn에선 현재 더미화를 시켜 input으로 넣어줘야만 작동이 되게 되어있다. 이 경우 질적변수의 카테고리가 많지 않은 경우 문제가 없지만, 카테고리가 많은 경우 수백개의 더미변수들이 생겨나버려 1) 차원이 커지고, 2) 각 변수들도 '그 더미변수의 클래스인지' '아닌지'만을 나타내는 매우 **sparse한 변수**가 되버려, 다른 양적변수들에 비해 필연적으로 중요하게 고려되지 못하게 된다.(비록 원래 중요한 변수였더라도!) 따라서 질적변수의 카테고리가 엄청나게 많을경우, 섣부른 더미화+RF적합은 위험할수 있다. 이때는 질적변수를 더미화하지 않고도 처리할수 있는 H2O를 써보는것도 좋다. [참고](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/)

### 8.2-3 Boosting

이번엔 decision tree를 업그레이드 시킬 수 있는 다른 방법인 Boosting에 대해 알아보자. 미리 개념에 대해 말해보자면, Boosting은 강력하지 않으나 보완에 초점을 맞춘 약한 모델(Weak leaner)를 결합해서 정확하고 강력한 모델(Strong Learner)를 만드는 것을 통틀어 의미한다. boosting은 회귀나 분류를 위한 다른 방법들에도 사용이 될 수 있지만 여기서는 decision tree에 국한하여 다뤄본다.

bagging은 bootstrap을 통해 생긴 여러 개별적인 데이터셋에 대하여 각각 개별적인 decision tree를 적합하는 방법이였다. 이때, 각 tree가 만들어지는 과정은 다른 tree들과는 관계가 없다. 그러나 Boosting은, 관계가 있다.(!) Boosting은 Bagging과 비슷한 방식으로 작동하지만, tree들이 **순차적으로 만들어진다**는 차이점이 있다. 구체적으로, 각각의 tree는 이전 tree의 정보를 이용하여 만들어진다. 따라서 Boosting은, **bootstrap sampling을 이용하지 않는다**. 대신 각각의 tree는 원래의 training data에서 (이전의 tree를 토대로) 수정된 training data set에 적합을 한다.

> 수정된 training data set이 조금 애매한 표현이지만, 이는 Boosting 기법마다 다음 함수에서 적합할 training data 다르게 구하기 때문이다. AdaBoost 같은 경우는 전체 데이터 중 이전 모델이 틀린 관측치의 가중치를, boosting tree같은 경우 이전모델의 residual을 업데이트하여 적합한다.( AdaBoost 참고는 ESL) **부스팅은 잘못 분류된 개체들에 집중하여 새로운 분류규칙을 만드는 단계를 반복하는 방법** 이라는 것만 기억하면 된다. bagging 과 boosting의 차이를 그림으로 쉽게 보여준 [참고](https://blog.naver.com/muzzincys/220201299384)

수정된 원래의 data set에 적합을 한다는게 무슨뜻일까? 이 말은, 다음 모델을 적합할때 바로 원래의 data set $$Y$$에 적합을 하는게 아니라, 현재 모델이 잡지 못한, 현재모델과 Y와의 차이인 **residual에 적합**을 한다는 것이다. 단계별로 좀더 자세히 알아보자.

> 다음은 residual을 통해 모델이 잘 못잡는 data에 집중을 한 형태를 보여주고 있다. 그러나 이는 gradient boost에서 gradient를 만드는 loss가 L2, 즉 $$c(y_i-\hat y_i)^2$$의 형태인 경우에 한정한 이야기이다. 즉, gradient boost의 근본적인 원리는 아니고 쉽게 이해할수잇는 한 갈래인것. gradient boost tree는 엄청난 [참고](http://xgboost.readthedocs.io/en/latest/model.html)

1. 우리의 최종 예측값 $$\hat f(x)$$를 우선 빈 값(0)으로 둔다. (즉, $$\hat f(x)=0$$) 첫번째 실행의 경우 residual $$r_i=y_i$$이다. 

2. for $$b=1,2,..,B$$ 만큼 다음을 반복한다.

   - 미리 지정한 $$d$$개 만큼의 split을 가지고 있는 tree를 예측변수X와 잔차$$r$$의 쌍 $$(X,r)$$에 적합하여, tree $$\hat f^b$$를 만든다. 첫번째 실행의 경우 기존의 ($$X,Y$$)에의 적합과 같다.

   - 새로운 tree $$\hat f^b$$를 미리 지정해둔 반영비율$$\lambda$$만큼 반영해준다. 다음과 같이 표현할 수 있다.

     $$\hat f(x) \leftarrow \hat f(x) +\lambda \hat f^b(x)$$

   - $$\hat f(x)$$에도 새로운 tree를 더해주었으니, residual에도 그 영향을 빼주어 계산한다. (단순하게 업데이트된 $$\hat f(x)$$의 잔차를 나타낸 것이다.)

     $$r_i \leftarrow r_i -\lambda\hat f^b(x_i)$$

   - 2번 과정을 반복하여, 업데이트되는 residual $$r_i$$에 계속 적합을 한다. B번만큼.

3. 이를 충분히 큰 수 B만큼 반복하면, 최종 모델은 2-2번에서 모든 tree들을 반영비율을 곱해 만들어진 모델이 된다. 즉 다음과 같다.

   $$\hat f(x)=\sum_{b=1}^B\lambda\hat f^b(x_i)$$

이를 보면 알 수 있지만, 처음에는 원래의 데이터 Y에 적합을 하지만, 그 이후에는 원래의 데이터 Y가 아닌 이전tree의 **잔차에 적합**을 하고 있다. 따라서 이는 천천히 배우는(slowly learning)의 방법이다. classification tree의 경우도 비슷한 원리로 적용될 수 있으나, 조금 더 복잡하다. 여기서는 다루지 않는다.

각각의 tree는 terminal node가 몇개 없는 크기가 매우 작은 tree일 수도 있다. 이는 hyper parameter $$d$$로 지정해준다. Boosting tree에는 다음과 같은 3개의 hyper paramter가 있다.

1. **Tree의 갯수 $$B$$** (split의 갯수가 아니다! 몇개의 tree를 만들지, 즉 몇번의 step을 돌지를 의미하는 파라미터). Boosting은 Variance를 줄이는데에 초점을 맞춘 bagging과 달리 bias를 조금씩 줄여나가고자 하는 방식이기에, **$$B$$가 클 경우 overfit이 될 수 있다.** Tree의 갯수 $$B$$ 역시 Cross-validation을 통해서 선택한다
2. 각 모델을 그대로 더하는 것이 아니라, **반영비율 $$\lambda$$**를 곱해주었다. 이를 **shrinkage parameter**라고 불렀다. 이는 사실 각 step에서 배운 것을 얼마나 반영할지, 즉 learning rate와 비슷한 개념이다. 보통 $$\lambda$$는 0.01이나 0.001을 사용한다. $$\lambda$$가 작으면 최적의 성능을 보이기 위해 더 큰 수의 $$B$$가 필요할 것이다. (한번의 step에서 학습하는 비율이 적으니까.)
3. **각 tree내에서 split의 갯수 $$d$$**. 즉 boosting의 **complexity**를 조절해주는 지표이다.(complexity가 높은 모델->flexible한 모델->variance는 크고 bias는 작고.)  Boosting은 이미 bias를 줄이는데에 목표를 두어 설계된 방법이기에, 큰 complexity를 필요로 하지 않는다. 역시나 overfit의 위험이 있기 때문이다. 때때로, $$d=1$$, 즉 **하나의 분류기준만이 있는 것이 제일 좋을때도** 있다. 이는 각 tree가 하나의 분류기준, 즉 하나의 변수기준만을 포함했다는 점에서, **additive model**의 맥락에서 생각할 수도 있다. 이러한 맥락에서, $$d$$는 $$d$$개의 변수를  고려하게 되기 때문에 $$d$$를 interaction의 깊이라고 생각할 수도 있다. (선형회귀의 $$X_1X_2$$항 처럼 여러 변수를 고려하는 tree가지가 만들어지기 때문.)

다음은 random forest의 예시에 사용되었던 데이터에 boosting tree를 적합한 그림이다. 

![tree13](https://user-images.githubusercontent.com/31824102/36580559-a98295fe-1860-11e8-832c-26bbfd1a96dd.PNG)

사실 세 선모두 표준편차를 고려하면 유의미하게 다르지 않지만, 그래도 depth $$d$$가 1인 모델이 더 잘하였다는 점이 눈에 띈다. Boosting은 이전 모델의 실수를 기반으로 만들어지는 모델이기에, 각각의 개별 tree는 작은 tree여도 충분한 경우가 많다.





> 여기서는 Boosting tree를 다루었지만, weak을 결합하여 strong을 만드는 boosting의 개념은 여러가지로 적용될 수 있다. 심지어는 Gradient에 적용될 수도 있는데, 목표와의 차이를 Loss function을 통해 정해주고 그로써 정량화된 목표와의 차이, Gradient를 토대로 함수자체를 update해주는 것이다. 대표적으로 Loss function이 L2($$\frac{1}{2}(y-f_i)^2$$)인 경우, 기울기는 $$y-f_i$$이다. 이를 고치는 것을 다음 모델의 타겟으로 넘기는 것이다. 모델을 적합하고, 다시 잔차에 다른 모델을 적합하고, 다시 그 잔차에 모델을 적합하고...이러한 방식이 L2를 Loss로 설정한 경우의 Gradient Boosting이다.

---

참고 : 

쩌는 슬라이드 : https://www.slideshare.net/freepsw/boosting-bagging-vs-boosting