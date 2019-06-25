---
layout: post
title: "[데이터분석 정리]Local Outlier Factor(LOF) 개인적 정리"
categories:
  - 머신러닝
tags:
  - Local Outlier Factor
  - LOF
  - outlier detection
comment: true
---



원문 : https://dl.acm.org/citation.cfm?id=335388

LOF는 대표적인 outlier detection의 기법중 하나이다. LOF의 문제의식은, 문제의식은 기존의 방법들이 **local정보에 대한 고려**가 없다는것이다. 데이터들간의 특성에 따라, 어떤 집단(혹은 군집)에선 매우 가까운 거리가, 어떤 집단에선 매우 먼 거리일 수 있다는 것이다. 자세한 설명은 아래의 그림과 함께 하겠다.

Density based method는 **density가 상이한 클러스터들**이 있을때 문제가 발생한다. 기존의 dense based 방법론들은 'dense' 라는 개념을 정의하기 위해, 특정한 **window size나 최소 갯수**등을 이용하였다. 예를들면 '거리가 c 이하인 window 내에 들어오는 데이터가 k개 이상인가?'로 dense를 지정하였다. 그러나 density가 상이한 경우, 기존의 방법론처럼 dense에 대한 절대적인 기준을 지정할수가 없어지기 때문이다. knn-distance method역시, 각 클러스터에 대해 outlier를 고르기 위한 적절한 knn-distance가 달라진다. 다음 그림을 보면 이해가 빠르다. 집단 $$C_1$$과 집단 $$C_2$$의 density가 다르기에, $$o_1$$은 걸러내기 쉽지만 $$o_2$$는 걸러내기가 어렵다. $$C_1$$의 대부분 데이터들이 그정도는 떨어져 있었기 때문에, 일정 거리로 기준을 삼을 경우, $$C_1$$혹은 $$C_2$$에만 특화된 outlier detction을 하게 된다.

<img width="211" alt="lof1" src="https://user-images.githubusercontent.com/31824102/54182467-eb27d800-44e4-11e9-98c0-c5142a46d930.PNG">

이러한 문제의식에서, **local의 상대적인 dense**를 비교하여 outlier를 정하자는 lof가 나왔다. 큰 틀은, neighbor들의 dense를 고려하여 비교한다는 것이다. 이때 몇가지 새로운 정의들이 나온다.

**1. k_distance(p)**

우선 **k_distance(p)는**, 특정 데이터p에서의 k개 nearest neighbor까지의 거리이다. (3_distance(p)는 3번째로 가까운 데이터와의 거리) 이게 후에 상대적인 dense로써 작용할것이다. 

또, distance가 continuosu라면 3_distance내에 정확히 3개의 neighbor가 들어있겠지만, 거리가 1,2,3,3,3,3같이 discrete해서 겹치는 경우라면 3_distance내에 5개든 10개든 neighbor로 들어있을 수는 있다. 이를 따로 나타내주기 위해 k_distance(p)안에 들어온 데이터갯수를 $$N_k(p)$$라고 부른다.

**2. reachability distance(p,o)**

**reach_dist(p,o)**는, p에 대해서 생각할때, **주변 데이터 o의 k_distance를 고려한** 거리이다. 관심데이터p가 주변데이터o의 k_distance내에 들어와 있으면 o의 k_distance, 그것보다 밖에 있는 경우면 그냥 p와o의 거리를 잰다. 식으로 나타내면 다음과 같다.

reach-distance(p,o)=max{k_distance(o),dist(p,o)}

p와 o가 매우 붙어있더라도, o의 k_distance만큼은 거리를 뻥튀기해서 계산해주겠다는 개념이다. 이는 후에 이 reach_distance로 서로 dense를 비교할것이기 때문에, 너무작은 값을 갖지 않도록 하는 **일종의 범퍼**이다.

**3. local reachability density(p)**

이제 거의 다왔다. lrd(p)는, p주변의 k_neighbor들과의 reach_dist의 평균을 inverse취한 것이다. 식으로 나타내면 다음과 같다.
$$
lrd_k(p)=[{\frac{\sum_{o\in N_k(p)}(reah\_dist(p,o))}{N_k(p)}}]^{-1}
$$
이를 통해 주변의 dense를 고려한 p점에서의 'neighbor들과의 적당한 거리'를 나타낼 수 있다. 물론, lrd는 inverse이라는걸 알아만두자.

그림을 통해 보면 쉽다. 그림 [원문](https://jayhey.github.io/novelty%20detection/2017/11/10/Novelty_detection_LOF/)

<img width="484" alt="lof2" src="https://user-images.githubusercontent.com/31824102/54182468-eb27d800-44e4-11e9-8c1d-3aee82c7f327.PNG">

case1의 경우,  파랑점의 lrd는 초록점들과의 거리의 평균, 혹은 뻥튀기된, 초록점들의 k_distance들의 평균의 역수가 된다. 그러나 k_distance던 그냥 거리던 평균거리가 작을 것이므로, lrd의 값은 크게 된다.

반면 case2의 경우, 평균거리는 상당히 클것이기에 lrd는 작은 값을 갖게 된다.

**4. Local Outlier Factor(p)**

드디어 마지막 단계이다. LOF(p)는, p의 $$N_k(p)$$에 속하는 모든 다른점$$o$$에 대해서 lrd의 비율을 구하고 이를 평균낸것이다. 수식으론 다음과 같다.
$$
LOF_k(p)=\frac{\sum_{o\in N_k(p)}\frac{lrd(o)}{lrd(p)}}{N_k(p)}
$$
쉽게 말해 주변의 점들 o와의 dense(lrd)를 비교하여 평균낸것이다. 사실 lrd(o)/lrd(p)는 이전의 정의가 역수임을 고려하면 **'p의 평균거리'/'o의 평균거리'를 구하고, 이를 평균낸것**으로 보면 된다. (편의상 'neighbor들과의, (reach_dist라는 버퍼를 씌운) 평균거리'를 그냥 '평균거리'라고 표현했다.) 

내 관심대상인 **p의 'neighbor들과의 평균거리'**를 **주변 neighbor들의 '평균거리'**와 비교하는 것이다. (사실 이게 더 직관적인거같은데 왜 굳이 역수를 썻는지 모르겠다)

<img width="515" alt="lof3" src="https://user-images.githubusercontent.com/31824102/54182471-eb27d800-44e4-11e9-990a-75474a4cce14.PNG">

case1이나 case3같이 주변애들과 '평균거리'가 크게 차이나지 않는 점의 경우 lof는 1에 근사하게 나올 것이다. 그러나 case2 같이 주변애들(초록점)이 가진 평균거리에 비해 평균거리가 더 긴 데이터(파랑점)의 경우는 lof가 1보다 더 크게 나오기 쉽상일 것이다. **즉**, lof$$\approx$$1이면 정상데이터, lof$$\gg$$1이면 outlier인 셈이다.

> 이렇게 상대적으로 비교하기 위해서, 버퍼인 reach_dist의 정의가 필요했던듯하다. 데이터가 **엄청나게 dense한지점에 있는 경우** 미세한 차이로 ratio가 **엄청 sensitive해질 수** 있으니, 버퍼를 씌워줘서 robust하게 만들었다.

이를 통해 local한 기준에서 평가를 하기에, 다음 그림과 같이 밀집한 지역에서는 더 빡빡한 기준으로 outlier를 잡고, 엉성한 지역에서는 더 엉성한 기준으로 outlier를 잡아낼 수 있게 된다. 문제점은 1.k를 몇으로 할지인 고질적인 문제. 그러나 경험적으로 **k=20**정도로 하는것이 좋다고한다. 그리고 **2. threshold를 얼마로 잡아야할지를** 알기 힘들다는 점이다. (파이썬에선 contamination이라는 옵션으로 train data중 몇%가 outlier인지를 우리가 지정해준다. auto는 0.2)

다음은 서로다른 dense를 가진 집단에 대해 LOF를 나타낸 toy example이다. 우상단의 집단와 좌하단의 집단들간의 dense가 다름을 볼 수 있다. 그러나 서로 다른 dense에도 불구하고, 상대적으로 집단에서 벗어나있는 데이터는 대략적으로 lof가 1.1이상을 띄고 있음을 볼 수 있다.

<img width="379" alt="lof4" src="https://user-images.githubusercontent.com/31824102/54182454-e8c57e00-44e4-11e9-8b00-fb3f2f7fb765.PNG">



