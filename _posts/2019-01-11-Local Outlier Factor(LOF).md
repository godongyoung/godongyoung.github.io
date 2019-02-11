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

# Local Outlier Factor(LOF)

원문 : https://dl.acm.org/citation.cfm?id=335388

문제의식은 기존의 방법들이 local한 정보에 대한 고려가 없다는것.

구체적으로는 Density based method는 density가 다른 클러스터들이 있을때 문제가 발생한다. dense를 정의할 window size나 최소 갯수등을 지정할수가 없어지기 때문.(HDBSCAN은 괜찬나???) knn-distance method역시, 각 클러스터에 대해 outlier를 고르기 위한 적절한 knn-distance가 달라진다. 다음 그림을 보면 이해가 빠르다. $$C_1$$과 $$C_2$$의 density가 다르기에, $$o_1$$은 걸러내기 쉽지만 $$o_2$$는 걸러내기가 어렵다. $$C_1$$의 대부분 데이터들이 그정도는 떨어져 있었기 때문.

![lof1](C:\Users\admin\내파일\4-1.5\개인적인 복습 및 정리\data\lof1.PNG)

이러한 문제의식에서, **local의 상대적인 dense**를 비교하여 outlier를 정하자는 lof가 나옴. 큰 틀은, neighbor들의 dense를 고려하여 비교한다는 것이다. 이때 몇가지 새로운 정의들이 나온다.

**1. k_distance(p)**

우선 **k_distance(p)는**, 특정 데이터p에서의 k개 nearest neighbor까지의 거리이다. (3_distance(p)는 3번째로 가까운 데이터와의 거리) 이게 후에 상대적인 dense로써 작용할것이다. 

또, distance가 continuosu라면 3_distance내에 3개의 neighbor가 들어있겠지만, 거리가 1,2,3,3,3,3같이 discrete해서 겹치는 경우라면 3_distance내에 5개든 10개든 neighbor로 들어있을 수는 있다. 이를 따로 나타내주기 위해 k_distance(p)안에 들어온 데이터갯수를 $$N_k(p)$$라고 부른다.

**2. reachability distance(p,o)**

**reach_dist(p,o)**는, p에 대해서 생각할때, **주변 데이터 o의 k_distance를 고려한** 거리이다. p가 주변데이터o의 k_distance내에 들어와 있으면 o의 k_distance, 그것보다 밖에 있는 경우면 그냥 p와o의 거리를 잰다. 식으로 나타내면 다음과 같다.

reach-distance(p,o)=max{k_distance(o),dist(p,o)}

p와 o가 매우 붙어있더라도, o의 k_distance만큼은 거리를 뻥튀기해서 계산해주겠다는 개념이다. 이는 후에 이 reach_distance로 서로 dense를 비교할것이기 때문에, 너무작은 값을 갖지 않도록 하는 일종의 범퍼이다.

**3. local reachability density(p)**

이제 거의 다왔다. lrd(p)는, p주변의 k_neighbor들과의 reach_dist의 평균을 inverse취한 것이다. 식으로 나타내면 다음과 같다.
$$
lrd_k(p)=[{\frac{\sum_{o\in N_k(p)}(reah-dist(p,o))}{N_k(p)}}]^{-1}
$$
이를 통해 주변의 dense를 고려한 p점에서의 'neighbor들과의 적당한 거리'를 나타낼 수 있다. 물론, lrd는 inverse이라는걸 알아만두자.

그림을 통해 보면 쉽다. 그림 [원문](https://jayhey.github.io/novelty%20detection/2017/11/10/Novelty_detection_LOF/)

![lof2](C:\Users\admin\내파일\4-1.5\개인적인 복습 및 정리\data\lof2.PNG)

case1의 경우,  파랑점의 lrd는 초록점들과의 거리의 평균, 혹은 뻥튀기된, 초록점들의 k_distance들의 평균의 역수가 된다. 그러나 k_distance던 그냥 거리던 평균거리가 작을 것이므로, lrd의 값은 크게 된다.

반면 case2의 경우, 평균거리는 상당히 클것이기에 lrd는 작은 값을 갖게 된다.

**4. Local Outlier Factor(p)**

드디어 마지막 단계이다. LOF(p)는, p의 $$N_k(p)$$에 속하는 모든 다른점$$o$$에 대해서 lrd의 비율을 구하고 이를 평균낸것이다. 수식으론 다음과 같다.
$$
LOF_k(p)=\frac{\sum_{o\in N_k(p)}\frac{lrd(o)}{lrd(p)}}{N_k(p)}
$$
쉽게 말해 주변의 점들 o와의 dense(lrd)를 비교하여 평균낸것이다. 사실 lrd(o)/lrd(p)는 이전의 정의가 역수임을 고려하면 **'p의 평균거리'/'o의 평균거리'를 구하고, 이를 평균낸것**으로 보면 된다. (편의상 'neighbor들과의, (reach_dist라는 버퍼를 씌운) 평균거리'를 그냥 '평균거리'라고 표현했다.) 

내 관심대상인 p의 'neighbor들과의 평균거리'를 주변 neighbor들의 '평균거리'와 비교하는 것이다. (사실 이게 더 직관적인거같은데 왜 굳이 역수를..)

![lof3](C:\Users\admin\내파일\4-1.5\개인적인 복습 및 정리\data\lof3.PNG)

case1이나 case3같이 주변애들과 '평균거리'가 크게 차이나지 않는 점의 경우 lof는 1에 근사하게 나올 것이다. 그러나 case2 같이 주변애들이 가진 평균거리에 비해 평균거리가 더 긴애들은 lof가 1보다 더 크게 나오기 쉽상일 것이다. **즉**, lof$$\approx$$1이면 정상데이터, lof$$\gg$$1이면 outlier인 셈이다.

> 이렇게 상대적으로 비교하기 위해서, 버퍼인 reach_dist의 정의가 필요했던듯하다. 데이터가 **엄청나게 dense한지점에 있는 경우** 미세한 차이로 ratio가 **엄청 sensitive해질 수** 있으니, 버퍼를 씌워줘서 robust하게 만든듯

이를 통해 local한 기준에서 평가를 하기에, 다음 그림과 같이 밀집한 지역에서는 더 빡빡한 기준으로 outlier를 잡고, 엉성한 지역에서는 더 엉성한 기준으로 outlier를 잡아낼 수 있게 된다. 문제점은 1.k를 몇으로 할지인 고질적인 문제. 그러나 경험적으로 k=20정도로 하는것이 좋다고한다. 그리고 **2. threshold를 얼마로 잡아야할지를** 알기 힘들다는 점이다. (파이썬에선 contamination이라는 옵션으로 train data중 몇%가 outlier인지를 우리가 지정해준다. auto는 0.2)

![lof4](C:\Users\admin\내파일\4-1.5\개인적인 복습 및 정리\data\lof4.PNG)


