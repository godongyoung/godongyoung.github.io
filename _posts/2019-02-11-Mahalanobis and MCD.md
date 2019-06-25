---
layout: post
title: "[데이터분석 정리] Mahalanobis거리와 MCD 개인적 정리"
categories:
  - 머신러닝
tags:
  -  Mahalanobis distance
  - Minimum covariance determinant
  - outlier detection
  - fast MCD
comment: true
---
마할라노비스 거리는 다변량 거리의 기본이다. 개념자체는 쉽다. 다변량의 데이터에서, 분포의 형태를 고려하여 거리를 재겠다는 문제의식에서 등장한 거리 척도이다. 

$$
d(u,v)=\sqrt{(u-v)\Sigma^{-1}(u-v)^T}
$$

다변량의 데이터 $$u$$와 $$v$$의 mahalanobis거리를 구하는 식이다. 대표적으로는 $$u$$에는 각 데이터, $$v$$는 데이터의 평균이 될것이다. (예를 들면 $$u=$$(키1,몸무게1), $$v=$$(키평균,몸무게평균))

식을 보면, 마치 단변수에서 z-score를 구하듯이, covariance matrix의 inverse matrix를 곱하여 거리를 재는 방식이다. 이를 통해 변수들간의 correlation등 분포를 고려하여 거리를 잴 수 있다. 실제로, 모든 변수들이 independant이고 variance가 1로 정규화되어 있다면, $$\Sigma=\boldsymbol I$$가 되고, 마할라노비스 거리는 유클리드 거리와 같아진다. 

(즉, $$d(u,v)=\sqrt{(u-v)I^{-1}(u-v)^T}=\sqrt{(u-v)\cdot(u-v)}=\sqrt{(u_1-v_1)^2+..+(u_p-v_p)^2}$$) 아래의 그림처럼, 데이터들이 강한 correlation을 가지고 있는 경우 그 분포를 고려하여, 유클리디언 상으로는 같은 거리일 지라도 correlation에 따라 중심에서 먼 정도가 달라지게 된다. 그림 상의 두 빨간 점은 모두 원점으로부터의 거리는 같지만, mahalanobis는 등고선에서 나와있듯이, x축의 데이터를 중심에서 더 가까운 데이터라고 본다. (correlation을 고려하였을때 y축의 데이터가 더 rare한 경우이기 때문이다.)

<img width="488" alt="mahalanobis1" src="https://user-images.githubusercontent.com/31824102/60076891-a8bc9b00-9763-11e9-9116-221ba9a1274e.PNG">



scipy : https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html

sklearn : https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html

### distribution of distance

원문 : https://core.ac.uk/download/pdf/22873068.pdf

신기한점은, 만약 데이터가 **다변량 정규분포를** 따른다면, distance의 **exact distribution**을 알 수 있다. (물론 그 distribution이 true parameter $$\mu,\Sigma​$$에 베이스하기에 결국은 근사적.) **변수가 p개인** 데이터의 maha거리의 제곱은, 다음의 **카이제곱분포**를 가진다.

$$
d^2_{\Sigma}(X_i,\mu)\sim \chi^2_{p}
$$

카이제곱분포의 성질에 따라 평균p와 분산 2p도 자연스레 내포한다.

> 느낌적으로는, maha거리가 마치 p차원의 데이터에 대한 normalize처럼 작용하기에, 표준정규분포Z를 제곱해서 p개 더한듯한 느낌이라고 생각하자

(그 외에도, 이리 저리 조합해서 beta 분포, F분포 등을 도출해낼 수 있다. 이건 너무 깊게 들어가기에, 논문을 참고하는것이 좋다.)

이에 따라 카이제곱분포의 0.975 quantile지점까지를 기준으로 삼고 **다변량의 관점**에서, outlier이다 아니다 등으로 딱 떨어지게 논할수가 있다. 이에 따라 단변량의 z-score를 고려하는 것보다 좀더 합리적인 outlier를 검출할 수 있게 된다.

그러나 (뒤에설명할) MCD를 이용한 robust에서는 exact distribution을 도출할 수 없다. MCD estimator역시 consistency를 가지기에 근사적으로 카이제곱을 따른다 할 수 있지만, 경험적으로 0.975로 끊을 경우 필요이상을 이상치라고 말하게 된다고 한다. 뒤에서 더 자세히 설명하겠다.

# Minimum covaraicen determinant(MCD)

원문 : https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd.pdf

Mahalanobis를 구하기 위해선 **covariance matrix**를 구해야하고, 사실 이것이 majhalanobis거리의 핵심이다. 그러나 variance는 제곱텀에서 만들어지기에, **outlier에 굉장히 취약**하다. 따라서 몇개의 이상한 극단치 데이터들이 covairance matrix를 망칠 수 있고 이에 따라 합리적이지 않은 distance가 측정될 수 있는것이다. 이렇듯 oulier에 취약한 variance의 문제를 완화하고자 나온 개념이 바로 MCD이다. **robust variance**정도로 받아들이면 좋다. Variance matrix를 이용하는 mahalanobis 거리를 공부하면서 접하게 된건데, mcd를 이용할 경우 아래의 그림과 같이, 소수의 outlier에 대해 영향을 받지 않고 variance를(나아가 mahalanobis 거리를) 계산하게 된다. 단변수뿐아니라 다변수에서도 활용가능하다. robust한 cov_matrix를 구하는것이기에, **outlier로부터 robust한 correlation을 구하는데도 사용**될 수 있다.

<img width="384" alt="mahal1" src="https://user-images.githubusercontent.com/31824102/54182455-e8c57e00-44e4-11e9-8e34-a71efd12540f.PNG">

정의는 간단하다. 주어진 n*p의 데이터에서 **Determinant of sample covariance matrix**를 **최소로 만드는 h개의 데이터**를 뽑아 그들만 이용하여 variance나 mean을 구한다. 

>Cov_mat의 Determinant는 단변수의 std와 아주아주 rough하게 연관되 있다. multivariate normal일경우 어느정도 [성립](https://math.stackexchange.com/questions/889425/what-does-determinant-of-covariance-matrix-give). 즉 coaviance matrix의 '크기'를 determinant로 계산한다고 보면 된다. 단변수인 경우 그냥 variance를 최소로 하는 h개를 iteration돌며 뽑아서 계속 계산해본다생각하면 된다. 그러나 matrix의 경우 크기라는 개념이 애매해지기 때문에 determinant를 사용하게 된다.

MCD계산에 사용되는 데이터의 갯수 h는 $$(n+p+1)/2\le h\le n$$을 이용한다. ($$2p\le n$$의 조건이 필요하지만 차원의 저주로부터 안정적이기 위해 $$5p\le n$$정도를 요한다고 한다.) 이 경우 consistenct나 asymptotic normality등의 **대표본성질이 여전히 만족**한다는 특성이 있다.

또한, MCD estimator(mu,sigma)는 **affine equivariant**하다는 특성이 있다.

> affine equivariant란?
>
> for any nonsingular matrix $$A$$ and constant vector $$b\in R^p$$에 대해서,
>
> '(AX + b)의 MCD_mu' = A'(X)의 MCD_mu' + b
>
> '(AX + b)의 MCD_sigma' = A'(X)의 MCD_sigma' A^T
>
> 즉, 우리가 흔히 아는 mu와 variance의 사칙연산의 성질이 MCD estimator에게도 그대로 유지된다!
>
> 왜? 모든 subset h에 대해서, Determinant의 order가 유지되기 때문에. 원문참조
>
> <img width="563" alt="mcd1" src="https://user-images.githubusercontent.com/31824102/54182457-e95e1480-44e4-11e9-8400-98af856d5b91.PNG">

이러한 방법을 통해, 극단적인 데이터를 제외하고 좀더 robust한 covariance matrix를 계산할 수 있게 된다. 또한, covariance matrix의 off-diagonal term이 covariance라는 점에서, 이는 outlier를 제외한 **robust correlation**으로도 활용될 수 있다.

그러나 정확한 MCD 계산은 전체 n개의 데이터 중 h개의 데이터를 계속 뽑아서 계속 variance matrix를 구하고 determinant를 계산해야하기 때문에 $$_nC_h$$번의 계산작업을 요한다. 즉 computation이 매우 버겁다. 이를 완화하고자 FAST_MCD 방법이 탄생하였다.

### Fast MCD

$$_nC_h$$는 n이 커질때마다 기하급수적으로 커진다. 따라서, 굉장한 연산량을 요한다. 이를 근사하기 위한 fast MCD방법이 있는데, 다음과 같은 순서를 통해 계산된다. 

1. n개의 data중 h개의 subset $$H_1$$을 뽑고, 그들로 $$\hat \mu_1, \hat \Sigma_1$$를 구한다.

2. 위에서 구해진 estimator를 이용해서 전체n개의 데이터에 대한 mahalanobis 거리를 계산한다, 즉

   $$d_1(i):=\sqrt{(x_i-\hat\mu_1)^T\hat\Sigma_1^{-1}(x_i-\hat\mu_1)}$$, for $$i=1,..,n$$

3. 2번에서 구해진 거리에 따라, 거리가 가장 최소인 h개의 subset $$H_2$$을 다시뽑고, 그들로 $$\hat\mu_2,\hat\Sigma_2$$를 구한다. (이경우, det($$\hat\Sigma_1$$)$$\ge$$det($$\hat\Sigma_{2}$$)이 보장된다.)

4. 2번3번을 반복한다. 언제까지? det($$\hat\Sigma_i$$)=0이되거나, det($$\hat\Sigma_i$$)=det($$\hat\Sigma_{i+1}$$) 이 될때까지!

이와 같은 방법은 정확한 MCD보다 빠르게 수렴하게 된다. **그러나, global minimal point에 도달한다는 보장이 없다.** 고로 approximation. 

> 또한 initial $$H_1$$을 뽑을때에는 h개가 아닌 p+1개를 뽑거나, 그 p+1개가 det($$\Sigma_1$$)=0인경우 하나씩 더 추가하며 뽑는 initialize가 더 좋다고 한다. 단순히 outlier가 덜 들어가기 때문이라고 설명해주었다.

### 활용법

MCD로 oulier를 trim한후 regresion하는 Least Trimmed Square와(trimmed방법은 여러가지로 사용된다. trimmed mle등 무수히 많을수)  robust Covariance matrix의 eigenvector를 이용하는 ROBPCA등이 가장 대표적이다.

덧. 아래는 원문을 읽어야 더 도움이 되는 부분인 이론적인 파트이다. 그러나 MCD의 distance 자체가 활용되는 경우는 많지 않기에, 참고만 해도 좋을듯하다.

### MCD 를 이용한 distance의 distribution

역시나 원문 : https://core.ac.uk/download/pdf/22873068.pdf

해당 논문에서는 이론적인 도출은 못함. 그러나 추론적인 도출과, 그에 대해 경험적인 진단을 해서 정당성을 획득했다. 추론의 과정을 아주 축약하자면 다음과 같다.

1. 데이터가 정규분포를 따를때, MCD로 찾은 Sigma($$S^*$$)는 다음의 분포를 따른다. ($$m$$은 unknown degress of freedom, $$c$$는 다음을 만족시키는 adjust constant $$E[S^*]=c\Sigma$$). 
   $$
   mc^{-1}S^*\sim Wishart_p(m,\Sigma)
   $$

2. tail쪽에 있는 data들은 MCD의 estimate과 독립인듯한 형태를 띄고 있다.

3. 고로 tail의 데이터 $$X_i$$는 $$S^*$$와 독립이기에, tail의 element는 다음의 F분포로 근사할 수 있을것이다!
   $$
   \frac{c(m-p+1)}{pm}d^2_{s^*}(X_i,\bar X^*) \approx F_{p,m-p+1}
   $$








3번을 보면 알 수 있듯이, 엄밀하게는 distance에 대한 분포라기 보단 extreme데이터에 국한된 distance의 분포이다. 또한 $$m,c$$에 대한 추정은 $$m,S^*$$이 multiple of Wishart의 분포를 가지고 있다는 가정하에서 다음의 추정을 할 수 있다고 한다. 

<img width="328" alt="mcd2" src="https://user-images.githubusercontent.com/31824102/54182458-e95e1480-44e4-11e9-9656-fb43b8c59802.PNG">

