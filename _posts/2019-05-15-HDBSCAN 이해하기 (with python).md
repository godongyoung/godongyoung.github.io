---
layout: post
title: "[데이터분석 정리]HDBSCAN 이해하기 (with python)"
categories:
  - 머신러닝
tags:
  - HDBSCAN
  - DBSCAN
  - clustering
  - outlier detection
  - python implementation
comment: true
---

density based clusering 방법론중 가장 대표적인 방법이 바로 DBSCAN이다. 그러나 DBSCAN은 local density에 대한 정보를 반영해줄 수 없고, 또한 데이터들의 계층적 구조를 반영한 clustering이 불가능하다. 이를 개선한 알고리즘이 HDBSCAN이다. 다음은 파이선의 hdbscan 패키지에서의 설명글을 바탕으로 hdbscan의 적합방법과 특성에 대해 정리한 글이다. 

toy example에 대해 직접 hdbscan을 적합하며, 데이터에 어떤식으로 적합이 이뤄지는지를 따라나가보자.

## 준비 단계

먼저 필요한 패키지들을 import 해보자.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
%matplotlib inline
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
plt.rcParams["figure.figsize"] = [9,7]
"""!pip install hdbscan"""
import hdbscan
```

다음으로 적합을 할 toy 데이터를 만들어보자. 여기에선 기존의 샘플 데이터에서 약간 손을 봤다.  우리는 아래의 데이터를 클러스터링 해볼것이다.


```python
num=100
moons, _ = data.make_moons(n_samples=num, noise=0.01)
blobs, _ = data.make_blobs(n_samples=num, centers=[(-0.75,2.25), (1.0, -2.0)], cluster_std=0.25)
blobs2, _ = data.make_blobs(n_samples=num, centers=[(2,2.25), (-1, -2.0)], cluster_std=0.4)
test_data = np.vstack([moons, blobs,blobs2])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
plt.show()
```


![output_5_0](https://user-images.githubusercontent.com/31824102/60085113-b2e69580-9773-11e9-8fdf-9e9b7b4aec4b.png)

반원형 데이터들은 **매우 오밀 조밀**하고, 왼쪽위와 오른족 아래의 원들은 그보다는 **밀도가 낮은 타원**이다. 그리고 오른쪽위와 왼족아래는 그것보다 **더더욱 밀도가 낮은 타원형태**이다.(분산이 4배)  이러한 데이터는 dense가 각기 달라, 만약 반원의 기준에 맞추게 되면 타원데이터들은 모두 noise로 처리가 되거나 이상한 클러스터에 속하게 될것이다.

그럼 이 데이터에 hdbscan을 적합하고, 해당 결과물을 가지고 직접 따라가보자.


```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)
```

## 이제, HDBSCAN에 대해 알아보자!

HDBSCAN이 어케 작동하는지를 다음의 스텝을 따라 확인해볼것이다.

1. Transform the space according to the density/sparsity.

2. Build the minimum spanning tree of the distance weighted graph.

3. Construct a cluster hierarchy of connected components.

4. Condense the cluster hierarchy based on minimum cluster size.

5. Extract the stable clusters from the condensed tree.


## Transform the space according to the density/sparsity.
우선 첫째로 하고픈건, distance를 좀더 robust하게 만드는것이다. 왜그렇게 하냐? 계층적 클러스터링의 대표적인 알고리즘 single linkage은 작은 distance에 특히 민감하게 반응하는데, 이때 noisy한 데이터로 생긴 distance로 hierachy가 심하게 변동되기 때문이다. 

고로 우리가 사용할 **transformed된 distance metric**은 다음과 같다. 이때 core_k(a)는 a의 k-th nearest neighbor까지의 거리이다. (robust한 distance를 만드는 방식은 LOF의 distance와 매우 유사하다!) 

$$d_{mreach−k}(a,b)=max[{core_k(a),core_k(b),d(a,b)}]$$

이를 **mutual reachability**라고 부른다. 두점 a와 b의 거리를 잴때 **{1. a의 이웃과의 거리, 2. b의 이웃과의 거리 3. a와 b자체의 거리}중 max값**을 고르는 것이다. 이로써 dense한 지점의 데이터는 core_k가 매우 작기에 d(a,b)를 바로 사용하고, dense가 낮은 지점의 경우 우연히 한점이 바로 옆에 존재해도, core_k의 주변 정보를 사용하게 된다. 이는 distance의 robustness를 늘리고, 최종적으로는 더 효율적인 clustering을 가능하게 한다고 한다. [참조](https://arxiv.org/pdf/1506.06422v2.pdf)

## Build the minimum spanning tree of the distance weighted graph.


이제 이 mutual reachability를 이용하여 각 데이터들간의 거리를 구할 수 있다. 이를 이용해서 각 데이터를 이은 graph를 그릴 것이다. 

데이터를 각 꼭지점으로 삼으며 잇되, 그 이은 선(edge)에 점수(mutual reachability)를 부여한다.  distance가 weight인 graph(길이가 길수록 weight도 커짐)로 만드는것이다. 사실 의미상 weight보단 그냥 점수, 혹은 인덱스 정도로 이해하면 된다.

패키지상에선 minumum spanning tree라는 함수를 통해 이것의 graph의 결과를 보여줄 수 있다. 트리의 적합은 아직 추가되지 않은 점중 **가장 가까운**(가장 점수가 낮은)edge를 **하나씩만** 추가하며, 결과적으로 모든 점을 포괄할때까지 트리를 키워나간다. (이때 wegiht는 그냥 distance가 아닌 mutual reachabillity인것 상기, 거리는 어차피 고정이므로, 어느점에서 트리를 시작하던 최종 트리는 똑같음. unique)


```python
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', 
                                      edge_alpha=0.6, 
                                      node_size=10, 
                                      edge_linewidth=2)
```

![output_11_1](https://user-images.githubusercontent.com/31824102/60085114-b2e69580-9773-11e9-9fb1-dc52dfaffaf4.png)



## Construct a cluster hierarchy of connected components.

이제 이를 바탕으로 '계층'을 만드려한다. 구체적으로는 이 점수(혹은 weight)를 점차 낮추면서, 하나씩 graph를 끊는다.(mutual reachability distance가 0.9인 지점의 연결선을 끊고, 0.8끊고, 0.7 끊고...이런식!) 그후,만들어진 minimum spanning tree를 가장 가까운 거리부터 (우리가 아는 기존의 hierachy clustering처럼) 묶는다. 


이때 가장 가까운 애 하나만 있으면 그 component자체를 연결해주니, Single linkage라고 할 수 있다. (그러나 pure distance가 아니라 mutual reachability를 사용했으니, **robust single linkage**라고 한다)

아래의 그림을 보면 mutual reachability(y축)에 따라 생성된 hierachy를 볼 수 있다.
(주의! 아직까지는 robust single linkage를 이용한 hierachy clustering의 방법까지만 설명했다. HDBSCAN안나왓다.)


```python
 clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
```

![output_14_1](https://user-images.githubusercontent.com/31824102/60085115-b37f2c00-9773-11e9-8e8f-1fa453b33ecd.png)



보통의 single linkage cluster는 여기에서 눈대중으로 괜찮아보이는 cut level을 정한다. (즉, distance를 정한다. 예를 들어 mutual reachability=0.3에서 평행선 쭉 그어서 cluster만들기)

그러나 우리는 variabel density cluster를 만들고 싶다. 어떤 클러스터에서는 distance=0.3이 큰 거리이지만 또다른 dense를 가진 클러스터에선 그다지 크지 않은 거리일 수도 있기에!
여기에서 HDBSCAN의 알고리즘이사용된다.

## Condense the cluster hierarchy based on minimum cluster size.

위의 그림에서 볼 수 있듯이, 또 쉽게 상상할 수 있듯이 threshold distance가 내려가면서 hierachy가 분할될때, 분할의 많은 경우가 데이터 1개, 2개가 떨어져나오는 경우들이 많아 지저분 하게 된다. (위 그림 기준으로는 dinstance 0.4이하로는 거의 다 데이터 한개가 떨어져 나오는 경우이다.) 이런 경우를 '2개의 클러스터로 나뉘어진것'으로 보지 않고, '한개의 클러스터가 데이터를 잃은 것'(이를 fell out으로 표현한다) 으로 치부한다. 마치 noise로 치부하는 느낌이다. (이때 필요한게 **minimum size**로, HDBSCAN의 하이퍼파라미터이다)

이렇게 쭉 내려가서, 최종적으론 **minimum size이상의 크기를 가진 component들**이 남게 된다. 이를 똑같이 덴드로그램으로 그린것이 아래그림이다. (선의 너비는 그 component에 포함된 데이터의 수)

```python
clusterer.condensed_tree_.plot()
```

![output_17_1](https://user-images.githubusercontent.com/31824102/60085117-b37f2c00-9773-11e9-8664-9fb8cec51d42.png)



## Extract the stable clusters from the condensed tree.
그러나 다한게 아니다. 이건 robust single linkage에서 noise같은 split을 처리해준거지, local density를 반영한게 아니다. 

결국 저거가지고, 클러스터를 만들어야 한다. 그림의 하단부분을 보면, 비록 minimum size를 만족하긴 했지만 아주 끄트머리가서 분할된 애들이 있다. 이들은, (원래 같은 클러스터의 데이터인데) 우연히 그 data들이 5개,5개씩 아주 살짝 더 뭉쳐있어서 떨어진것으로 볼 수도 있다. 즉, 우리가 찾는 이상적인 클러스터는 위의 그림에서 **오랫동안 지속되왔던 줄기들**이다. 이러한 직관을 수식으로 정리하여 cluster를 만들어준것이 HDBSCAN이다.

위의 덴드로그램에서 y축을 마치 위에서 부터 시작해서 아래로 내려오는 시간축 처럼 보자. y축을 따라 내려오면서, 클러스터가 분할되어 **새로운 클러스터가 생기고(birth)**, 그중 noise로 fell out 되는 데이터가 몇개씩 존재하다가, 결국 min_size이상의 **2개의 클러스터로 분할(death)**된다. (물론 구분안되고 끝가지 갈수도 있다.) (여기서 noise로 fell out 되는 것은 기준distance를 만족시키지 못하지만 또 min_size를 만족하지 못한 데이터들이다. min_size를 만족하는 데이터 군들의 edge가 끊길 경우, 이는 클러스터의 분할로 본다)

> 분할, 혹은 탄생의 시점을 재는 측도로써, (분할 혹은 탄생될 때의) distance가 아닌 $$\lambda$$를 사용하게 된다. 이때, $$\lambda$$는 mutual reachability distance의 역수이다.(즉 $$\lambda=\frac{1}{distance}$$) 
>
> local dense를 더 잘 표현해주기 위해 작은 distance값엔 민감하게 반응하고 큰 값엔 둔감하게 반응하게 하기 위해 distance자체가 아닌 역수로 사용한듯 하다.

여기서, 우리가 찾고싶은 클러스터는 **'오래동안 살은' 클러스터**이다. 이를 위해 각 cluster에 대해서, 그 클러스터가 탄생된 값을 $\lambda_{birth}$, 그 클러스터가 또다른 2개의 클러스터로 분할되는 시점을 $\lambda_{death}$라고 한다. 그리고 해당 클러스터내의 각 데이터에 대해서는, 그 데이터가 (만약 fell out 됬다면)  fell out되는 시점이 존재하는데 이를 $\lambda_{p}$라고 한다. (끝까지 남아있던애들은 당근 $\lambda_{p}=\lambda_{death}$)

그럼 오래 살은 안정적인 클러스터들은, 데이터들이 오밀조질 연결되어 별로 떨어지지(fell out되지) 않다가, threshold를 너무나 줄여 **억지로 떨어지는 클러스터들**일 것이다. 따라서 **cluster의 안정성(stability)**를 다음과 같이 정의한다
$$\sum_{p\in Cluster}(\lambda_p-\lambda_{birth})$$

1. 이를 각각의 component에 대해서 모두 실행한다. (parents, child 가릴것 없이 아래서부터 위로 올라오며 모두 계산한다.) 
2. 아래에서 위로 올라가며 두개의 child cluster가 parents로 합쳐질때마다, 그 parents의 stability와 두 child cluster의 stability의 sum을 비교한다. **child의 stability sum** 이 더 크면 2개의 child 를 클러스터로 유지하고, **paranets의 stability**가 더 크면 parents를 클러스터로 인정한다. 

이를 아래에서 위로 전부다 훑게 되면, 최종적인 variable dense를 반영한 cluster들이 남게 된다! 

이를 그림으로 한것이 아래와 같다(동그라미쳐진 애들이 살아남은 클러스터)


```python
clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
```

![output_20_1](https://user-images.githubusercontent.com/31824102/60085101-b11cd200-9773-11e9-868a-783821515947.png)


## 결과물 및 각 데이터의 probability
또한 부가기능으로 다음을 할 수 있다.
각 클러스터에 대해, 앞선 과정에서 구한 $\lambda_p$, 즉 클러스터에서 떨어져 나간 시점을 모아, (각 클러스터내에서) [0,1]에 오도록 scaling한다. 이 값이 큰 데이터들은 클러스터가 태어자마자마 fell out된 애들이므로, 이를 **'해당 클러스터에 속할 확률'**으로 해석할 수도 있다.(굳이. 좀더 있어보이려고)

아래의 그림은 이를 활용하여, 각 점들을 클러스터별 색깔로 표시하되 **속할확률이 작은 데이터들**은 그 색을 desaturate하여 회색에 가깝게 표현한 fancy한 그림이다.


```python
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.3,0.3,0.3) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_22_1](https://user-images.githubusercontent.com/31824102/60085102-b1b56880-9773-11e9-9470-465ee39f63af.png)


# DBSCAN과 비교

그럼 hdbscan이 실제 기존의 dbscan과는 얼마나 성능차이가 날까? 이를 위해 같은 예시데이터에 대해, sklearn의 dbscan과 비교해보았다.


```python
from sklearn.cluster import DBSCAN
plt.rcParams["figure.figsize"] = [6,5]
db = DBSCAN(eps=0.2, min_samples=10).fit(test_data)
```


```python
palette = sns.color_palette()
cluster_colors = [palette[col]
                  if col >= 0 else (0.5, 0.5, 0.5) for col in
                  db.labels_]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_26_1](https://user-images.githubusercontent.com/31824102/60085103-b1b56880-9773-11e9-91a0-151d5ab07ca4.png)


반달형보다는 오히려 타원의 밀집한 부분만을 잡아낸다. DBSCAN은 **epsilon안의 (절대적인) 데이터수**로 따지니까 타원형태가 더 유리할것같다.

이번엔, 하이퍼파라미터 eps를 좀더 조정해보자. 


```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.35, min_samples=10).fit(test_data)
```


```python
palette = sns.color_palette()
cluster_colors = [palette[col]
                  if col >= 0 else (0.3,0.3,0.3) for col in
                  db.labels_]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_29_1](https://user-images.githubusercontent.com/31824102/60085104-b1b56880-9773-11e9-9e10-7679b7097149.png)


모든 클러스터를 잡아내긴 하지만, 역시나 **variable dense에 제대로 적응하지 못하고** 몇개의 데이터를 noise라고 잡아낸다. 이는 각 데이터가 다른 클러스터에 비해서는 멀리떨어져 있는데, 이를 **절대적인 epsilon**으로 잡아내려하면서 생긴 문제라고 할 수 있다

# DBSCAN과 비교2

좀더 극단적인 variable dense의 case로 다시한번 비교를 시도해보자


```python
num=500
#moons, _ = data.make_moons(n_samples=num, noise=0.1)
blobs, _ = data.make_blobs(n_samples=num, centers=[(-5,20.25), (5.0, -20.0)], cluster_std=0.25)
blobs2, _ = data.make_blobs(n_samples=num, centers=[(2,2.25)], cluster_std=4)
test_data = np.vstack([ blobs,blobs2])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
plt.show()
```


![output_32_0](https://user-images.githubusercontent.com/31824102/60085105-b1b56880-9773-11e9-9d50-3019ba52e6b8.png)

가운데의 **매우 sparse한 집단**과, 위아래로 **매우 dense한 집단**이 2개 있는 데이터이다.


## DBSCAN의 경우


```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=10).fit(test_data)
```


```python
palette = sns.color_palette()
cluster_colors = [palette[col]
                  if col >= 0 else (0.5, 0.5, 0.5) for col in
                  db.labels_]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_35_1](https://user-images.githubusercontent.com/31824102/60085106-b24dff00-9773-11e9-983b-56807729efdf.png)

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.4, min_samples=10).fit(test_data)
```


```python
palette = sns.color_palette("husl", 12)
cluster_colors = [palette[col]
                  if col >= 0 else (0.3,0.3,0.3) for col in
                  db.labels_]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_37_1](https://user-images.githubusercontent.com/31824102/60085107-b24dff00-9773-11e9-9465-bb9a126d144b.png)


같은 setting하에서, variable dense를 잘 못잡는다. 그럼 파타미터 튜닝을 해볼까?


```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=1, min_samples=10).fit(test_data)
```


```python
palette = sns.color_palette("husl", 12)
cluster_colors = [palette[col]
                  if col >= 0 else (0.3,0.3,0.3) for col in
                  db.labels_]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_40_1](https://user-images.githubusercontent.com/31824102/60085109-b24dff00-9773-11e9-833e-ed890ed3a657.png)


결과는 하나만 제시했지만, 여러값을 시도해보았을때 저게 최선이었다. 한눈에 봐도, variable dense를 잡지못하고 고정된 **eps안에서 잡으려 하기에** 차이가 있음을 알 수 있다. 

## HDBSCAN의 경우

아래는 파라미터를 바꿔 시도해본 HDBSCAN이다.


```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.3,0.3,0.3) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_43_1](https://user-images.githubusercontent.com/31824102/60085110-b2e69580-9773-11e9-83c9-d10c3497fd24.png)




```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
clusterer.fit(test_data)

palette = sns.color_palette("husl", 12)
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.3,0.3,0.3) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
```

![output_44_1](https://user-images.githubusercontent.com/31824102/60085112-b2e69580-9773-11e9-881e-ccebaefae859.png)

서로 다른 두 parameter의 경우에서, 모두 variabel dense를 잡아낸것을 볼 수 있다!

---

참조 

파이선 HDBSCAN 패키지 : <https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html>

참조 코드 원문 : https://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb#How-HDBSCAN-Works

