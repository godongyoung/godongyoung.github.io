## Memory Networks 논문 리뷰

[TOC]

논문 : https://arxiv.org/abs/1410.3916  (2014)

## Abstract

새로운 학습방식인 memory network를 제시한다. memory network는 long-term memory요소와 inference 요소(뭘까...QA에서는 질문인듯. 어떤 task에서 추론을 위한 요소인듯. 질문이면 input도 아니니까, 이를 inference component라고 표현하는 듯 하다.)를 통해 reason(판단)을 한다. long-term memory가 효과적으로 작동하는 Question answering task에 적용하였다. 

굉장히 범용적인 개념인데다 여러가지 적용을 해본거라, 설명도 범용적이다.. 

## 1. Introduction

대부분의 모델들은, 특히 input이 매우 길어질때 long-term memory component를 잡아내고 이를 inference와 매끄럽게 결합하는것을 잘 못한다.  예를 들어, 하나의 story를 듣고 question에 답하는 경우를 생각해보자. 이론상으로 RNN과 같이 이전의 흐름들을 읽고 다음 word를 예측하는 모델들이 이를 해낼 수 있다. 그러나 하나의 hidden state와 weight로 이루어진 그들의 memory능력은 매우 작다. 과거의 내용들이 fixed vector로 압축되어 표현되어야 하기에, 결국 과거의 정보를 제대로 구분해내지 못한다. 이에 따라 RNN은 들어온 input seq를 그대로 내뱉는 간단한 task에서 약점을 보인다.(!) 이는 text뿐아니라 audio등 log term memory가 필요한 어느곳에서도 동일하다.

우리는 이러한 문제를 해결할 수 있는 memory network를 제시한다. 주요 아이디어는 기존의 머신러닝 방법으로 구해진 inference에 읽고 쓰여질 수 있는 memory component를 결합하는 것이다. 이를 어떻게 잘 결합할 수 있는지를 학습한다. 섹션2에서 구체적인 frame work를 제시하고 센션3에서 QA task에 대해 실행해본 결과를 제시할 것이다. 

## 2. Memory Networks

메모리 네트워크는 $$m_i$$로 index가 되어있는 '메모리$$\boldsymbol m$$'과, 학습되는 4개의 요소 '$$I,G,O,R$$'로 이루어진다.

- $$I$$ : input feature map. input을 feature representation으로 바꿔준다
- $$G$$ : generalization. 새로운 input이 들어오면 이를 토대로 이전 memory를 update해준다.
- $$O$$ : output feature map. 새로운 input과 현재의 memory를 토대로 새로운 output(feature representation)을 만든다
- $$R$$ : response. output(feature representation)을 원하는 형태로 변환시킨 뒤 response를 한다.

구체적으로, input $$x$$가 들어왔을때 모델의 작동방식은 다음과 같다.

1. $$x$$를 feature representation $$I(x)$$로 바꾼다

2. 해당 input을 토대로 memory를 update한다. $$m_i=G(m_i,I(x),\boldsymbol m), \forall i$$

   > (메모리와 메모리의 update방식은 말그대로 아무거나 가능하다. i번째 인풋을 i번째 메모리에만 넣어줄 수도 있고, i번째 메모리는 이전0~i까지의 input을 반영할수도 있고(이경우 attention을 활용한 RNN의 느낌적인 느낌), 혹은 input이 모든 slot의 메모리에 영향을 줄수도 있을 것이다.)

3. 새로운 input($$I(x)$$)와 메모리($$m$$)를 토대로 output feature $$o$$를 계산한다.  $$o=O(I(x),\boldsymbol m)$$

4. output feature $$o$$를 response로 디코딩한다. $$r=R(o)$$

물론 test 과정에서는 memory는 저장되지만 I,G,O,R은 update되지 않고 유지된다. 이때 I,G,O,R은 머신 러닝에서 이미 존재하는 어떤것도 사용가능하다. (SVM, Decision tree, etc)

$$I$$ component : I는 parsing, entity resolution([참고](http://www.datacommunitydc.org/blog/2013/08/entity-resolution-for-big-data))과 같은 기존의 pre-processing과정일 수도 있고, input을 internal feature representationd로 인코딩하는과정 역시 사용될 수 있다.

$$G$$ component :  G의 가장 간단한 form은 $$I(x)$$를 메모리의 한 슬롯에 저장하는 것이다. 
$$
\boldsymbol m_{H(x)}=I(x)
$$
여기서 H(.)는 슬롯의 한부분을 지칭하는 func을 의미한다. 즉 (가장 간단한)G는 $$\boldsymbol m$$의 H(x)부분만을 업데이트해주고 나머지 memory는 건들지 않는 함수이다. 더욱 복잡한 G는 새로운 input x에 기반하여 이전메모리(혹은 전체 이전메모리)를 업데이트해줄 것이다. 만약 input이 character나 word level이라면 이들을 group해서 memory slot에 넣어줄 수 있을 것이다 .

만약 memory가 매우 거대, 즉 wikipedia같이 거대한 input이 들어온다면, 메모리를 organize할 필요가 있는데, 이역시 indexing 을 해주는 H(.)으로써 해결할 수 있다. 예를 들어 entity에 따라서 같은 slot으로 취급하도록 design(혹은 그런방식으로 train)하는 것이다. 결과적으로 scale에 대해서 효율적이려면 G(그리고 뒤의 O도) 가 모든 memory를 대상으로 하는 것이 아니라 선정된 subset, 즉 유의미한 topic의 memory에 대해서만 작동해야 한다. 뒤의 실험에서 이를 다룬다

$$O,R$$ component : O component는 memory를 읽고 어떤 memory가 관련되어있는지를 계산한다. R component는 이를 통해 나온 O를 토대로 최종 response를 한다 R은 output O에 대한 RNN decoder일 수 있다. 우리의 가정은 'memory에 대한 선택이 없다면 이 RNN의 성능이 좋지 못할 것이다'이다. 

## 3 A MemNN Implementation for Text

메모리 network를 neural network에 적용했다해서 MemNN. text로 된 input, output을 가지고 간단한 implementation을 해보았다.

### 3.1 Bacis Model

I가 input을 받아 embedding을 한다. 이 input이 fact를 나타내는 문장과 질문을 하는 문장이라 해보자.(뒤에 input이 word-based sequence인것도 다룬다). 가장 간단한 G를 생각해보면, 임베딩된 text가 다음 메모리 슬롯에 저장될 것이다. 즉 새로운 메모리 슬롯만을 업데이트하고 이전 메모리는 건들지 않는것이다. 좀더 복잡한 방식의 G는 다음 섹션에서 다룬다.

inference의 핵심은 O,R에 있는데, O는 input x에 대해 유의미한 k개의 메모리를 선정하여 output feature을 만든다. 우리는 k를 2로 선정하였다. 즉 여러 문장과 질문이 input으로 들어온다면, 그 질문에 도움될만한 2개의 메모리, 여기서는 한문장->한 메모리 슬롯이니 2개의 문장을 고르는 것이다. 

k=1일때, 가장 관련잇는 memory(supporting memory라 표현한다) $$o_1$$은 다음과 같이 정의할 수 있다.
$$
o_1=O_1(x,\boldsymbol m)=argmax s_O(x,m_i), \forall i
$$
이때 $$s_O$$는 input sentence(여기서는 예시가 문장이었으니)x와 $$m_i$$ 쌍의 match를 점수매기는 함수이다. 어떻게 이를 계산하는지는 조금 뒤에 설명한다.

k=2일 경우, 2번째 관련있는 memory $$o_2$$는 다음과 같다.
$$
o_2=O_2(x,\boldsymbol m)=argmax s_O([x,m_{o_1}],m_i), \forall i
$$
즉 다음 관련잇는 memory는 이전 메모리들과 input과의 **list**와의 match로써 평가한다.(bag-of-word model을 사용했다. 즉, 단순히 $$s_O(x,m_i)+s_O(m_{o_1},m_i)$$을 최대화 하는 값을 찾는것과 같다. 다른 형태로 고려할수도 있을 것이라고만 언급햇다..) 이 경우 $$R$$로 들어가는 최종 output은 $$[x,m_{o_1},m_{o_2}]$$이다. 

이를 통해 최종 $$R$$이 response를 만든다. 이때 RNN을 사용할 수 있을 것이다. 또한, single word으로 response가 제한되는 경우(예를 들어 Q:사과는 어딧어? A:부엌.)에 대해서 $$[x,m_{o_1},m_{o_2}]$$의 조합을 평가하기 위해서 다음과 같은 **rank**를 사용할 수 있을 것이다.
$$
r=argmax_{w\in W}s_R([x,m_{o_1},m_{o_2}],w)
$$
$$W$$는 dictionary에 있는 모든 단어이고, $$s_R$$은 match를 점수매기는 함수이다. 결국 앞에 했던 $$S_O$$와 비슷하게, dic에 있는 모든 단어 중 list $$[x,m_{o_1},m_{o_2}]$$에 가장 적절하게 매칭될만한 단어를 찾는 것이다. 

![memory1](https://user-images.githubusercontent.com/31824102/37249968-efdb67b6-24e9-11e8-8356-14786fabebfb.PNG)

다음은 예제질문이다. input question $$x$$='Where is the milk now?'였을때 $$O$$는 이에 대해 모든 memory를 점수매긴다. 이 경우 input이 sentence였으니, 지금까지 들어온 sentence들과 input $$x$$와 가장 관련있는 메모리를 $$m_{o_1}$$으로써 내보낼 것이다. (ex $$m_{o_1}$$='Joe left the milk') 그리고 $$[x,m_{o_1}]$$ 에 대해 가장 관련있는 $$m_{o_2}$$를 찾을 것이다. 예를들어 $$m_{o_2}$$='Joe travelled to the office'(우유를 방치하기 전 가장 최근에 간 장소) 따라서 $$R$$은 $$s_R([x,m_{o_1},m_{o_2}],w)$$에 따라 'office'를 출력할 것이다.

그럼 적절한 match를 평가하는 적절성 평가함수 $$s_O,s_R$$은 어케 정의하느냐, 우리는 두 함수에 대해 다음과 같이 동일한 정의를 사용하였다
$$
s(x,y)=\Phi_x(x)^TU^TU\Phi_y(y)
$$
이때 $$U$$는 n X D의 embedding matrix이다. n은 임베딩 dimension, D는 feature의 수를 의미한다. $$\Phi_x,\Phi_y$$는 D차원 feature space로 매핑된 text의 embedding feature representation이다. 가장 간단한 매핑은 bag of words이다. 이 경우 $$D=\lvert W\lvert$$. 우리는 dictionary내의 모든 word가 3개의 representation을 갖는다는 의미에서 $$D=3\lvert W\lvert$$를 사용하였다. ($$W$$는 dictionary에 있는 모든 단어)

> 결국 x와 y의 내적인데...내적으로 유사도를 구하는것과 비슷한 개념으로 이해하였다.

#### Training

train은 fully supervised방식으로 하였다. 즉, input에 대한 desired reponse가 있고, 그 input에 대한 관련있는 **메모리sentence역시 라벨링** 되어있다. 즉, 다음의 최적값을 알고있다

$$o_1=O_1(x,\boldsymbol m)=argmax s_O(x,m_i), \forall i$$

$$o_2=O_2(x,\boldsymbol m)=argmax s_O([x,m_{o_1}],m_i), \forall i$$

training은 **margin ranking loss**와 SGD를 통해 이루어 진다. 즉 $$x$$와 그에 대한 true response $$r$$, supporting sentence $$m_{o_1},m_{o_2}$$를 알고 있을때 Loss func은 다음과 같다.

![memory2](https://user-images.githubusercontent.com/31824102/37249975-f173b60a-24e9-11e8-90ed-59a47cae8675.PNG)

이때 $$\bar f,\bar {f'}, \bar r$$은 모두 잘못된 답이다. 즉, 실제답과의 차이가 각각 모두$$\gamma$$이상 만큼 나는것을 목표로 하는 것이다. 실제 답에 대한 score를 더 키우고자 학습하려 할 것이다. 이를 최소화하도록 GD를 사용하여 $$s_O$$와 $$s_R$$의 parameter  $$U_O,U_R$$을 조정한다. SGD를 사용하여 모든 traing set에 대해 $$\bar f,\bar {f'}, \bar r$$를 계산한게 아니라 sample로써 $$\bar f,\bar {f'}, \bar r$$를 계산했다.

또한 RNN을 사용한 $$R$$단계에서는 언어모델에서 자주 사용하는 log likelihood을 사용하였다. (???여러분의 도움이 절실합니다. we replace the last term with the standard log likelihood used in a language modeling task, where the RNN is fed the sequence [x, o1, o2, r].) test단계에서는, [x,o_1,o_2]이 주어졌을때 max likelihood인 r을 반환한다.

다음의 섹션에서 좀더 확장된 모델을 다룬다.

### 3.2 Word Sequence as Input

input이 sentence level이 아닌 word level이고, fact statement와 question이라는 segmentation이 되지 않은 경우를 생각해보자. 즉 QA에선, 누가 사실에 대한 '설명'인지, 누가 '질문'인지도 모르는 경우다.이 경우 segmentation function을 추가로 학습시켜야 한다. 즉 segmented되지 않은 word들의 seq들의 끝을 계속 input으로 받아 break point를 찾아내는 함수이다. segmentation이 발생하여 지금까지의 seq가 segment라는 신호가 오면 이 seq를 memory에 저장한다. segmentatoin func는 다음의 형식을 취한다
$$
seg(c)=W^T_{seg}U_S\Phi_{seg}(c)
$$
$$U_S\Phi_{seg}(c)$$는 이전에 나왔던 형태로, feature map과 embedding matrix이다 . $$W_{seg}$$는 embedding space를 classify할 수 있는 vector이고, $$c$$는 bag of word로 표현된 seq of input words이다. 저 classifier를 어떻게 학습햇는지는, 다음과 같다.

fully supervised setting이기에, 역시나 supervised 학습을 할 수 있다. 예를 들어 'Where is Bill?'이라는 Q에 대해 'Bill is in the Kitchen'이 supporting seq라는 것을 아니까, 해당 seq에서 fire를 하는 것이다.('Bill is in' 같이 unfinished문장에서 fire를 안하고)

![memory8](https://user-images.githubusercontent.com/31824102/37249969-f015c78a-24e9-11e8-92bf-c8ba17a3e766.PNG)

모든 supporting segment($$f$$)에 대해선 $$\gamma$$ 이상의 확신을 하고, unsupporting($$\bar f$$)에 대해선 $$-\gamma$$이하를 반환해야 하는 loss func이다.

만약 특정 margin $$\gamma$$에 대해 $$seg(c)>\gamma$$라면 이seq $$c$$를 segment로 간주한다.

이후의 과정은 같다. 

## 3.3 Efficient Memory via Hashing

만약 메모리가 매우 크다면 $$o_1=O_1(x,\boldsymbol m)=argmax s_O(x,m_i), \forall i$$와 같은 형태로 메모리를 선택하는 것이 expensive할 것이다. 따라서 우리는 lookup과정에서 hassing trick([참고](https://en.wikipedia.org/wiki/Hash_table#Collision_resolution))를 사용할 것이다. input $$I(x)$$를 하나 혹은 여러개의 bucket에 hashing을 한다.(???) 그 후 same bucket에 있는 메모리 $$m_i$$에 대해서만 score를 매긴다. hasing은 다음 2가지 방법으로 한다. 1) hashing word, 2) clustering word embedding. 1번의 경우, word dictionary만큼의 bucket을 만든다. 그리고 input sentence를 그것과 관련있는 word의 bucket에 hashing한다. 그러나 1번의 문제는 $$I(x)$$에 한번이라도 등장을 해야 해당 메모리가 고려될 것, 즉 매우 sparse하다는 것이다. 2번은 이를 클러스터링으로써 보완해준다. 임베딩 메트릭스 $$U_O$$를 학습시키고 난 후, 워드벡터$$(U_O)_i$$들을 K-means 클러스터링 해준다. 즉, K개의 bucket으로 분류한다. 그리곤 given sentence가 해당 bucket의 word를 포함하고 있는 경우 각 bucket에 hash한다. 비슷한 단어의 word vector는 함께 클러스터링  될것이므로 memory도 함께 처리할 수 있다. 이때 K를 정하는 것은 speed-accuracy trade-off의 문제이다

## 3.4 Modeling Write Time

여기부터 설명이 점점 이상해진다.

우리의 모델을, '언제' 메모리 slot이 쓰여졌는지를 고려할 수 있도록 확장할 수 있다. 이는 '프랑스의 수도는 어딘가요?'와 같은 fixed fact에 대해 답할때는 중요한것이 아니지만, 앞의 milk예시 처럼 story에 관한 질문일 경우 중요한 요소가 된다. 

경험적으로 우리는 다음과 같은 해결책을 찾았다. 기존의 scoring function $$s_O(x,y)=\Phi_x(x)^TU^TU\Phi_y(y)$$의 형식에서, 다음과 같은 3개의 인자를 받는 함수로 변화하였다.
$$
S_{O_t}(x,y,y')=\Phi_x^TU_{O_t}^TU_{O_t}(\Phi_y(y)-\Phi_y(y')+\Phi_t(x,y,y'))
$$
앞에는 똑같은 형식인데,(물론 U는 위에랑 아래가 같은애는 아니다) 뒤의 형식이 다르다. 이때 y'는 또다른 memory로 이해할 수 있다. $$\Phi_t(x,y,y')$$는 0,1의 값을 가지는 3개의 feature를 사용하는데, 각각 x가 y보다 큰지, x가 y'보다 큰지, y가 y'보다 큰지를 나타낸다. (즉 다른 $$\Phi$$들도 3차원이 되고, 이 $$\Phi_t(x,y,y')$$를 사용하지 않을때는 전부 0값이 된다). 만약 $$S_{O_t}(x,y,y')>0$$이라면 모델이 y를 더 선호하게 되고 $$S_{O_t}(x,y,y')<0$$이라면 모델이 y'을 더 선호하게 된다. 이렇게 해당 식을 모든 메모리i ($$i=1,...,N$$)에 대해 반복하여 winning memory를 계속 갱신한다. (???Appendix C를 보자)

## 3.5 Modeling Previously Unseen Words

사람 역시 새로운 단어를 접하게 된다. 예를 들어 반지의 제왕에서 'Boromir'라는 단어를 처음 접하게 되었을때, 언어모델이 이를 어떻게 처리해야할까? 이상적으로는 하나의 예문만을 보고 동작을 해야할것이다. 가능한 한가지 방법은 주변의 단어들로 그 위치의 단어를 예측하고, new word가 그와 비슷한 단어일 것이라 생각하는 것이다. 우리는 이 아이디어를 우리의 네트워크 $$S_O, S_R$$과 결합하였다. 

구체적으로 input으로 들어오는 모든 단어에 대해 함께 등장하는, 즉 왼쪽의 단어와 오른쪽의 단어를 bag of word로 저장하였다. unknown word도 이런 feature들로 표현될 수 있다. 따라서 feature representation D를 기존의 $$3\lvert W\lvert$$에서 $$5\lvert W\lvert$$로 확장하였다. (각각 unknown bag에 대해 $$\lvert W\lvert$$이 추가되었다). training에서, 각 step의 d%마다 droupout의 형태로 해당 단어를 n차원 임베딩 단어가 아닌 unknown단어라 치부하여 context를 통해 대신 표현하였다. 이런 방식으로 new word를 다루는 방식을 학습하게 하였다....$$S_O$$의 phi를 5W로 확장했다는 건 알겠는데.. 그래서 결국 missing 3W는 어떻게 한건지....?

## 3.6 Exact Matches and Unseen Words

임베딩 모델은 n차원이라는 적은 차원으로 인해 word match를 효과적으로 할 수 없다. 하나의 해결책은 x,y의 pair를 점수매기는 것이다
$$
\Phi_x(x)^TU^TU\Phi_y(y)+\lambda\Phi_x(x)^T\Phi_y(y)
$$
즉 bag of word끼리의 매칭 score를 $$\lambda$$와 함께 추가하는 것이다. (이게 왜 효과가 있는지는 잘..설명조차 없...)

또다른 방법은 n차원을 그대로 유지하지만 feature representation D를 matching feature로 확장하는 것이다(D는 feature의 수를 의미한다)matching feature란 단어가 x와 y에 동시에 등장한 것을 의미하는 것이다. 메모리y의 어떤 단어가 x와 겹친다면 그 matching feauture를 1로 만든다. (결국 위의 feature map들끼리의 내적과 거의 같은 의미인듯하다) 즉, x에 대해 conditon이 부여되어 구성된 $$\Phi_y$$를 통해  $$\Phi_x(x)^TU^TU\Phi_y(y,x)$$로써 match score를 매기는 것이다. unseen word역시 비슷한 방식으로 모델링될 수 있다. 이는 feature space를 $$8\lvert W\lvert$$로 키운다. $$\lvert 5W\lvert$$에 기존 representation의 크기인 $$3\lvert W\lvert$$이 추가되서 인듯..하다

## Experiment.

생략. 부가적인설명없이 표랑 잘 답변한 예시만 다닥다닥... RNN, LSTM보다 잘했다뿐...

## Conclusion and Future work

MemNN for text를 더 발전시켜야 한다. 더 어려운 QA나 open-domain에 대해작동하는것 등. QA를 위한 추론을 위해서 multi-hop을 해야하는 task, 더 많은 구조와 더 많은 동사, 명사를 가진 복잡한 data에 대한 작동. 대부분의 data set이 Q,A만 있고 우리가 한것 처럼 supporting fact에 대한 라벨은 없으므로 보다 완화된 supervised도 고려되어야 한다. 이는 우리가 소개한 MemNN보다 더 뛰어난 형태일 것이며 이밖에도 여러 variant들이 있을 수 있을 것이다. 

---

참고 : 인공지능을 위한 머신러닝 알고리즘 11강 메모리 네트워크 | T아카데미. 설명짱짱 친절하다. end-to-end MMn에 대한 설명도 함께 한다.

https://www.youtube.com/watch?v=vDQf7lcenfI