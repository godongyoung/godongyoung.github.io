---
layout: post
title: "[NLP]  Attention Is All You Need 논문리뷰"
categories:
  - 딥러닝
tags:
  - Deep Learning
  - NLP
comment: true
---

{:toc}

논문 : https://arxiv.org/abs/1706.03762

### Abstract

현재 대부분의 sequence model은 cnn이나 rnn을 encoder, decoder로써 활용하고 있다. 그 중 가장 좋은 성능을 보이는 모델은 attention mechanism을 활용한 encoder, decoder모델이다. 우리는, cnn과 rnn을 없애고 attenton에만 기반을 둔 단순한 network인 Transformer를 제안한다. 이를 통해 paralleizable이 가능해졌고, train 시간을 대폭 감소시켰다. 2개의 machine translation 실험을 해본결과 성능을 보였다.

### 1. Introduction

RNN, 특히 LSTM은 sequence model과 machine translation에서 이미 state of art로 자리잡았다. 

RNN모델은 input과 output sequence의 position들을 계산하는데 뛰어나다. 이 positoin대로 순서대로 연산을 하면서, 이전 hidden state $$h_{t-1}$$과 그 position에서의 input $$t$$를 통해 새로운 hidden state $$h_t$$를 만들어 낸다. 따라서 구조상 sequential 한 특성을 가지고 있기에, 이는 parallelization에 취약하다는 단점이 있다. 이는 sequence길이가 길어진 경우 batch로써 풀고자 할때 큰 문제가 된다. (parralelization이란? [참고](https://www.computerhope.com/jargon/p/parallelization.htm))

한편 attention mechanism은 input output seq의 길이에 관계없이 dependency를 모델링할 수 있게 해줌으로써 seq modeling에 필수적인 요소가 되었다. 그러나 대부분 이 방법은 아직 rnn과 결합되는 형태로 쓰이고 있다. 

우리의 모델인 Transformer는 대신 attention mechanism에 전적으로 기반한다. 이를 통해 input과 output의 global dependency를 잡아낸다. 이는 학습속도도 빠르고 parralelization도 수월하다.

### 2. Background

sequential한 연산을 줄이고자 하는 목표로 ByteNet등이 생겨났는데, 이들은 모두 hidden representation을 parallel하게 계산하고자, CNN을 활용한다. 그러나 이들은 2개의 positoin 상 멀리 떨어져있는 input output을 연결하는데에 많은 연산을 필요로 한다(number of operation required). 따라서 distant position에 있는 dependency를 학습하기에는 힘들다. Transformer에서는 attention-weighted position을 평균취해줌으로써 effective는 잃었지만,  이 operation이 상수로 고정되어있다. effective에 대해선 Multi-Head Attention으로 이를 극복한다.(section 3.2)

Self-attention은 seq representation을 얻고자 한 sequence에 있는 다른 position을 연결해주는 attention기법이다. 이는 지문이해나 요약등의 과제에서 다양하게 활용되고 있다.

그러나 RNN이나 CNN없이 self-attention만으로 representation을 구한 모델은 우리 Transformer가 처음이다.

### 3. Model Architecture

가장 뛰어난 sequence model은 encoder-decoder구조를 활용한다. encoder가 symbol representation($$x_1,..,x_n$$)을 가지고 있는 input sequence를 연속적인 representation ($$z_1,..,z_n$$)으로 바꿔준다. 그리고 그 z를 가지고 decoder가 순차적으로 symbol을 가진 output sequence ($$y_1,..,y_m$$)을 만들어낸다. 이때 symnol을 만드는 과정은 auto-regressive한데, 각 $$y_i$$를 만들어내는 단계에서는 이전의 만들어진 symbol도 input으로 사용한다는 의미이다.

Transformer는 다음의 구조를 가지는데, self-attention의 stack버젼과 인코더 디코더 모두 fc layer를 가진 형태이다.

![attention1](https://user-images.githubusercontent.com/31824102/36531999-78eda612-17b7-11e8-9793-ef8dcd78dd9b.PNG)

#### 3.1 Encoder and Decoder Stacks

**Encoder**: 인코더는 6개의 동일한 레이어를 stack하였다.(stack하였다는 개념이 정확히 멀까..???ㅠ다시보자ㅠㅠ) 각 레이어는 2개의 sub-layer를 가지고 있다. 하나는 multi-head self-attention mechanism이고, 하나는 position별로 fully connected된 단순한 feed-forward network이다. 각 sub-layer에는 residual connection을 연결하였고, 이후 normalization을 하였다. 따라서 최종 output은 NormLayer($$x+$$Sublayer($$x$$))의 형태이다. skip connection을 수월하게 해주기 위해 sub-layer와 embedding layer는 모두 output차원 512를 갖도록 하였다.

**Decoder**:  디코더 역시 6개의 동일한 레이어를 stack하였다. 그러나 2개의 sub-layer외에도 또 다른 sub-layer를 추가하였는데, stack된 encoder의 output에 multi-head attention을 하는 layer이다.(띄용???output이 decoder의 input아니야??? 그름 두개 같은거 아닌가...???) 역시나 각 sub-layer에 residual connection을 해주었고 layer normalization을 해주었다. 또한, 각 position이 뒤따라 오는 position에 attend하는 것을 방지하기 위해 디코더의 self-attention을 약간 수정하였다. 이는 i번째 position의 prediction이 i번째 이전의 output에만 의존할 수 있도록 만들어준다...?

### 3.2 Attention

attention은 query와 key, value의 pair를 토대로 output을 만들어 반환해주는 함수이다. (어느 key에 attention을 할지를 의미하는 것인듯 아닌듯!) 여기서 output은 value의 weight sum인데, weight는 해당 key의 compatibility function을 통해 계산된다. 이 compatibility function에 대해 바로 이어 설명하는듯

#### 3.2-1 Scaled Dot-Product Attention

우리의 attention은 Scaled Dot-Product Attention이다. Attention의 대표적인 방법으로는 additive attention과 dot-product attentino이 있는데, 우리는 dot-product에서 $$\sqrt{d_k}$$로 scaling해준것만 다르다. additive는 single hidden layer를 가진 nn을 이용하는데, 이론적으로 둘이 비슷하지만 dot-product가 더 빠르고 효율적이다. 우리의 attention은 다음과 같다.

여기서 input은 query와 key, value로 이루어져 있다. 

어떤 부분에 attention을 할지, 즉 weight를 정하기 위해선 다음의 과정을 거친다. Query와 Key를 내적을 하고, 이를 key의 차원$$d_k$$의 제곱근으로 나누어줘 scaling을 해준다. 최종적으로 이를 softmax를 통하여, value에 곱해줄 weight를 구한다. 실제로는 여러개의 Query와 그에따른 key, value를 묶어서 각각 Q,K,V라는 matrix를 만들고 이를 matrix계산을 한다. 즉, 다음과 같다.
$$
Attention(Query,Key,Value)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
그림으로 나타낸거

![attention2](https://user-images.githubusercontent.com/31824102/36532004-79a6aa5e-17b7-11e8-9c9f-10e935136036.PNG)

#### 3.2-2 Multi-Head Attention

여기부터 더더 이해가 안되는데ㅠㅠㅠ query와 key, value를 $$d_k$$차원에 linear projection을 시켜서 각각의 projected된 버젼의 h개의 queriy, key, value를 가지고 h번의 attention을 한다. 그리고 이 h개를 다시 concat하고 다시 project을 시켜 최종값을 얻는다...이렇게 함으로써 각 position의 다른 subspace에서의 representation 정보 역시 잡을 수 있게 된다.

![attentin3](https://user-images.githubusercontent.com/31824102/36532003-79793f7e-17b7-11e8-9110-c8d793db4a2c.PNG)

여기서 h=8으로 하였다. 

#### 3.2-3 Applications of Attention in our Model

Transformer는 multi-head attention을 3가지 방법으로 사용하였다.

1. encoder-decoder attention에서, 쿼리는 이전 디코더 레이어에서 온것이고, key, value는 인코더의 output에서 온것. 이렇게 함으로써 input sequence의 모든 positon에 decoder가 attend할수 있다.
2. 인코더는 self-attention layer를 포함하고 있다. self-attention layer에선 key, value, query가 모두 인코더의 이전레이어에서 온것이다. 인코더에서 각 position은 이전 인코더 레이어의 모든 position에 대해 attend할 수 있다
3. 디코더에도 위와 같은 self-attention layer가 있다. Auto regressive의 성질을 유지해주기 위해 left방향의 information은 막았다.(???뭔말)


### 3.3 Position-wise Feed-Forward Networks

attention sub-layer말고도 인코더 디코더는 모든 position에 동일하게 적용되는 fc도 있다. 이것은 Relu를 포함한 2개의 선형변환으로 이루어져 있는데, 식은 이렇다
$$
FFN(x)=max(0,xW_1+b_1)W_2+b_2
$$
물론 각 layer마다 파라미터는 다르다. 이걸 표현하는 또 다른 방식은 kernel size1의 2 convolution이다. input과 output차원은 512이고, hidden layer의 차원은 2048이다.

### 3.4 Embedding and Softmax

대부분의 seq transduction model들과 같이 input output token을 벡터로 만들어주는데 learned embedding을 사용하였다. decoder를 통해 나온 representation은 FC와 softmax를 거쳐 다음token의 probability로 나온다. 

### 3.5 Positional Encoding

우리 모델이 rnn이나 cnn이 안들어가있기에, sequence order정보를 이용할 수 있도록 상대적인, 혹은 절대적인 position 정보를 넣어줘야 한다. 이를 위해 인코더 디코더의 bottom의 inpput embedding에 'positional encoding'을 추가하였다. positional encoding은 임베딩 차원과 동일하며 따라서 합쳐질 수 있다. 

positional encoding을 할 수 있는 방법은 많으나 우리는 cosine function을 사용하였다
$$
PE_{(pos,2i)}=sin(pos/10000^{2i/d_k})
$$

$$
PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_k})
$$

pos는 position이고 i는 차원이다.(???웬 차원) 즉 각 positional encoding의 차원이 sin곡선을 가진다는 것이다. 이렇게 함으로써, $$PE_{pos+k}$$가 $$PE_{pos}$$의 linear funtion이 되므로 relative position의 정보를 배울 수 있을 것이라 가정하였다. 또한 이렇게 함으로써 training때 없었던 길이의 sequence가 들어와도 잘 작동할 수 있을 것이라 생각하였다.

### 4. Why Self-Attention

여기서는 sequence tranduction을 하는 인코더 디코더에서 왜 symbol representation을 어쩌구 representatin으로 바꿔주기 위해 self-attention을 사용하였는지를 rnn과 cnn과 비교하여 설명하겠다. 우리는 다음 3가지를 고려하였다.

첫째, 레이어마다의 연산복잡도. 얼마나 연산이 parallelized될 수 있을지. 세번째는 path길이에 따른 long-range dependency. long-range dependency는 여러 seq transduction의 성능에서 중요한 요소이다. 각 position들간의 path가 짧아질수록 long-range dependency가 가능해지기에, 여러 network간의 최대 path를 비교해봤다.

![attention4](https://user-images.githubusercontent.com/31824102/36532001-794c50a4-17b7-11e8-99ab-4174fd3d4476.PNG)

표에도 나와있듯이 self-attention이 모든 position을 동일한 수(costant)의 sequential operation으로 연결하고 있다. 반면 RNN은 이를 못한다. layer당 복잡도는 sequence length $$n$$이 representation dimension $$d$$보다 작을때 더 빠른데, 대부분의 sota 모델은 n<d이다. 

매우 긴 sequence에 대해서 연산을 향상시키기 위해 self-attention을 neighborhood$$r$$에 한해서 하도록할 수 있는데, 이경우 path 길이가 늘어난다. 이는 future work로 남겨둔다.

conv layer는 커널로 인해 rnn보다 주로 더 계산이 많다. 그러나 seperable convolution(???이건또 뭔데)은 복잡도를 확연히 줄일수가 있다. 

또한 좀더 해석가능한 모델이 되는데, 뒤에 appendix에 남겨두었지만 이를 이용해 다른 task를 하도록 학습할 수 있을것이다.(???appendix에 뭐가일을까...봐보자ㅠㅠ)

### 5. Training

#### Training Data and Batching

4.5 million개의 sentence pair가 있는 WMT 2014 English-German를 사용하였다.  sentence encoding은 source와 target vocabulary에서 37000개의 단어가 겹쳐 byte-pair encoding을 사용하였다. (참조논문만 보여주고 설명은 안함) English-French로는 36Milion개의 WMT 2014를 사용, 32000의 단어가 있었다. sentence pair는 길이에 따라 batch로 묶였다. 각 batch는 약 25000개의 source token과 역시 약 25000개의 target token이 있었다.

#### Optimizer

Adam optimizer를 사용했고 $$\beta_1=0.9$$, $$\beta_2=0.98$$, $$\epsilon=10^{-9}$$로 하였다. 

learning rate는 다음과 같은 공식으로 설정했다.
$$
lrate=d^{-0.5}_{model}\cdot min(stepnum^{-0.5}, stepnum\cdot warmstep^{-1.5})
$$
이렇게 함으로써 warmup_step에서는 learning rate를 선형적으로 증가시켰다가 이후에는 step_number의 square root로 천천히 감소시킨 learning decay를 가능하게 했다. warmup_step=4000으로 하였다.

#### Regularization

##### Residual Dropout

각 sub-layer가 skip connection과 더해지고 normalized되기 전에, sub-layer의 output에 dropout을 0.1로 적용하였다. embedding과정에서도 dropout을 하였다.

### 6. Results

#### 6.1 Machine Translation

WMT 2014 Eng-Ger에서 big transformer model이 앙상블을 포함한 이전모델을 2.0 BLEU score로 앞섰다. base모델 역시 training 비용을 고려하였을때 이전모델들과 견줄만하다.

WMT 2014 Eng-French에서는 big model이 이전의 다른 single model보다 training 비용은 1/4로 줄었음에도 BLEU는 더 좋았다. 

training cost는 학습시간과 사용된 GPU수, 각 GPU의 연산능력을 곱하여 추정하였다.

![attention5](https://user-images.githubusercontent.com/31824102/36532000-79221ece-17b7-11e8-9612-2f1f5e6fa956.PNG)

#### 6.2 Model variations

Eng-Germ 모델을 newstest 2013이라는 새로운 데이터에 적용해보면서 Transformer의 설정을 조금씩 바꿔보며 기록해보았다. 이때 beam search를 사용하였다.

(A) 연산량은 유지하면서 attention head의 수나 key, value의 차원을 조절해보았다. head가 너무 많은것도, 적은것도 성능에 악영향을 주었다.

(B) attention key size $$d_k$$를 줄이는것 역시 결과가 안좋았다. 아마 compatibility function이 dot product의 이점보다 더욱 복잡한것이 필요한것같다.

또한 역시나 모델의 차원이 크고 head를 8에서 16으로 늘린 big model이 최고의 성능을 보였다. 

### 7. Conclusion

attention에만 기반을 둔 인코더-디코더 모델이다. translation task에서, RNN이나 CNN보다 훨씬 빠르게 학습될 수 있다. Eng2Ger에선 앙상블모델보다 성능이 뛰어났다. 

---

참고

NLP attention에 관한 좋은 자료 : http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/

갓갓갓갓갓 attention is all you need 정리글 : https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.Wo5Z1KjFI2w