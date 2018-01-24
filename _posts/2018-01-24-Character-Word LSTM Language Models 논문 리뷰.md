---
layout: post
title: "[NLP] Character-Word LSTM Language Models 논문 리뷰"
categories:
  - 딥러닝
tags:
  - Deep Learning
  - NLP
comment: true
---

[TOC]

char-word LSTM 논문 리뷰

https://arxiv.org/abs/1704.02813

#### Abstract

Character-Word를 둘다 사용하는 LSTM으로써 word based의 문제를 해결하고, 파라미터 수를 줄였다. Character 단위의 정보는 단어의 비슷함을 알려주기도 하고 unknown word나 infrequent word 대해서도 적용할 수 있어 모델의 성능을 높여준다. 영어와 네덜란드어에 대하여 실험을 해보았을때, 이 모델이 파라미터가 더 많은 word-level모델에 비해서도 잘 작동하였다.

#### 1. Introduction

Language model로 LSTM과 그것의 변형인 GRU가 많이 사용되었다. LSTM이 더 성능이 좋다고 알려져 있기에, LSTM-based 모델에 집중하였다.

기존의 neural net Language model은 다음과 같은 결점이 있다. 1). parameter를 최적화하는데 많은 training data를 필요로해서, 빈도수가 적은 단어에 대해서는 parameter 가 부정확하다. 2) 더욱 큰 한계는 one-hot 벡터로써 인코딩을 함으로써 단어 구조 내부의 정보를 사용할 수 없었다는 것.

>  예를 들어 'felicity'는 '더할나위 없는 행복' 이란 뜻으로, 대부분의 경우OOV(out-of-vocabulary)로 분류되지만, '-ity'라는 subword로 끝난다는 점에서 'ability' , 'complexity'등을 보고 명사라고 판단할 수 있을 것이다. 기존의 모델은 이를 반영할 수 없었다.

>  우리는 character와 word embedding을 합침으로써 이런 단어구조의 정보를 활용할 수 있게되었다.  embedding을 합침으로서, 기존의 bag-of-characters 방식(input을 character들의 뭉치로 보는것)과 다르게 철자들의 order를 유지하면서도 각각의 철자 역시 보존할 수 있게 되었다. 또한, 훨씬 작은 차원의 character embedding matrix를 가진 character embedding과 부분적으로 바꾸었으므로, word embedding의 사이즈 역시 축소되게 된다. 이는 결국 파라미터의 감소로 이어진다. (vocabulary의 크기는 embedding size의 크기와 정비례는 아니더라도 비례인듯???) 추가로, 비슷한 character sequence가 꼭 앞에만 등장하는것이 아니기에, (ex overfitting, underfitting) character를 forward로 넣어보는 것과 backward로 넣어보는 것을 둘다 해보았다.

논문의 결과를 **먼저 요약**하자면 다음과 같다.

1. LSTM에서 word와 subword infomation을 합치는 방법, 즉 word와 character embedding을 합치는 방법을 을 제시하였다
2. word-level 임베딩 사이즈를 줄임으로써, 파라미터의 수를 효과적으로 감소시켰다
3. 같은 hidden unit수(고로 더 많은 수의 파라미터)를 가진 word-level 모델과 비교해보고, 같은 파라미터 수를 가진 word-level 모델과도 비교해보았다. char-word model의 성능이 좋았다. 또한, backward order로 character를 넣는 것이 성능이 더 좋았다.
4. 모델에 포함되지 않았던, OOV word에 대하여 성능이 좋았다.



### 2. Related Work

RNN에서 character-level을 사용하거나 둘의 정보를 합친 연구도 있었고, 형태소나 음절 같은 다른 subword information을 모델링한 연구도 있었다.(어잉?대단한데?) 그러나 여기에선 character 단위의 subword에만 집중한다. 역대 연구 설명들 주르륵...

CNN에서도 character 단위가 가능했다. char CNN과 highway, LSTM layer를 합하여 좋은 성과를 냈다. 그러나 여기에서 주된 성과는 highway 덕분이었다. high way layer를 없애니 CNN과 LSTM을 가진 모델은 2개의 hidden layer를 가진 기존 word-level 모델보다 성능이 안좋았다 한다. 데이터 양이 많아질 경우는 좀 더 잘했다.

### 3. Character-Word LSTM Language Models

기존의 word-level LSTM의 작동은 다음과 같다.

1. t 번째 단어가 one-hot vector $$w_t​$$로써 인코딩된다.

2. 이 one-hot vector $$w_t$$는 embedding matrix $$W_w$$가 곱해져, 최종적으로 word embedding $$e_t$$를 만들어 낸다.

   ![char-lstm-embed](https://user-images.githubusercontent.com/31824102/35316684-e6853892-00ca-11e8-879e-22477bda6b6c.PNG)

3. 이 $$e_t$$는 non-linear operation인 LSTM layer에 들어가고 최종적으로 output layer에서 softmax로 다음에 올 단어에 대한 확률을 반환한다.

기존의 모델에서 달라지는 점은 word embedding 부분이 character embedding과 합쳐지는 것인데, 구체적으론 다음의 수식과 같다.

![char-lstm-embed2](https://user-images.githubusercontent.com/31824102/35316796-6be6e4d6-00cb-11e8-99b1-38acdb92666b.PNG)

저기서 $$c_t^1$$은 첫번째 character의 one-hot encoding이고 $$W_c^1$$은 그것의 embedding matrix이다. 전체  단어인 $$w_t$$와 그 안의 각각의 character $$c_t^1,..c_t^n$$이 각각 embedding되어 합쳐진것이, LSTM의 input이 되는것이다. 이때, 전체 embedding size는 일정하게 유지한다. 따라서 word embedding의 size가 줄어든다. (keep the total embedding size constant, the 'word' embedding size shrinks in size) 

![char-lstm-embed3](https://user-images.githubusercontent.com/31824102/35316798-6c509598-00cb-11e8-8fda-5b26101780cd.PNG)

> (10x7) (7 x1) =(10x1)이게 기존
>
> (8x7)(7x1)=(8x1)이렇게 기존거를 줄이고
>
> (2x1)(1x1)=(1x1) 철자임베딩을 넣는식
>
> (2x1)(1x1)=(1x1) 두번째 철자.
>
> eT=[(8x1)(1x1)(1x1)]=(10x1)

이때, 합쳐지는 character의 갯수는 일정한 상수 $$n$$으로 고정되었고, 이를 넘는 경우 순서대로 n번째까지만 넣었다. 반대로 character 갯수가 짧을 때에는, special symbol을 padding으로 넣어주었다. 또, 넣을때에는 character의 순서를 유지하였다. (순서를 무시하였을 경우 성능 향상이 없었다.)

#### 3.1 Order of characters

character가 추가되는 순서를 forward로도 backward로도 해보았다. 영어와 네덜란드어 에는 접미사가 중요한 경우가 많기에, 단어의 끝에 더 강조를 두기 위해 backward로 해보았다.  또한 양쪽 방향(both)도 해보았다.

#### 3.2 Weight sharing

character embedding 에서, 첫번째 character든 두번째 character든 같은 vocabulary(a~z, 1~10 등등)을 것이라는 점에서 같은 Weight를 사용하는 것도 의미가 있다. 이 경우 수식은 다음과 같다

![char-lstm-embed4](https://user-images.githubusercontent.com/31824102/35316693-e85accfe-00ca-11e8-88d4-d3f01bb70072.PNG)

그러나 영어에서 맨 마지막에 오는 's'가 특별하게 복수의 의미를 갖듯이, 각 위치마다 다른 의미를 가지고 있다고 볼 수도 있다. 

따라서 weight 를 공유하는 모델과 공유하지 않는 모델을 모두 실험하였다.

#### 3.3 Number of parameters

total 임베딩 중 일부가 character를 modeling하는 데에 사용되었다는 점에서, 실제 word embedding은 훨씬 작아졌다 기존의 word-level LSTM에서 embedding matrix의 파라미터는 vocabulary size **V**와 total embedding size **E**의 곱인 $$V \times E$$였다. 여기서 total embedding size $$E$$는 word embedding size $$E_w$$와 같다. ($$E=E_w$$)

![lstm1](https://user-images.githubusercontent.com/31824102/35316692-e81207f8-00ca-11e8-834e-89e411d3301f.PNG)

그러나 char-word level 모델의 파라미터 갯수는 다음과 같다. 

![lstm2](https://user-images.githubusercontent.com/31824102/35316691-e7e08f0c-00ca-11e8-8ada-1f73bc96f6ef.PNG)

여기서 **n은 들어가는 character 갯수이고,C는 character size, $$E_c$$는 character embedding size**이다. V가 C보다 훨씬 크므로, 파라미터 갯수를 엄청나게 줄였다고 할 수 있다. 

> $$E-n\times E_c=E_w$$아닌가? 라고 생각해서
>
> 그럼 $$V\times (E_w)+n\times(C\times E_c)>V \times E_w$$인데..?? 라고 생각할 수 있지만 total size가 fix면 $$E_{w2}$$가 $$E_{W1}$$보다 더 작아진 애이다. 그냥 word embedding size를 줄여줬다 이해하면 된다.

character에서 weight sharing을 해준다면, n개의 각각의 파라미터가 필요 없으므로 다음과 같이 더 줄어든다

![lstm3](https://user-images.githubusercontent.com/31824102/35316690-e7aadc68-00ca-11e8-8d9a-33ca0bd176a3.PNG)

### 4. Experiments

#### 4.1 Setup

test와 train은 TensorFlow로 하였다. 200개의 hidden unit이 들어있는 2개의 layer로 된 small LSTM과 650개의 hidden unit이 들어있는 2개의 layer로 된 large LSTM으로 실험하였다. **embedding layer의 크기는 항상 hidden layer의 크기와 같게하였다. **

small model : 13에폭, 첫 4에폭 후에 0.5로 learning rate decay, 25% dropout

large model : 39에폭, 6에폭 후에 0.8로 decay, 50% dropout

영어 data set으로는 Penn Treebank(PTB)를 사용. 900k training word token, 70k validation set, 80k test set. 많지 않은 수이지만 여러 연구에서 쓰인 데이터다.  비교를 위해 이전 연구들과 같이 소문자로 통일하였고, unknown word는 $$<unk>$$라고 썻다.  character vocabulary size는 **48**이다. (이전 논문은 70개였는데...이미 다르지 않나....)

네덜란드어 data set으로는 Corpus of Spken Dutch(CGN)가 쓰였다. 회의나 토론, 강연 등의 내용이 담긴 데이터셋이고 1.4M train token, 180 validation set,190k test token이 있었다. 성능 비교를 위해, 저 중 PTB와 같은 data set size 만이용하였고 같은 **voabulary size(10k)**로 제한했다.

#### 4.2 Baseline models

비교를 위해 만든 baseline model. **1)** 하나는 같은 hidden unit이 있는(따라서 더 많은 파라미터를 가지고 있는) LSTM, **2)**하나는 비슷한 수의 파라미터를 가진 LSTM이다. 즉, small model에서는 unit 을 200 -> 175, large에서는 650 -> 475로 줄였다. 

(비슷한 수의 parameter를 갖도록 한 버젼에 대하여) embedding layer size도 475이니까, large word-level model의 embedding matrix크기는 10000 X 475 =475,000개이다. 

> **(475X10000)** (10000X1)= (475X1), 10000개는 vocabulary의 크기. voca는 10k로 고정해놧었음.

한편 10개의 character를 넣고 거기에 각각 25 embedding size의 character를 사용한 경우 총 (10000 X (650 -10 X 25)) + 10 X (48 X 25) = 412,000개이다. 

> (**(650-10X25)** X 10000)(10000X1)=(400X1)
>
> **10개 X (25X48)**(48X1) = 10개 X (25X1), 합치면 (650X1)

#### 4.3 English

![lstm4](https://user-images.githubusercontent.com/31824102/35316689-e77710e0-00ca-11e8-8a11-ce9b571862ec.PNG)

영어 dataset PTB에 대해서 보았을때, small model의 경우 전체 임베딩에서 단어 임베딩이 과반을 차지하는 경우(embedding size가 15이고 단어갯수가 7개 이상)를 제외하고는 같은 파라미터갯수의 base model보다 뛰어났다. 같은 hidden unit(200)을 가진 base model보다는, 약간 좋아졌다. character embedding 의 size가 커지면 반대로 word embedding size가 줄어들어 충분한 정보를 주지 못하였다. 즉, 큰 character embedding size의 경우 좋은 성능을 보이지 못했다. 

제일 좋은 성능을 보인것은 character embedding size 5로 3~7개의 character만을 추가한 경우였다. (!! 앞에서 부터 넣는것과 뒤에서 부터 넣는 것의 차이점인듯. n개 이외에는 아예 잘라서 안쓰는거니까.)

large model의 경우에도 character embedding size가 지나치게 커지면 좋은 결과를 보이지 못했다. 제일 좋은 성능을 보인 것은 **character embedding size가 25이고 8개의 character만을 넣는 모델**이었다. 

단어를 넣는 순서에 대해서도 달리 해보았다.

![lstm5](https://user-images.githubusercontent.com/31824102/35316688-e7482186-00ca-11e8-909b-2bba3796efee.PNG)

가장 좋은 성능을 보인 것은 both로, 앞에서 3개, 뒤에서 3개 넣은 것이다. 그러나 넣는 character 수를 늘리면 오히려 성능이 저하되었다. forward, back ward중에서는 backward가 전반적으로 좀더 잘했다.

결론 : 추가되는 character embedding이 전체 embedding size에 비례하여야 한다. word embedding이 더 많이 들어가야 한다. Large model에서 제일 좋은 성능 향상을 보였다.

#### 다른 연구의 모델들과 비교

![lstm6](https://user-images.githubusercontent.com/31824102/35316687-e718297c-00ca-11e8-85ee-61702480efd3.PNG)

또, 기존의 char-level로 해본 Kim et al보다(highway를 제거한모델에서) 성능이 좋았고, bidirectional로 char-word embedding을 하고 dropout을 쓰지 않은 Cho과는 비슷한 성능이었지만(비교를 위해 우리모델도 dropout안함) 우리모델이 덜 complex하다.

#### 4.4 Dutch

richer morphology. 형태학적으로 더 뛰어나다...예상한대로 English에서보다 성능향상이 더 뛰어났다. 최적의 결과를 낸 모델설정은 English와 같았다. 

#### 4.5 Random CW models

character를 넣는 것이 input에 noise를 넣는 역할로써 성능향상이 이뤄진것 아니냐를 탐구하기 위해, (noise가 성능향상시키나???) 진짜 noise, 즉 무작위 character를 넣은 모델을 만들어 보았다. 다른 설정은 모두 동일.

그에 따른 결과는 다음과 같다. 무작위 noise모델을 base-model과 우리모델과 비교했을때 차이를 나타낸 표.

![lstm7](https://user-images.githubusercontent.com/31824102/35316686-e6eb13f6-00ca-11e8-976f-c116e66782d0.PNG)

Random noise모델은 English에서는 baseline보다도 못했고, Dutch에서는 baseline모델보다는 잘햇으나 우리모델보다는 훨씬 못했다. 고로, character를 넣는게 유의미하다.

#### 4.6 Sharing weights

위에서 설명한대로 각 위치의 character 임베딩을 모두 통일해보았다. 그러나 이 경우에도 첫번째 character는 첫번째 자리에 들어갈것이므로, 위치 정보가 사라진것은 아니다. (다만 's'의 예시처럼 위치에 따른 특별한 의미가 있을 것이라는걸 반영못해준듯). 이렇게 하면 모든 character가 같은 embedding으로 매핑되면서, 파라미터수가 많이 줄어든다. 이를 baseline과 sharing안한 char-word와 비교한 결과.

![lstm8](https://user-images.githubusercontent.com/31824102/35316685-e6bde836-00ca-11e8-8ffa-0bf33d8dbdfb.PNG)

English의 경우 small model을 제외하고 baseline보다 더 나은 결과를 보였으나, weight sharing안한 모델이 '약간' 더 좋았다. 그로 positon마다 특별한 의미가 있는것 같다.

#### 4.7 Dealing with out-of-vocabulary words

언급하였듯이 Out of Vocabulary의 단어들에 대해 character 정보를 알 수 있으니 error가 줄지 않을까 기대했다. 이에 대해 다음과 같이 실험했다.

OOV word가 들어왔을때 나오는 다음 단어에 대한 probability들을 check해보았다. 기존이 word-level model과 가장 좋았던 char-word model (25 char embed size, 6 char, backward)로 해보았다. 그 결과 Char-word model이 실제 단어에 높은 확률을 부여한게 17483번, word model이 더 높은 확률 부여한게 10724번으로, OOV word에 실제 도움이 된다할 수 있다. (한 3문장으로 스르륵 넘어갓다..)

### 5. Conclusion and future work

char-word LSTM은 기존의 word LSTM에 비해 다음과 같은 장점이 있다.

- LSTM의 size를 줄여준다
- 성능은 더 좋아진다. (improves the perplexity)

또한, 여러 char embedding size, 여러 추가되는 char 갯수, 여러 order로 해보았는데, char embedding이 차지하는 비중이 지나치게 커지면 안되고, backward order가 조금더 잘했고, 양방향으로 조금씩 넣는것도 잘했다. 

random noise를 넣어본 모델보다 잘했기에, char를 넣는게 noise를 만들어서 잘한 것은 아니라 할 수 있다. 

weight sharing은, share 안한 모델보다 약간 못했다. 마지막으로 char-word model은 OOV에 대해 더 잘 작동했다.

#### future work

char-LSTM과 word-LSTM을 합치는것도 시도해볼만하다. 그리고 char를 one-hot vector가 아닌 co-occurrence vector로 넣는것도 유의미할듯하다. 사람의 언어구조를 모방해서 frequent에는 word-LSTM, infrequent에는 subword-LSTM을 해보는 것을 어떨까.