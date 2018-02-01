---
layout: post
title: "[NLP] Show and Tell: A Neural Image Caption Generator 논문 리뷰"
categories:
  - 딥러닝
tags:
  - Deep Learning
  - NLP
comment: true
---

{:toc}

show and tell 논문 리뷰

https://arxiv.org/abs/1411.4555

### Abstract

현재까지의 발전된 vision영역과 machine translation영역을 조합하여, recurrent architecture에 기반한 generative모델을 만들었다. 즉 이미지를 보고 자동으로 설명을 하는 모델. (모델은 train image의 target description에 대한 likelihood을 최대화 시키는 형식으로 학습하였다.) 여러 데이터셋에 기반한 양적, 질적평가를 통해 보았을때 종종 사람에 견줄만한 정확도를 보여주었다.

### 1. Introduction

이미지의 내용을 적절한 문장으로 설명해내는 것은 기존의 computer vision의 주된 목표였던 이미지 classification이나 object recognition문제보다 훨씬 어려운 문제이다. 단순히 이미지에 들어있는 object를 잡아내는 것이 아니라 그들의 특성, 하고 있는 활동, 다른 object와의 관계 까지 이해해야 하기 때문이다. 뿐만 아니라 이해한 바를 자연어로써 표현하려면 language model 역시 필요하게 된다.

가장 초창기 시도는 위의 문제들을 해결하기 위한 기법들을 연결시켜보는것이었으나, 우리는 이미지 input이 들어오면 바로 target sequence of words의 likelihood를 maximize할 수 있는 single joint model을 선보이고자 한다. 

이는 근래의 machine translation에서 다른 언어로 해석된 target sentence의 likelihood를 최대화하는 방식에서 근거하여 고안되었다. 몇년동안 machine translation은 단어를 각각 해석하고, 이를 정렬하는 등의 분리된 작업들을로써 연구되었지만 최근의 RNN은 이를 훨씬 쉬우면서도 뛰어난 성능으로써 해낼 수 있다는걸 보여주었다. “encoder” RNN이 source sentence를 읽고 이를 fixed-length vector representation으로 바꾸면, “decoder” RNN이 그에 기반하여 target sentence 를 만들어 낸다. 

우리는 해당 구조에서 encoder RNN을 CNN으로 대체한 방식을 선보인다. CNN은 input image를 fixed-length vector로써 임베딩함으로써 이미지에 대한 우수한 representation을 만들어 낼 수 있다. 따라서, 이미지 분류를 목적으로 pre-train된 CNN의 last hidden layer를 decoder RNN의 input으로 넣어 sentence를 만들어내는 형식의 구조이다. 이를 Neural Imgae Caption(NIC)라 부를것이다.

우리의 공헌은 다음과 같다. 우리의 시스템은 end-to-end 시스템이다. (기존의 두가지 방법을 따로 합치던것과 다르다는 의미인듯). 전체 neural net이 SGD를 이용하여 학습된다. 또한 더큰 corpora에 pre-trained되어 더 좋은 성능을 기대할 수 있다. . 무엇보다, 기존의 최상의 결과보다 훨씬 뛰어난 성능을 보인다. Pascal dataset에서 기존의 모델의 BLEU 점수(높을 수록 좋음)가 25 였던 반면 NIC는 59점을 기록하여, 사람의 점수 69와도 비교할만한 성능을 보였다.

### 2. Related Work

object recognition에서 attribute와 location도 인식할 수 있게 되면서, 이를 이용하여 description을 generate하려는 시도들이 있었으나 이는 수작업이 많이 들어가고 text generation의 표현력이 좋지 못하다.

다른 시도는 이미지와 description을 같은 벡터공간에 co-embedding해서 이미지와 비슷한 embedding공간에 있는 description을 반환하는 모델이 있었다. 그러나 이는 새로운 description은 만들어내지 못한다. 즉 training data에 있던 object라도 보지 못한 관계(unseen composition)를 맺고 있으면 이에 대한 description을 만들지 못한다. 또한 반환된 description의 평가 역시 할 수 없다.

우리는 이미지 분류를 위한 CNN과 sequnce modeling을 위한 RNN 을 합쳐서 single network를 만들었다. RNN은 이 single network에서 학습되었다. (CNN은 pre-trained). 즉 기존의 sentence를 받던 모델에서 input이 (convolution 연산을 거친) image로 바뀐 것이다. 기존에도 비슷한 모델이 나온바 있으나 우리는 더욱 powerful하고 direct한 RNN model을 사용하여 RNN이 text로 설명되는 object를 더 잘 받아들일 수 있도록(keep track..) 만들었고, 그에따라 성능이 눈에 띄게 증가하였다.

### 3. Model

최근의 연구에 따르면 올바른 sequence model하에서, correct translation의 확률을 maximizing하는 방향으로 train을 하는 것이 성능이 좋다고 알려져 있다. 이러한 방식으로 RNN은 input을 고정된 차원의 벡터로 encoding하고 이를 sentence로 decode하게 된다. 이와 같은 형식을 image translation에도 적용하였다.

따라서 주어진 이미지에 대해 correct description에 대한 확률을 maximize하도록 하였다. 식으로 나타내면 다음과 같다. 

![show1](https://user-images.githubusercontent.com/31824102/35560187-8c80dcd2-05a4-11e8-8441-fd92bb1a4559.PNG)

모든 경우에 대해 input image $$I$$가 주어졌을때의 correct description $$S$$의 확률을 최대화 할 수 있는 parameter $$\theta$$를 구하는 것이다.

이때 description S는 길이가 정해져있지 않으므로, 실제 답의 길이가 N이라면 그 답(description)에 대한 확률은 고정된 수식이 아니라 다음과 같이 $$S_0,..,S_N$$까지의 확률을 결합확률로 나타내야 한다. 

![show2](https://user-images.githubusercontent.com/31824102/35560186-8c3d87f2-05a4-11e8-946a-38d8f9b94b74.PNG)

> 수식 t=1이어야 될꺼같은데..
> $$
> p(S|I)=p(S_1|I,S_0)p(S_2|I,S_0,S_1)..p(S_N|I,S_0,..,S_t-1)
> $$
> 즉 (I가 주어졌을때의 0번째 단어가 $$S_0$$일 확률) X (I, $$S_0$$이 주어졌을때의 1번째 단어가 $$S_1$$일 확률) X ...의 형태인 것이다.

train 과정에서는 이미지와 실제 description이 $$(S,I)$$로 주어지고, 위의 식에 따라 최적화를 한다. (SGD를 사용하였다.)

t-1번째까지의 단어가 주어졌을때 t번째 단어에 대한 확률 ![show3](https://user-images.githubusercontent.com/31824102/35560184-8c0dffbe-05a4-11e8-8ca6-3c5a87cc0863.PNG)은 RNN으로 구현을 하는데, t-1번까지의 '조건'들이 fixed length 인 hidden state $$h_t$$로 표현이 되는 것이다. 알다시피 이 hidden state는 새로운 input $$x_t$$가 들어오면 non-linear 함수를 거쳐 새로 업데이트 된다.

![show4](https://user-images.githubusercontent.com/31824102/35560183-8bca7bb8-05a4-11e8-8309-44bb539322dd.PNG)

이 때 RNN의 성능을 향상시키기 위해 다음 2가지가 고려되었다. 1) 어떤 non-linear function을 써야할까 2) 어떤 방식으로 이미지와 단어가 $$x_t$$로써 들어갈 수 있을까

non-linear로는 성능이 뛰어나다 알려진 LSTM를 사용하였다. (non-linear라길래 Relu 그런걸 생각했는데 LSTM의 update방식을 통틀어서 non-linear라고 부를 수 있구나..) 또한 image representation으로는 object recognition, detection에 뛰어난 성능을 보이는 CNN을 사용하였다. 이때, batch normalization을 사용하였다. 

### 3.1 LSTM-based Sentence Generator

LSTM을 택한 주된 이유는 vanishing gradient문제를 잘 해결해주기 때문이다. 

![show5-LSTM](https://user-images.githubusercontent.com/31824102/35560181-8b8e41c0-05a4-11e8-85f0-e8af14d52226.PNG)

이해를 돕기 위해 논문에서 제시된 그림보다 직관적인 그림은 다음과 같다.([참고](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))

![LSTM](https://user-images.githubusercontent.com/31824102/35672650-1cca790e-0737-11e8-8390-fe804a342a6a.PNG)

LSTM은 매 input마다 업데이트 되는 memory cell C가 핵심 요소이다. 이 cell C는 3개의 gate가 각각 곱해지며 통제를 받는데, gate가 0이면 값을 반영을 하지 않고, 1이면 반영을 하는 형태이다. 3개의 gate는 각각 현재의 cell 값을 '잊을지 말지'를 통제하는 forget gate(f), 어떤 input을 반영해줄지를 통제하는 input gate(i), 어떤 것을 output으로 내보낼지를 통제하는 output gate(o)가 있다. 

![show5-LSTM2](https://user-images.githubusercontent.com/31824102/35560179-8b4d5cbe-05a4-11e8-88a9-4f3f68f5380b.PNG)

각각에 대한 수식은 위와 같다. (위의 수식에선 hidden state를 $$m_t$$로 나타냈다.)

최종적으로는 $$m_t$$에 softmax를 씌워 모든 단어에 대한 확률 $$p_t$$를 만들어 낸다.

#### Training

이러한 구조를 가진 LSTM은 이미지와 이전까지의 단어를 토대로 (만들어진 desc의 앞단어를 의미하는 것인가) ![show3](https://user-images.githubusercontent.com/31824102/35560184-8c0dffbe-05a4-11e8-8ca6-3c5a87cc0863.PNG)문장의 다음단어를 예측해내도록 학습된다. 따라서 이를 recurrent한 그림이 아닌 unfold된 그림으로 그리면 다음과 같다. 물론 LSTM은 parameter를 share한다.

![show5-LSTM3](https://user-images.githubusercontent.com/31824102/35560178-8b1998c0-05a4-11e8-985f-8c53dfa0c312.PNG)

이때 $$S_t$$는 각각 dictionary size의 차원인 one-hot vector이다. $$S_0$$과 $$S_N$$은 각각 시작과 끝을 나타내는 special stop word로 지정하였다. 또한 각각의 단어 S에 word embedding $$W_e$$를 해줌으로써 CNN을 통한 image representation과 word가 같은 차원에 있도록 하였다. 즉 **첫번째 input,** $$S_{-1}$$은 CNN(I)이고, 여기서 나온 hidden state이 **두번째 input,** $$S_0$$이 합쳐져 output $$S_1$$과 새로운 hidden state를 만드는 것이다. 매 time step마다 image를 따로 넣어주는 방식을 시도해 보았으나, 이는 overfitting의 더 안좋은 결과를 낫는것으로 경험적으로 밝혀졌다.

이때의 loss는 각 step에서의 negative log likelihood의 합이다. 

![show6](https://user-images.githubusercontent.com/31824102/35560177-8adf4990-05a4-11e8-9c84-87cd987ca103.PNG)

이를 최소화 시키는 방향으로 LSTM의 parameter와 word embedding $$W_e$$, CNN의 image embedding을 하는 top layer를 학습한다. 

#### inference

sentence를 만드는 방법은 여러가지가 있는데, 첫번째 방법은 $$p_1$$에 따라 첫번째 단어를 만들고, 이를 다시 input으로 넣어줘 $$p_2$$를 만들고, 이를 end-of-sentence 토큰이 나오거나 최대 길이가 될때까지 반복하는 Sampling기법이 있다. (각 단계에서 최고의 하나의 단어만을 뽑기에, 매우 greedy한 방법이다. 이는 하나의 단어가 잘못될 경우 그에 기반한 모든 결과가 망가지는 위험이 있다. )

두번째로는 매 t번째까지의 input으로 만들어진 최적의 문장 k개를 후보로 저장하고 또 그 후보들로 만든 t+1번째까지의 문장 중 k개의 문장을 반환하는 것이다. 이를 BeamSearch라고 한다. 

여기에선 BeamSearch 방식으로 실험을 진행하였고, k=20으로 하였다. k=1일 경우 결과의 BLEU수치는 평균적으로 2점정도 내려갔다.(그럼 5개의 description에 20개의 상당히 다른 문장들. 어떻게 Loss를 만드는걸까)

### 4. Experiments

#### 4.1 Evaluation Metrics

주어지지 않은 이미지에 대한 description으로써 잘했는지에 대한 평가 척도는 선행연구들에서 많이 제안되었다. 제일 좋은 것은 사람들에게 평가를 맡기는 것이지만, 우리의 선행연구에서 사람들의 평가와 평가척도가 어느정도 일치함을 밝혀냈다. 

image description에 가장 많이 쓰이는 척도는 BLEU이다. 이는 reference sentence(예를 들면 사람이 만들어낸 description)와 몇개의 n-gram이 겹치는지(찾아보자!!!)를 나타내는 측도이다. (대부분의 연구는 1-gram, 즉 unigram으로 진행되었다). 우리는 이 BLEU를 이용하였고 이때의 reference sentence와 우리의 generated output은 다음에서 확인할 수 있다. [http://nic.droppages.com/](http://nic.droppages.com/)

BLEU 말고도 우리의 목표 description과 얼마나 가까운지를 평가하기 위해 perplexity를 지표로 사용할 수도 있다. 

> 주어진 이미지에 대해 correct description에 대한 확률을 maximize하도록 했던 입장에서.
>
> ![show1](https://user-images.githubusercontent.com/31824102/35560187-8c80dcd2-05a4-11e8-8441-fd92bb1a4559.PNG)

perplexity는 추정된 단어들에 대한 probability의 역수의 기하평균이다. 이는 validation set(정답description이 있다)에 대한 hyper parameter tuning에서는 사용되었으나 BLEU가 많이 선호되어 제시하진 않았다. (연구자들의 추가적인 논의를 위해 다른 평가척도인 ME-TEOR, Cider라는 것도 제시하였다)

### 4.2 Datasets

사용한 데이터셋은 다음과 같다.

![show7](https://user-images.githubusercontent.com/31824102/35560176-8a7d3106-05a4-11e8-93be-881c79b55846.PNG)

SBU를 제외하고는 모두 '5개의 문장' 라벨이 있다. SBU는 Flikcr에서 유저들이 올린 description이므로 noise가 심한 dataset이라 할 수 있다. 또한 Pascal 데이터는 test를 위해서만 사용하였는데, 나머지 4개의 data set으로 학습을 하고 평가를 하는 식으로 사용되었다.

### 4.3 Results

#### 4.3-1 Training Details

supervised approach는 많은 데이터를 필요로하지만, 최상의 이미지는 10만개뿐이었다. 고로 데이터가 더 많아지면 더 좋은 결과를 낼것이라 기대한다

overfitting을 방지하기 위해 ImageNet을 통해 pretrained된 CNN으로  weight initialize를 하였다. word embedding $$W_e$$도 pretrained된 것을 써보았으나 효과가 없었다. 

- CNN을 제외한 weight은 randomly initialized
- 모든 weight은 SGD로 학습, learning rate은 고정
- embedding size와 LSTM size는 512로 같다
- ovrfitting을 방지하기 위해 dropout과 ensemble model
- hidden unit과 depth를 다양하게

 Dropout과 ensemble은 BLEU의 큰 향상을 보이지 못했다. 

#### 4.3-2 Generation Results

![show8](C:\Users\admin\내파일\3-2.5\스터디\NLP스터디\data\show8.PNG)

Table1은 여러 평가척도로 내본 성능 결과. 사람보다 좋은 결과가 있는데 실제 비교에선 이만큼 잘하진 못했으므로, 평가척도 역시 더많은 연구가 필요하다.

Table2는 이전 연구들과의 비교. 여기서 BLEU는 5명의 사람들이 만들어낸 description을 토대로 만든 BLEU를 평균낸것. Human은 다른 4명의 description과 비교해서 낸 점수수 로햇다가 차별인것같아서 다시 5명의 description과 비교(오잉? 본인것도 포함햇단 소리인가???)

평가지표는 선행 연구들에서 많이 사용되던 BLEU-4로 하였다. 

#### 4.3-3 Transfer Learning, Data Size and Label Quality

한 data set에서 만들어진 모델을 다른 data set에서 평가해서(transfer) domain의 mismatch에도 불구하고 high quality data, more data로 극복가능한지 보았다. 

같은 유저그룹에 의해 만들어져 label이 비슷할 것인 Flickr 30과 Flickr 8로 transfer learning을 해보았다.  Flickr 30로 학습하였을 경우(데이터가 4배 정도 많다) Flickr 8에서 BLEU가 4점 올랐다. 이는 많은 데이터에 기반한 overfitting방지의 역할을 한듯하다. MSCOCO는 Flickr 30보다도 5배더 크지만, mismatch도 더 큰 데이터인데, 이 경우 BLEU가 10점 내려갔다. 그러나 여전히 준수한 description을 보였다.

마지막으로 SBU는 크기가 크지만 사람이 만든 description이 아닌 단순 캡션, 즉 좋지 못한 labeling이었는데 MSCOCO의 모델을 SBU에 돌려보았을때 28점에서 16점으로 떨어졌다.

고로 크기도 질도 중요하다..domain이 비슷한 data set은 역시 중요하다..

#### 4.3-4 Generation Diversity Discussion

generating의 관점에서, 모델이 새로운 caption을 만들어냈는지, 다양하고 high quality인지 보고자 하였다. 

![show9](https://user-images.githubusercontent.com/31824102/35672988-11b6ccf6-0738-11e8-9a78-d3a086f3d742.PNG)

이는 Beam Search방법에서 N개의 best를 뽑은것. 다양한 단어, 다양한 관점에서 문장이 만들어졌다. 상위 15개의 generated 문장들을 보았을때는 58점 정도로, 사람과 견줄만한 수준이었다. 이 중 볼드체는 training set에 없었던 문장들이다. 보통 best 문장 중 80%가 training set의 문장이었다. training data가 적으므로(5개 라벨) 예시문장을 만들어내게 되었다. 그러나 15개의 best문장만을 보았을때는 절반정도가 새로운 description이었고 BLEU점수도 비슷해서, 다양하고 high quality의 문장을 만들어냈다고 볼 수 있다.

#### 4.3-5 Ranking Results

우리는 좋지 않은 평가 방법이라 생각했지만 다른 연구들이 많이 썼던 ranking score도 써봤는데, 여기서도 좋은 점수가 나왔다.

#### 4.3-6 Human Evaluation

![show10](https://user-images.githubusercontent.com/31824102/35672652-1d23c162-0737-11e8-8dc7-5c69c8b309d1.PNG)

몇개의 test set에 대해 사람이 바로 평가를 내려보게도 하였다. 그러나 이때 ground truth에 대해 reference sentence 점수차도 많이나 역시나 BLEU가 완벽한 평가지표가 아님을 보였다.

#### 4.3-7 Analysis of Embeddings(이부분 이해가잘..)

이전 단어 $$S_{t-1}$$을 LSTM의 input으로 넣어주기 위해 사용한 word embedding은 one-hot-encoding과 다르게 dictionary size에 제한되지 않는다. (띄용?왜죠???) 따라서 다른 모델들과 함께 jointly trained될 수 있다. 다음은 학습된 embedding space에서 근접이웃들의 몇가지 예시이다.

![show11](https://user-images.githubusercontent.com/31824102/35672651-1cf6b9ce-0737-11e8-98b9-6fe42710dc24.PNG)

이렇게 model에서 학습된 관계는 vision component에도 도움을 줄 수 있는데, horse와 pony, donkey를 비슷한 위치에 있음으로 CNN이 horse-looking 동물의 feature를 extract하는 것이 더 수월해진다 (띄요옹? CNN은 pretrained이고, 여기서 나온걸 넣어주는거 아니엇나)

### 5. Conclusion

CNN을 사용한 Encoding과 RNN을 사용한 sentence generating을 하는 neural net을 만들었다. 학습은 likelihood를 maximize하는 방향으로 학습하였다. 우리의 NIC는 양적평가(BLEU등), 질적평가에서 모두 뛰어났다. image description을 가진 size가 더 늘어나면 NIC의 성능도 증가할 것이다. 

추가로 이미지와 텍스트가 각각 있는 unsupervised data에서 image description의 성능을 향상시킬 수 있는 방법을 연구하는 것도 흥미로울 것이당.