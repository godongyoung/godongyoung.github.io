---
layout: post
title: "[NLP] Character-level Convolutional Networks for Text Classification 논문 리뷰"
categories:
  - 딥러닝
tags:
  - Deep Learning
  - NLP
comment: true
---

char-CNN논문 리뷰

https://arxiv.org/abs/1509.01626

### Abstract

이 논문은 Character-level convolutional networks을 다양한 데이터셋에 시도해보았고 성능을 비교해보았다. 비교 대상은 전통적인 모델인 bag of words, n-grams, 그들의 TF-IDF버젼, 그리고 딥러닝 모델인 RNN, word-based CNN이었고, 비교 결과상당한 성능을 보였다.

### 1. Introduction

text classification 이란? nlp의 전통적인 topic중 하나로 자연어가 미리 지정된 category중 어디에 속할지를 맞추는 것이다. 현재까지는 n-grams과 같은 순서가 있는 단어의 조합을 다루는 등의 '단어 단위'의 기술들이 활용되고 있고, 또 성능이 뛰어났다.

기존의 연구들이 CNN이 언어에 대한 사전 정보 없이도 임베딩된 단어들에 잘 적용될 수 있음을 보여주었다. 그러나 여기에선 text를 철자단위로 다루어 CNN을 시도해보았다. 기존에도 character-level n-gram이나 단어 단위로 CNN을 한 후 철자 단위로 특징을 뽑는 등 철자단위의 nlp 작업들이 있었으나 철자단위에 바로 CNN을 적용한것은 처음시도이다. 

철자 단위의 CNN은, 모든 단어구조는 철자단위로 되어있을 것이기에 **1)** 단어가 segmentation이 가능한지에 관계없이(tokenize에 대한 어려움을 얘기하는듯) 작동할 수 있고 **2)** 오타나 이모티콘등과 같은 비정상적인 철자에 대해서도 자연스럽게 배울수 있다는 강점이 있다.

### 2 Character-level Convolutional Networks

####  2.1 Key Modules

주된 구성은 1-D input, 즉 1차원 벡터에 대한 1-D convolution과 똑같다. input g(x)와 사이즈가 k, stride가 d인 kernel function f(x)와 convolution h(x)가 있다고 하면 

$h(y)=\sum _{x=1}^{n} [f(x)\cdot g(d \cdot(y-1)+k-(x-1))]$으로 식으로 쓸 수 있는데, 그냥 우리가 알고 있는 1-D CNN.....저기서 $f \cdot g$는 내적이 아니라 element wise곱을 의미. 그냥 우리가 알고 있는 filter개념...이미지에서 처럼 filter를 여러개 만들어 여러 output $h_{j}$를 만들수 있다.

여기에 더 깊은 모델을 위한 max-pooling을 해준다. max-pooling외의 방식을 쓴 ConvNets의 경우 6개 이상을 쌓는데 실패했다 한다.

non-linearity로는 ReLU를 사용 ($h(x)=max\left\{0,x\right\}$), SGD를 사용.

####2.2 Character quantization

모델의 input으로는 encoding된 알파벳이 들어온다. encoding은 one-hot encoding을 한다.

모델에서 사용한 '철자'로는(alphabet이 굳이a,b,c,d의 개념이 아니었다.) 알파벳 26개, 10개의 숫자, 33개의 다른 부호들(-,;.!?등등)로 총 70개의 character가 있고, 이외의 빈칸을 포함한 '철자'가 아닌 것은 zero vector로 처리한다. 논문의 후에 알파벳의 대소문자를 구분한 모델과 비교해볼것이다.

#### 2.3 Model Design

large feature와 small feature로 2개의 ConvNet을 구성했고, 모두 6개의 conv layer와 3개의 fully-connected layer가 있다. 3개의 fc 사이에는 정규화를 위해 2개의 dropout이 있고 dropout의 확률은 0.5이다. filter의 stride는 1, pooling은 non-overlapping을 하였으므로 stride는 3.

![char-cnn-table](C:\Users\admin\내파일\3-2.5\스터디\NLP스터디\data\char-cnn-table.PNG)

large feature와 small feature는 filter가 몇개의 feature를 잡아낼지를 설정한것, 즉 filter의 수라고 이해하였다(맞나요???)

이후 6번째 layer를 통해 나온 output을 다음과 같은 output unit을 가진 FC에 넣는다. 마지막 output unit은 문제에 따라 다르다(ex. 목표가 10개의 class로 classification이면 10개의 output unit올 설정)![char-cnn-table2](C:\Users\admin\내파일\3-2.5\스터디\NLP스터디\data\char-cnn-table2.PNG) 

#### 2.4 Data Augmentation using Thesaurus

이전의 연구에서 딥러닝 모델을 돌릴때 적절한 data augmentation을 통해 모델이 가져야 하는 invariance property를 가지게 되면(data에 국한되지 않은 general property)  generalization에 성능이 좋다고 밝혀냈다.  텍스트에서는 철자들이 엄격한 순서와 의미를 가지고 있기에ㄹ 이미지처리에서와 같이 signal을 바로 변환해줄 수는 없다. 텍스트 augmentation의 제일 좋은 방법은 사람이 rephrase하는 것이지만, 이는 현실적으로 불가능하다.(data augmentation이란? 쉽게 말해 model을 좀더 robust하게 만들기 위해 같은 데이터에 다양한 변화를 약간씩 주는것. 자세한 이해는 [여기](http://nmhkahn.github.io/CNN-Practice)) 따라서 대부분은 그 단어나 구절을 유의어(synonyms)로 바꾸는 형식의 data augmentation을 하였다.

유의어는 온라인 영어사전인 English thesaurus에서 가져왔다. 여기에는 여러 유의어가 많이 쓰이는 순으로 정렬되어 제시되는데, 몇개의 단어를 바꿀것인지 또 몇번째 순위의 유의어로 바꿀것인지는 각각 0.5확률의 기하분포를 따르게 하였다.

### 3. Comparison Models

#### 3.1 Traditional Methods

전통적인 방법으로는 multinomial logistic regression이 쓰였다.

- Bag-of-words and its TF-IDF : 상위 빈도 50000개의 단어들을 가지고 출현수를 단어의 feature로한 bag-of-words와 출현수 대신 TF-IDF로 한 모델. 여기서 IDF는 train set의 전체 sample중 해당 단어를 가지고 있는 sample로 계산.
- Bag-of-ngrams and its TF-IDF : 5-grams 까지 중 가장 frequent한 n-gram 500,000개. TF-IDF는 동일한 과정
- Bag-of-means on word embedding : train data에 word2vec을 사용한 것에 k-means clustering을 하여, 분류하는것. 5회이상 출현한 모든 단어를 고려하였고 embedding의 demension은 300이었다. 각 bag-of-means(cluster를 말하는듯)의 평균 feature는 count로 하였는데, 5000이 평균이었다.

#### 3.2 Deep Learning Methods

딥러닝의 비교대상으로는 word-based CNN과 LSTM을 하였다.

- word-based CNN : pretrained word representation(word2vec과 lookup table)을 사용했고 embedding size는 똑같이 300이다.(lookup table은 미리 있는 값으로 각 특정값을 매칭해주는 표인듯. 영상처리에 쓰인다. [참고](http://terms.naver.com/entry.nhn?docId=839052&cid=42344&categoryId=42344)) 비교를 위해 char-CNN과 레이어 수나 output size는 같다.
- LSTM : 역시 word-based이고 pretrained word2vec으로 300차원 embedding.  이 모델은 모든 LSTM cell에서 나온 값을 평균내어 feature vector로 삼고, 이를 가지고 multinomial logistic regression을 하였다. 

부가적으로 알파벳 대/소문자를 구분해보았는데, 이는 대체로 성능이 더 좋지 않았다. 아마도 대/소문자에 semantic 차이가 없었기에 regularization으로써 기능을 하지 않았나 추측한다.

### 4. Large-scale Datasets and Results

text data중에는 CNN의 성능을 확인할만한 large-scale datasets 이 없었기에 여기저기서 따와서 만들었다.

거기에는 news article(내용과 topic label), DBPedia, Yelp review(리뷰 내용과 별몇개인지, 혹은 호1~불호4-즉 2개의 dataset을 여기서 구함) Yahoo! (질문과 답 내용, label) Amazon review (yelp과 같이 review와 별점, 호1~불호4)가 쓰임. **데이터의 양**은 뒤로갈 수록 방대함.(large-scale일 수록 CNN이 잘할것!) 여러가지 모델들의 최종 성능은 다음과 같다.![char-cnn-result](C:\Users\admin\내파일\3-2.5\스터디\NLP스터디\data\char-cnn-result.PNG)

파랑이 best, 빨강이 worst

### 5. Discussion

아래 그림은 각 method별, 각 주제별 Char-CNN과의 오류율 차이%로 나타낸것. (양의 방향으로 막대가 간게 Char-CNN이 잘한것)

![char-cnn-graph](C:\Users\admin\내파일\3-2.5\스터디\NLP스터디\data\char-cnn-graph.PNG)

- Character-level ConvNet is an effective method: word를 찾을 필요 없다는 점에서, 언어 데이터를 다른 데이터와 같은 방식으로 다룰 수 잇다는 강점이 잇다.
- Dataset size에 따라 데이터 사이즈가 작을때에는 n-gram TFIDF가 여전히 잘했다. 그러나 데이터 scale이 많아질수록 char-CNN이 잘했다.
- ConvNets may work well for user-generated data : Amazon같이 curated 되지 않은 user-generated text (신경쓰지 않은, 막 쓴, 정도의 의미)에서 char-CNN이 더 잘 작동하였다. 이는 현실문제에 더 작용될 가능성을 의미하지만, 논문에서 추가적인 실험은 하지 않았기에 확신하지 못함.
- 알파벳 대/소를 구분한것이 데이터 양이 많을수록 더 잘 못했는데, 아마 regularzation effect가 아닌가 '추측'한다.
- semantic 분류를 할것인지(아마존과 yelp), 주제 분류를 할것인지(다른 데이터들)의 차이에 따른 성능은 크게 차이가 없었다.
- Bag-of-means is a misuse of word2vec : word2vec을 단순 분포로 삼아 classification을 하는것은 잘못된 활용인거같다. 결과가 넘 안좋다.
- There is no free lunch : 모든 경우에 뛰어난 방법은 없었다. 우리 결과를 보고 적용에 참고해라

### 6. Conclusion and Outlook

Char-CNN은 효과가 있다. 그러나 데이터셋 크기, 어떤 철자를 고를것 등에 따라 차이가 있을것이다.







