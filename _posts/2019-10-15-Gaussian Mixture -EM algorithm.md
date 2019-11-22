---
layout: post
title: "[Code implementation] Gaussian mixture"
categories:
  - 머신러닝
tags:
  - Gaussian mixture
  - EM algorithm
  - implementation
comment: true
---



데이터가 주어졌을때, 해당 데이터가 미지의 정규분포들의 혼합된 분포에서 만들어졌다는 가정하에, 해당 분포의 혼합 비율과 각 분포들의 parameter(즉, mu와 sigma), 그리고 어떤 데이터가 어느 정규분포에 속할지를 풀어내는 방법론이다. 

가장 대표적으로는 EM algorithm으로 적합을 한다. 이때, group indicator는 latent variable로써, 일종의 missing data problem으로 볼 수도 있다. 따라서 주어진 데이터들로 구해지는 observed likelihood에 기반하여 missing data(여기서는 group indicator)의 conditional expectation을 maximize하는 parameter들을 찾게 된다. 

# Data generating


```R
rm(list = ls())
options(repr.plot.width=5, repr.plot.height=5)
n=300
set.seed(1001)

y=vector(length=n)
z=rmultinom(n,1,c(0.1,0.3,0.6))

n1=sum(z[1,])
n2=sum(z[2,])
n3=sum(z[3,])

y[(z[1,]==1)]=rnorm(n1,0,1)
y[(z[2,]==1)]=rnorm(n2,4,1)
y[(z[3,]==1)]=rnorm(n3,7,1)

true_eta=c(c(0.1,0.3,0.6),c(0,4,7),1) # pi, mu, sig2

plot(density(y),main="Normal Mixture (unknown mixing proportions",xlab="y")
```


![output_2_0](https://user-images.githubusercontent.com/31824102/69406285-11c13280-0d45-11ea-8055-e39a8db5e831.png)

#  EM Itertaion start

```R

#init value
n=length(y)
inp_K=2
#init value
K=inp_K
pi=rep(1/K,K)
mu=rnorm(K,sd = 30)#c(mu1,mu2,mu3)
sig2=2 # common variance
eta=c(pi,mu,sig2)
print(eta)


repeat{
  eta0=eta
  
  term=matrix(rep(NA,length(pi)*n), ncol = length(pi)) # saving value of each distn in each column
  ## E-step
  for(i in 1:length(pi)){
    term[,i]=pi[i]*dnorm(y,mu[i],sd=sqrt(sig2))# common variance
  }
  
  z=term/rowSums(term)
  
  ## M-step
  for(i in 1:length(pi)){ #update the parameters which maximize the Q-function
    
    pi[i]=sum(z[,i])/n
    mu[i]=sum(z[,i]*y)/sum(z[,i])
    

  }
  
  mu_mat=matrix(rep(mu,n),ncol = length(pi),byrow = T)
  sig2=sum(z*(y-mu_mat)^2)/n # sum for all i,k (common variance)
  
  eta=c(pi,mu,sig2)
  
  ## Convergence criteria
  diff=(eta0-eta)^2
  print(c(eta))#,logL))
  
  if(sum(diff)<1e-7) {cat('converge!\n');break}
}
#true_eta
cat('estimated paramter:',round(eta,2),'\n True parameter:',true_eta,'\n')


```

    [1]   0.50000   0.50000 -15.56966  11.44557   2.00000
    [1]  0.0006600105  0.9993399895 -1.9043484852  5.4641191864  6.0853710365
    [1]  0.001271911  0.998728089 -0.957922404  5.467428396  6.068737972
    [1]  0.002645395  0.997354605 -0.781882072  5.475809985  6.017866002
    [1]  0.005456573  0.994543427 -0.725702903  5.493189764  5.911302780
    [1]  0.01103187  0.98896813 -0.67838249  5.52772084  5.70096930
    [1]  0.02154037  0.97845963 -0.60592134  5.59277818  5.31134601
    [1]  0.03974788  0.96025212 -0.49287727  5.70563357  4.65470960
    [1]  0.06651677  0.93348323 -0.33950502  5.87245547  3.72513681
    [1]  0.09343209  0.90656791 -0.16216272  6.03860678  2.86440850
    [1] 0.10871129 0.89128871 0.01281094 6.12356374 2.50306621
    [1] 0.1159491 0.8840509 0.1447868 6.1562836 2.4168499
    [1] 0.1198564 0.8801436 0.2268347 6.1717975 2.3928645
    [1] 0.1221463 0.8778537 0.2761977 6.1804368 2.3832596
    [1] 0.1235372 0.8764628 0.3061812 6.1855807 2.3783766
    [1] 0.1243968 0.8756032 0.3246337 6.1887308 2.3756040
    [1] 0.1249330 0.8750670 0.3361018 6.1906867 2.3739498
    [1] 0.1252693 0.8747307 0.3432760 6.1919101 2.3729383
    [1] 0.1254809 0.8745191 0.3477826 6.1926787 2.3723114
    [1] 0.1256143 0.8743857 0.3506212 6.1931628 2.3719198
    [1] 0.1256986 0.8743014 0.3524121 6.1934683 2.3716740
    [1] 0.1257518 0.8742482 0.3535431 6.1936613 2.3715193
    [1] 0.1257855 0.8742145 0.3542579 6.1937832 2.3714217
    [1] 0.1258067 0.8741933 0.3547099 6.1938603 2.3713601
    [1] 0.1258202 0.8741798 0.3549957 6.1939091 2.3713212
    converge!
    estimated paramter: 0.13 0.87 0.35 6.19 2.37 
     True parameter: 0.1 0.3 0.6 0 4 7 1 

여기서 group indicator(1,2,3)의 부여는 임의적인 것이기에, 순서는 다를 수 있다. 실제 mixture분포의 true 비율인 0.1, 0.3, 0.6을 0.13, 0.35, 0.87로, true 평균인 0,4,7을 0.35, 2.37, 6.19로 잡아내고 있음을 볼 수 있다

```R
posterior_mean_pi=pi
posterior_mean_mu=mu

posterior_mean_sig2=sig2
Y=y
y_sim=rep(0,length(Y))
for(i in 1:length(Y)){
z_tmp = sample (seq(1,inp_K), size=1, replace=T, prob=posterior_mean_pi)
for(k in 1:inp_K){
  y_sim[i]=rnorm(1,mean = posterior_mean_mu[z_tmp],sd = sqrt(posterior_mean_sig2))
}
}
par(mfrow=c(2,1))
hist(Y,nclass=50)
hist(y_sim,nclass=50)
```


![output_4_0](https://user-images.githubusercontent.com/31824102/69406286-11c13280-0d45-11ea-8ec7-3b53f3a2f234.png)

