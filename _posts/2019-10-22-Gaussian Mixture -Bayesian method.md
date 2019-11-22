---
layout: post
title: "[Code implementation] Bayesian Gaussian mixture"
categories:
  - 머신러닝
tags:
  - Gaussian mixture
  - Bayesian
  - implementation
comment: true
---



베이지안 방법론을 통해 gaussian mixture를 푸는 코드이다. Frequentist의 접근법처럼 역시나 group indicator를 latent로 두어, 이에 대한 prior가 또 들어간다. group indicator는 multi-category를 가진 categorical variable이기에,  multi-category에 대해 conjugate한 Dirichlet distribution이 prior로 사용된다. 또한 추정해야할 모수가 group indicator, 각 분포의 mu, sigma로 여러개가 있어, MCMC를 통해 한번에 sampling하기 힘들다. 따라서 parameter가 많은 경우 자주 활용되는 Gibbs sampler를 이용하여, 각각의 parameter에 대해 conditional distribution에서 sampling을 한다.

베이지안 방법론으로 적합을 할 시, 각 parameter들에 대한 estimation이 주어진 MCMC sample들에 기반하여 쉽게 이뤄질 수 있기에, 기존 frequentist의 접근에선 수식적으로 까다로웠던 많은 부분들이 해결된다. 그러나 적절하지 못한 prior를 지정해줄 경우, 모델이 좋은 추정을 하지 못할 가능성이 존재한다.

# Library



```R
options(repr.plot.width=5, repr.plot.height=5)
library(MCMCpack)#for dirichlet

set.seed(1013)
rm(list = ls())

setwd('C:/Users/admin/내파일/대학원1학기/베이즈/HW2')
data=read.table('Pset2data.txt',header = T)
#Y=data$Gene2
Y=data$Gene1

hist(Y,nclass=100,main="Normal Mixture")
```


![output_3_0](https://user-images.githubusercontent.com/31824102/69405816-fefa2e00-0d43-11ea-8a70-8a07f8e0cb8e.png)


# Function


```R
gibbs_sampler=function(K=2, n_iter=1e3, Y=Y,no_library=FALSE){
  #K=2; n_iter=2e3; Y=data$Gene1;no_library=TRUE
  
  # create variable for saving each trace-----------------------------------
  z_trace=matrix(rep(NA,length(Y)*n_iter),ncol = n_iter)
  pi_trace=matrix(rep(NA,K*n_iter),ncol = n_iter)
  mu_trace=matrix(rep(NA,K*n_iter),ncol = n_iter)
  sig2_trace=matrix(rep(NA,K*n_iter),ncol = n_iter)
  llikelihood_trace=matrix(rep(NA,1*n_iter),ncol = n_iter)
  
  
  
  # initialize_value-----------------------------------
  
  z_trace[,1] = (ceiling(runif(n = length(Y),0,K)))
  pi_trace[,1] = rep(1/K,K)#rdirichlet(1,alpha = rep(1/K,K))
  pi_save=pi_trace[,1]
  mu_trace[,1] = rnorm(K,0,100)
  sig2_trace[,1] = rep(0.3,K)#rinvgamma(K,3,100)
  
  #iterate using Gibbs Sampling-----------------------------------
  for(t in 1:n_iter){
    if(t==1){
      pi_lv = pi_trace[,t]
      sig2_lv = sig2_trace[,t]
      mu_lv = mu_trace[,t]
      z_lv=z_trace[,t]
      eta=c(table(z_lv),pi_lv,mu_lv,sig2_lv)
      cat('z_lv,pi_lv,mu_lv,sig2_lv is : ',eta,'\n')
      next}
    
    
    #pi step------------    
    #lv for latest_value
    mu_lv = mu_trace[,t-1] 
    sig2_lv = sig2_trace[,t-1]
    z_lv = z_trace[,t-1]
    
    nk=rep(NA,K)
    for(i in 1:K){
      nk[i]=sum(z_lv==i)
    }
    
    #sampling pi, and save (accept prob for Gibbs is 1)
    #pi_trace[,t] = rdirichlet(1,nk+1)
    
    
    #when you cannot use rdirichlet---------------------------------------
    if(no_library==TRUE){
      stopifnot(K==2)
      n1=rbeta(n = 1,shape1 = nk[1]+1/2,shape2 = nk[2]+1/2)
      n2=1-n1
      pi_trace[,t]=c(n1,n2)
    }
    #when you cannot use rdirichlet---------------------------------------
    else{
      pi_trace[,t]= rdirichlet(1,alpha = (nk+1/K))
    }
    
    
    pi_lv = pi_trace[,t]
    
    #mu step----------
    condi_mean = rep(NA,K)
    for(i in 1:K){
      condi_mean[i]=(sum(Y[z_lv==i])/sig2_lv[i])/(nk[i]/sig2_lv[i]+1/100^2)
      stopifnot(sum(z_lv==i)==nk[i])
    }
    
    
    condi_sig2 = rep(NA,K)
    for(i in 1:K){
      condi_sig2[i]=1/(nk[i]/sig2_lv[i]+1/100^2)
    } 
    
    for(i in 1:K){
      #sampling mu, and save (accept prob for Gibbs is 1)
      mu_trace[i,t]=rnorm(1, mean = condi_mean[i], sd = sqrt(condi_sig2[i]))
    }
    
    mu_lv = mu_trace[,t]
      
    #zstep------------
    cat_prob=matrix(rep(NA,K*length(Y)),nrow = K)
    for(i in 1:K){
      cat_prob[i,]=pi_lv[i]*1/sqrt(sig2_lv[i])*exp(-1/2*(mu_lv[i]-Y)^2/sig2_lv[i])
    }
    cat_prob=(cat_prob/matrix(rep(colSums(cat_prob),each=K),nrow = K)) #normalize to sum1
    
    for(i in 1:length(Y)){
      #i=11
      z_trace[i,t] = sample(1:K, size=1,prob=cat_prob[,i],replace=TRUE)
      #z_trace[i,t] = which.max(rmultinom(1, 1, cat_prob[,i]))
    }
    
    
    z_lv=z_trace[,t]
    nk=rep(NA,K)
    for(i in 1:K){
      nk[i]=sum(z_lv==i)
    }
    
    #sig2 step----------
    #sampling sig2, and save (accept prob for Gibbs is 1)
    for(i in 1:K){
      if(no_library==TRUE){
        sig2_trace[i,t]=1/rgamma(1,100+1/2*sum((Y[z_lv==i]-mu_lv[i])^2),length(Y[z_lv==i])/2+3)
      }
      else{
      sig2_trace[i,t]=rinvgamma(1,shape = 100+1/2*sum((Y[z_lv==i]-mu_lv[i])^2),scale = length(Y[z_lv==i])/2+3)
      }
    }
    
    
    sig2_lv = sig2_trace[,t]
    eta=c(table(z_lv),round(pi_lv,2),round(mu_lv,2),sig2_lv)
    
    
    llikelihood=0
    for(k in 1:K){#k=2
      llikelihood=llikelihood+( #(Y-mu_lv[k])^2/sig2_lv[k]
        sum(log(( (dnorm(Y,mean = mu_lv[k],sd = sqrt(sig2_lv[k])))*pi_lv[k] )^(z_lv==k)))+
          (log((pi_lv[k])^(K-1)))+
          (-1/2e4*mu_lv[k]^2)+log(sig2_lv[k]^(-4))+(-100/sig2_lv[k])
      )
    }
    
    llikelihood_trace[,t]=llikelihood
    
    #print data -----------------------------------
    if((t<=10)|(t%%500==0)){#check first 10 steps or every 500 steps
      cat('for',t,', z_lv,pi_lv,mu_lv,sig2_lv is : ',eta,'\n')
    }
    
  }
  res=list(z_trace,pi_trace,mu_trace,sig2_trace,llikelihood_trace)
  names(res)=c('z_trace','pi_trace','mu_trace','sig2_trace','llikelihood_trace')
  
  return(res)
}

```

# Simulte & Convergence check


```R
inp_niter=2e3
inp_K=2
set.seed(1014)

#iterage multiple chain to compare result 
tmp1=gibbs_sampler(K=inp_K, n_iter=inp_niter, Y=data$Gene1)
tmp2=gibbs_sampler(K=inp_K, n_iter=inp_niter, Y=data$Gene1)
tmp3=gibbs_sampler(K=inp_K, n_iter=inp_niter, Y=data$Gene1)
```

    z_lv,pi_lv,mu_lv,sig2_lv is :  40 32 0.5 0.5 -191.0087 -27.92372 0.3 0.3 
    for 2 , z_lv,pi_lv,mu_lv,sig2_lv is :  38 34 0.56 0.44 0.22 0.59 0.2165637 0.1509482 
    for 3 , z_lv,pi_lv,mu_lv,sig2_lv is :  43 29 0.44 0.56 0.02 0.83 0.2510533 0.1417167 
    for 4 , z_lv,pi_lv,mu_lv,sig2_lv is :  57 15 0.6 0.4 0.05 1.01 0.3747806 0.1155301 
    for 5 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.81 0.19 0.15 1.89 0.3290877 0.06525928 
    for 6 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.83 0.17 -0.04 2.48 0.345712 0.07573328 
    for 7 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.88 0.12 0.07 2.4 0.2992068 0.07914449 
    for 8 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.8 0.2 0.2 2.41 0.3677189 0.09115946 
    for 9 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.86 0.14 -0.06 2.33 0.3403325 0.09380707 
    for 10 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.85 0.15 0.05 2.52 0.4521353 0.0782955 
    for 500 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.92 0.08 0.19 2.5 0.3426829 0.07064485 
    for 1000 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.84 0.16 0.08 2.45 0.3653833 0.07525163 
    for 1500 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.81 0.19 -0.01 2.51 0.255525 0.09068538 
    for 2000 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.88 0.12 -0.01 2.51 0.300147 0.07983018 
    z_lv,pi_lv,mu_lv,sig2_lv is :  35 37 0.5 0.5 182.5481 115.7144 0.3 0.3 
    for 2 , z_lv,pi_lv,mu_lv,sig2_lv is :  30 42 0.47 0.53 0.39 0.33 0.1380176 0.2655488 
    for 3 , z_lv,pi_lv,mu_lv,sig2_lv is :  22 50 0.49 0.51 0.53 0.29 0.1321068 0.2340113 
    for 4 , z_lv,pi_lv,mu_lv,sig2_lv is :  27 45 0.25 0.75 0.06 0.56 0.1543063 0.2010244 
    for 5 , z_lv,pi_lv,mu_lv,sig2_lv is :  28 44 0.35 0.65 -0.03 0.69 0.1593382 0.205812 
    for 6 , z_lv,pi_lv,mu_lv,sig2_lv is :  41 31 0.46 0.54 0.02 0.55 0.2156582 0.1554942 
    for 7 , z_lv,pi_lv,mu_lv,sig2_lv is :  44 28 0.48 0.52 -0.13 0.93 0.2514632 0.143165 
    for 8 , z_lv,pi_lv,mu_lv,sig2_lv is :  60 12 0.68 0.32 -0.03 1.18 0.261554 0.08524478 
    for 9 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.79 0.21 -0.01 2.24 0.3305866 0.07985622 
    for 10 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.84 0.16 0.01 2.31 0.3925597 0.07986116 
    for 500 , z_lv,pi_lv,mu_lv,sig2_lv is :  63 9 0.88 0.12 0.04 2.63 0.3045323 0.07649907 
    for 1000 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.88 0.12 0.06 2.42 0.3116635 0.069078 
    for 1500 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.89 0.11 0.11 2.51 0.3665746 0.09550593 
    for 2000 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.84 0.16 0.12 2.39 0.3540361 0.07251946 
    z_lv,pi_lv,mu_lv,sig2_lv is :  36 36 0.5 0.5 -5.090603 -59.35514 0.3 0.3 
    for 2 , z_lv,pi_lv,mu_lv,sig2_lv is :  38 34 0.48 0.52 0.31 0.52 0.1765433 0.1678907 
    for 3 , z_lv,pi_lv,mu_lv,sig2_lv is :  53 19 0.52 0.48 0.03 0.76 0.3007423 0.09123187 
    for 4 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.8 0.2 0.03 1.5 0.336586 0.06957606 
    for 5 , z_lv,pi_lv,mu_lv,sig2_lv is :  63 9 0.83 0.17 0.17 2.42 0.3289166 0.08441534 
    for 6 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.87 0.13 0.03 2.62 0.3206282 0.06735946 
    for 7 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.77 0.23 0.12 2.53 0.3357813 0.07837687 
    for 8 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.88 0.12 0.13 2.22 0.3373437 0.084334 
    for 9 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.82 0.18 0.04 2.6 0.2830813 0.08200646 
    for 10 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.78 0.22 0.03 2.67 0.3110761 0.07652402 
    for 500 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.9 0.1 0.04 2.45 0.3481991 0.07615492 
    for 1000 , z_lv,pi_lv,mu_lv,sig2_lv is :  62 10 0.92 0.08 0.07 2.56 0.3003156 0.07968798 
    for 1500 , z_lv,pi_lv,mu_lv,sig2_lv is :  61 11 0.85 0.15 0.07 2.48 0.3360475 0.101472 
    for 2000 , z_lv,pi_lv,mu_lv,sig2_lv is :  63 9 0.88 0.12 0.13 2.54 0.3583726 0.05720203 



```R
target=c('z_trace','pi_trace','mu_trace','sig2_trace','llikelihood_trace')[3]

trace_mcmc1=mcmc(t(tmp1[[target]])[500:inp_niter,])
#summary(trace_mcmc1)
plot(trace_mcmc1)

trace_mcmc2=mcmc(t(tmp2[[target]])[500:inp_niter,])
plot(trace_mcmc2)

trace_mcmc3=mcmc(t(tmp3[[target]])[500:inp_niter,])
plot(trace_mcmc3)
```


![output_8_0](https://user-images.githubusercontent.com/31824102/69405817-fefa2e00-0d43-11ea-9e5b-3c309e85df48.png)
![output_8_1](https://user-images.githubusercontent.com/31824102/69405818-ff92c480-0d43-11ea-88c6-ee9fb8a803b8.png)
![output_8_2](https://user-images.githubusercontent.com/31824102/69405819-ff92c480-0d43-11ea-8987-48d21b86e5b0.png)





수렴을 확인하기 위해 chain을 여러번 돌린다. 각 group indicator (여기선 2개의 group을 주었으니 1,2)는 임의적인 것이라, 순서가 바뀔 수 있다. 수렴을 확인하기 위한 다양한 방법이 있지만, 그중에서 multiple chain간의 with in variance와 between variance를 비교하는 Gelman Rubin statistics를 이용하였다. 보통 1.1이하의 값을 띄면 수렴했다고 판단한다.

```R
# convergence check with Gelman Rubin statistics
conv_m1=mcmc(t(tmp1[[target]])[500:inp_niter,2])
conv_m2=mcmc(t(tmp2[[target]])[500:inp_niter,2])
conv_m3=mcmc(t(tmp3[[target]])[500:inp_niter,2])

#combinedchains = mcmc.list(trace_mcmc1, trace_mcmc2,trace_mcmc3)
combinedchains = mcmc.list(conv_m1,conv_m2,conv_m3)
plot(combinedchains)
gelman.diag(combinedchains)
gelman.plot(combinedchains)
```


    Potential scale reduction factors:

         Point est. Upper C.I.
    [1,]          1          1

![output_9_1](https://user-images.githubusercontent.com/31824102/69405820-ff92c480-0d43-11ea-8d44-5bb7227cb774.png)
![output_9_2](https://user-images.githubusercontent.com/31824102/69405821-002b5b00-0d44-11ea-8585-efc99abfb3c4.png)



```R
### generate samples using posterior inference
one_tmp=tmp3
posterior_mean_pi=colMeans(t(one_tmp[['pi_trace']]))
posterior_mean_mu=colMeans(t(one_tmp[['mu_trace']]))
posterior_mean_sig2=colMeans(t(one_tmp[['sig2_trace']]))

y_sim=rep(0,length(Y))
for(i in 1:length(Y)){
  z_tmp = sample (seq(1,inp_K), size=1, replace=T, prob=posterior_mean_pi)
  for(k in 1:inp_K){
    y_sim[i]=rnorm(1,mean = posterior_mean_mu[z_tmp],sd = sqrt(posterior_mean_sig2[z_tmp]))
  }
}
par(mfrow=c(1,2))
hist(Y,nclass=50)
hist(y_sim,nclass=50)
```


![output_10_0](https://user-images.githubusercontent.com/31824102/69405823-002b5b00-0d44-11ea-8e87-3c5a53e8f077.png)

덧붙여 EM algorithm으로 동일한 데이터에 대해 적합해보았다. 

```R
###data check with traditional K-mixture-----------------------------
K_mixture=function(y=data$Gene1,inp_K=3){
  n=length(y)
  
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
  return(eta)
}
```


```R
mixture_res=K_mixture(y = data$Gene1,inp_K=2)
round(mixture_res,3)#c(pi,mu,sig2)
```

    [1]  0.5000000  0.5000000 -0.1572504  4.3961548  2.0000000
    [1] 0.8904799 0.1095201 0.1807714 2.3333463 0.3589446
    [1] 0.85349044 0.14650956 0.06586353 2.45927732 0.09452766
    [1] 0.84722265 0.15277735 0.05209006 2.43746643 0.07434040
    [1] 0.84722223 0.15277777 0.05208921 2.43746456 0.07433937
    converge!


![output_12_2](https://user-images.githubusercontent.com/31824102/69405824-002b5b00-0d44-11ea-9a45-c9c41cae47fa.png)

