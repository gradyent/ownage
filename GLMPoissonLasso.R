
# Test
rm(list = ls())

df_tycoon <- read.csv("Data//1. Modelling challenge//trainingset.csv")
df_tycoon$coast_length[is.na(df_tycoon$coast_length)] <- 0
df_tycoon$n_households <- log(df_tycoon$n_households) # log transformation
df_tycoon <- select(df_tycoon, -x_pos, -y_pos)

library(glmmLasso); library(dplyr)
library(MASS);library(nlme)

test_tycoon <- as.character(unique(df_tycoon$typhoon_name)[1])
df_tycoon[,5:36]<-scale(df_tycoon[,5:36],center=T,scale=T)
df_train <- filter(df_tycoon, !typhoon_name==test_tycoon)
df_test <- filter(df_tycoon, typhoon_name==test_tycoon)

### Model for completely destroyed:
df_train_comp <- select(df_train, -admin_L3_name, -part_damage_houses, -total_damage_houses)

## generalized additive mixed model
## grid for the smoothing parameter
lambda <- seq(500,0,by=-5)
family = poisson(link = log)

################## More Elegant Method ############################################
## Idea: start with big lambda and use the estimates of the previous fit (BUT: before
## the final re-estimation Fisher scoring is performed!) as starting values for the next fit;
## make sure, that your lambda sequence starts at a value big enough such that all covariates are
## shrinked to zero;

## Using BIC (or AIC, respectively) to determine the optimal tuning parameter lambda
  lambda <- seq(100,0,by=-1)

BIC_vec<-rep(Inf,length(lambda))
family = poisson(link = log)

# specify starting values for the very first fit; pay attention that Delta.start has suitable length! 
Delta.start<-as.matrix(t(rep(0,36)))
Q.start<-0.1

n <- colnames(df_train_comp)
v_exclude <- c("typhoon_name","admin_L3_code","admin_L2_code","comp_damage_houses")
glm_form <- as.formula(paste("comp_damage_houses~", paste(n[!n %in% v_exclude], collapse="+")))

cor1 <- numeric(length(lambda))

for(j in 1:length(lambda)){
  print(paste("Iteration ", j,sep=""))
  
  glm3 <- try(glmmLasso(glm_form,rnd = list(typhoon_name=~1), 
                    family = family, data = df_train_comp, 
                    lambda=lambda[j], switch.NR=T,final.re=T,
                    control=list(start=Delta.start[j,],q_start=Q.start[j])), silent=TRUE)
  
  print(colnames(glm3$Deltamatrix)[2:7][glm3$Deltamatrix[glm3$conv.step,2:7]!=0])
  BIC_vec[j]<-glm3$bic
  Delta.start<-rbind(Delta.start,glm3$Deltamatrix[glm3$conv.step,])
  Q.start<-c(Q.start,glm3$Q_long[[glm3$conv.step+1]])
  predicted_test <- predict(glm3, newdata = df_test)
  cor1[j] <- cor(df_test$comp_damage_houses, predicted_test)
}

plot(lambda, cor1, type="l")
plot(lambda, BIC_vec)
opt3<-which.min(BIC_vec)
lambda[opt3]

glm3_final <- glmmLasso(glm_form, rnd = list(typhoon_name=~1),  
                       family = family, data = df_train_comp, lambda=150)

summary(glm3_final)

predicted_test <- predict(glm3_final, newdata = df_test)
predicted_train <- predict(glm3_final, newdata = df_train)

plot(df_test$comp_damage_houses, predicted_test)
cor(df_test$comp_damage_houses, predicted_test)

plot(df_train$comp_damage_houses, predicted_train)
cor(df_train$comp_damage_houses, predicted_train)

## plot coefficient paths
par(mar=c(6,6,4,4))
plot(lambda,Delta.start[2:(length(lambda)+1),2],ylim=c(-1,2),type="l",ylab=expression(hat(beta[j])))
lines(c(-1000,1000),c(0,0),lty=2)
for(i in 3:36){
  lines(lambda[1:length(lambda)],Delta.start[2:(length(lambda)+1),i])
}
abline(v=lambda[opt3],lty=2)

