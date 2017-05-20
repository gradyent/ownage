# starting
rm(list=ls())

library(caret)
library(caretEnsemble)
library(doSNOW)
library(dplyr)
library(caTools)

df_tycoon <- read.csv("Data//1. Modelling challenge//trainingset.csv")
df_tycoon$coast_length[is.na(df_tycoon$coast_length)] <- 0


test_tycoon <- as.character(unique(df_tycoon$typhoon_name)[1])
df_train <- filter(df_tycoon, !typhoon_name==test_tycoon)
df_test <- filter(df_tycoon, typhoon_name==test_tycoon)

df_test<- df_test[,c(5,8:38)]
df_train<- df_train[,c(5,8:38)]

folds=10
repeats=2
myControl <- trainControl(method='cv', number=folds, repeats=repeats, 
                          returnResamp='final',
                          savePredictions=TRUE, 
                          verboseIter=TRUE,
                          index=createMultiFolds(df_train$comp_damage_houses, k=folds, times=repeats))
PP <- c('center', 'scale')

cl <- makeCluster(16, type = "SOCK")

registerDoSNOW(cl)

#Train some models
all.models <- caretList(df_train[-1], df_train$comp_damage_houses,metric = "Rsquared", trControl=myControl, tuneList=list(
  model1 <- caretModelSpec(method='gbm',tuneGrid=expand.grid(.n.trees=300, .interaction.depth=2, .shrinkage = 0.01, .n.minobsinnode = c(10))),
  model2 <- caretModelSpec( method='blackboost'),
  model3 <- caretModelSpec( method='parRF'),
  model5 <- caretModelSpec( method='knn', preProcess=PP),
  model6 <- caretModelSpec( method='earth', preProcess=PP),
  model7 <- caretModelSpec( method='glm',  preProcess=PP),
  model8 <- caretModelSpec( method='svmRadial', preProcess=PP),
  #model9 <- caretModelSpec( method='gam', preProcess=PP)
  model10 <- caretModelSpec( method='glmnet', preProcess=PP)
))

#Make a list of all the models
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$Rsquared)))

greedy_ensemble <- caretEnsemble(
  all.models, 
  metric="Rsquared",
  trControl=trainControl(
    number=2,
    classProbs = F
  ))
summary(greedy_ensemble)


stopCluster(cl)


model_preds <- lapply(all.models, predict, newdata=df_test, type="raw")
model_preds <- lapply(model_preds, function(x) x[,"M"])
model_preds <- data.frame(model_preds)


predicted <- predict(greedy_ensemble, newdata = df_test)

plot(df_test$comp_damage_houses, predicted)
cor(df_test$comp_damage_houses, predicted)


