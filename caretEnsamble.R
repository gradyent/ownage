# starting
rm(list=ls())

library(caret)
library(caretEnsemble)
library(doSNOW)

DataSet <- read.csv("Data//1. MOdelling challenge//trainingset.csv")

trainData <- DataSet[,c(5,8:38)]

trainData$coast_length[ is.na(trainData$coast_length)] = 0


folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, 
                          returnResamp='none',
                          savePredictions=TRUE, 
                          verboseIter=TRUE,
                          index=createMultiFolds(trainData$comp_damage_houses, k=folds, times=repeats))
PP <- c('center', 'scale')

cl <- makeCluster(3, type = "SOCK")

registerDoSNOW(cl)

#Train some models
all.models <- caretList(trainData[-1], trainData$comp_damage_houses, trControl=myControl, tuneList=list(
  model1 <- caretModelSpec(method='gbm',tuneGrid=expand.grid(.n.trees=500, .interaction.depth=15, .shrinkage = 0.01, .n.minobsinnode = c(10))),
  model2 <- caretModelSpec( method='blackboost'),
  model3 <- caretModelSpec( method='parRF'),
  model5 <- caretModelSpec( method='knn', preProcess=PP),
  model6 <- caretModelSpec( method='earth', preProcess=PP),
  model7 <- caretModelSpec( method='glm',  preProcess=PP),
  model8 <- caretModelSpec( method='svmRadial', preProcess=PP),
  model9 <- caretModelSpec( method='gam', preProcess=PP),
  model10 <- caretModelSpec( method='glmnet', preProcess=PP)
))

#Make a list of all the models
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$ROC)))

#Make a greedy ensemble - currently can only use RMSE
greedy <- caretEnsemble(all.models, iter=1000L)
sort(greedy$weights, decreasing=TRUE)
greedy$error

#Make a linear regression ensemble
linear <- caretStack(all.models, method='glm', trControl=trainControl(method='cv'))
linear$error


stopCluster(cl)

plot(model)

