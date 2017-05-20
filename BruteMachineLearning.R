# starting
rm(list=ls())

library(caret)
library(devtools)
library(caretEnsemble)
library(doSNOW)

DataSet <- read.csv("Data//1. MOdelling challenge//trainingset.csv")


trainData <- DataSet[,c(5,8:38)]

trainData$coast_length[ is.na(trainData$coast_length)] = 0

fitControl <- trainControl(method = "cv",
                           number = 5
)

Grid <- expand.grid( n.trees = seq(10,1000,10), interaction.depth = c(8), shrinkage = seq(0.1), n.minobsinnode = seq(10,10,10))

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

model <- train(comp_damage_houses~., data = trainData,
               trControl = fitControl,
               method = "gbm",
               verbose = FALSE,
               tuneGrid = Grid,
               metric = "Rsquared"
)

stopCluster(cl)

plot(model)


