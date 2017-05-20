# starting
rm(list=ls())

library(caret)
library(caretEnsemble)
library(doSNOW)
library(dplyr)
library(MASS);library(nlme)

df_tycoon <- read.csv("Data//1. Modelling challenge//trainingset.csv")
df_tycoon$coast_length[is.na(df_tycoon$coast_length)] <- 0


test_tycoon <- as.character(unique(df_tycoon$typhoon_name)[1])
df_train <- filter(df_tycoon, !typhoon_name==test_tycoon)
df_test <- filter(df_tycoon, typhoon_name==test_tycoon)

df_test<- df_test[,c(5,8:38)]
df_train<- df_train[,c(5,8:38)]


fitControl <- trainControl(method = "cv",
                           number = 5
)

Grid <- expand.grid( n.trees = seq(10,300,20), interaction.depth = c(2), shrinkage = seq(0.1), n.minobsinnode = seq(10,10,10))

cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)

model <- train(comp_damage_houses~., data = df_train,
               trControl = fitControl,
               method = "gbm",
               verbose = FALSE,
               tuneGrid = Grid,
               metric = "Rsquared"
)

stopCluster(cl)

plot(model)


predicted <- predict(model, newdata = df_test)

plot(df_test$comp_damage_houses, predicted)
cor(df_test$comp_damage_houses, predicted)


