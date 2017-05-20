# Test
rm(list = ls())

df_typhoon <- read.csv("Data//1. Modelling challenge//trainingset.csv")
df_typhoon$coast_length[is.na(df_typhoon$coast_length)] <- 0
df_typhoon$n_households <- log(df_typhoon$n_households) # log transformation
df_typhoon <- dplyr::select(df_typhoon, -x_pos, -y_pos, -typhoon_name, -admin_L3_name, -comp_damage_houses, -total_damage_houses, -pop_15, -perimeter, -admin_L2_code, -admin_L3_code)

library(glmmLasso); library(dplyr)
library(MASS);library(nlme);library(DiceKriging)
library(doSNOW)

test_typhoon <- as.character(unique(df_typhoon$typhoon_name)[1])
df_typhoon[,5:36]<-scale(df_typhoon[,5:36],center=T,scale=T)
df_train <- filter(df_typhoon, !typhoon_name==test_typhoon)
df_test <- filter(df_typhoon, typhoon_name==test_typhoon)

### Model for partially destroyed:
df_train_part <- df_typhoon

modeling_method <- "All"
scaling_method <- NULL
training_vector <- NULL

ModelBuilder <- function(YX.Train) {
  Time <- array(0, dim = c(3))
  
  cat("Starting iteration at "); print(Sys.time())
  Time[1] = Sys.time()
  Model.OLS <- lm(YX.Train[,1]~., data = YX.Train)
  Model.Kriging <- km(formula = ~., design = YX.Train[,-1], response = YX.Train[,1], covtype = "gauss", nugget = 1e-8 * var(YX.Train[,1]), control = list(pop.size = 200, maxit = 300))

  cat("Finished iteration at "); print(Sys.time())
  Time[2] = Sys.time()
  Time[3] = Time[2] - Time[1]
  
  Models <- list(Kriging = Model.Kriging, OLS = Model.OLS,
                 Time = Time)
  
  return(Models)
}

Predictor <- function(YX.Test, Models, ncomps = 20) {
  #This function is not very flexible, as I will assume it only gets called after running the above function first.
  Kriging.Preds <- predict(Models$Kriging, newdata = YX.Test[,-1], type = "SK")
  OLS.Preds <- predict(Models$OLS, newdata = YX.Test[,-1])
  
  Predictions = list(Y = YX.Test$Y, Y.PLS = 1, Kriging = Kriging.Preds, OLS = OLS.Preds, PLS = PLS.Preds, PLS_U = PLS_U.Preds)
  
  return(Predictions)
}


#Model Training
Rows.R1 <- sample(c(1:nrow(df_train_part)),700)
Rows.R2 <- sample(c(1:nrow(df_train_part)),700)
Rows.R3 <- sample(c(1:nrow(df_train_part)),700)
Rows.R4 <- sample(c(1:nrow(df_train_part)),700)
Rows.R5 <- sample(c(1:nrow(df_train_part)),700)
Rows.R6 <- sample(c(1:nrow(df_train_part)),700)
Rows.R7 <- sample(c(1:nrow(df_train_part)),700)
Rows.R8 <- sample(c(1:nrow(df_train_part)),700)
Rows.R9 <- sample(c(1:nrow(df_train_part)),700)
Rows.R0 <- sample(c(1:nrow(df_train_part)),700)


YX.Train <- df_train_part[c(Rows.R1),]
Run_1 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R2),]
Run_2 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R3),]
Run_3 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R4),]
Run_4 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R5),]
Run_5 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R6),]
Run_6 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R7),]
Run_7 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R8),]
Run_8 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R9),]
Run_9 <- ModelBuilder(YX.Train)
YX.Train <- df_train_part[c(Rows.R0),]
Run_0 <- ModelBuilder(YX.Train)

# YXS <- df_train_part
# df_train_part <- df_train_part[1:601,]

YX.Test       <- df_train_part[-c(Rows.R1),c(1,Temp)]
Run_1.Preds.R <- Predictor(YX.Test, Run_1)
YX.Test       <- df_train_part[-c(Rows.R2),c(1,Temp)]
Run_2.Preds.R <- Predictor(YX.Test, Run_2)
YX.Test       <- df_train_part[-c(Rows.R3),c(1,Temp)]
Run_3.Preds.R <- Predictor(YX.Test, Run_3)
YX.Test       <- df_train_part[-c(Rows.R4),c(1,Temp)]
Run_4.Preds.R <- Predictor(YX.Test, Run_4)
YX.Test       <- df_train_part[-c(Rows.R5),c(1,Temp)]
Run_5.Preds.R <- Predictor(YX.Test, Run_5)
YX.Test       <- df_train_part[-c(Rows.R6),c(1,Temp)]
Run_6.Preds.R <- Predictor(YX.Test, Run_6)
YX.Test       <- df_train_part[-c(Rows.R7),c(1,Temp)]
Run_7.Preds.R <- Predictor(YX.Test, Run_7)
YX.Test       <- df_train_part[-c(Rows.R8),c(1,Temp)]
Run_8.Preds.R <- Predictor(YX.Test, Run_8)
YX.Test       <- df_train_part[-c(Rows.R9),c(1,Temp)]
Run_9.Preds.R <- Predictor(YX.Test, Run_9)
YX.Test       <- df_train_part[-c(Rows.R0),c(1,Temp)]
Run_0.Preds.R <- Predictor(YX.Test, Run_0)

sum((Run_1.Preds.R$Y-Run_1.Preds.R$Kriging$mean)^2)/length(Run_1.Preds.R$Y)
sum((Run_2.Preds.R$Y-Run_2.Preds.R$Kriging$mean)^2)/length(Run_2.Preds.R$Y)
sum((Run_3.Preds.R$Y-Run_3.Preds.R$Kriging$mean)^2)/length(Run_3.Preds.R$Y)
sum((Run_4.Preds.R$Y-Run_4.Preds.R$Kriging$mean)^2)/length(Run_4.Preds.R$Y)
sum((Run_5.Preds.R$Y-Run_5.Preds.R$Kriging$mean)^2)/length(Run_5.Preds.R$Y)
sum((Run_6.Preds.R$Y-Run_6.Preds.R$Kriging$mean)^2)/length(Run_6.Preds.R$Y)
sum((Run_7.Preds.R$Y-Run_7.Preds.R$Kriging$mean)^2)/length(Run_7.Preds.R$Y)
sum((Run_8.Preds.R$Y-Run_8.Preds.R$Kriging$mean)^2)/length(Run_8.Preds.R$Y)
sum((Run_9.Preds.R$Y-Run_9.Preds.R$Kriging$mean)^2)/length(Run_9.Preds.R$Y)
sum((Run_0.Preds.R$Y-Run_0.Preds.R$Kriging$mean)^2)/length(Run_0.Preds.R$Y)


# Test <- data.frame(cbind(sort(abs(Run_1.Preds.R$Y - Run_1.Preds.R$Kriging$mean)),
#                    sort(abs(Run_2.Preds.R$Y - Run_2.Preds.R$Kriging$mean)),
#                    sort(abs(Run_3.Preds.R$Y - Run_3.Preds.R$Kriging$mean)),
#                    sort(abs(Run_4.Preds.R$Y - Run_4.Preds.R$Kriging$mean)),
#                    sort(abs(Run_5.Preds.R$Y - Run_5.Preds.R$Kriging$mean)),
#                    sort(abs(Run_6.Preds.R$Y - Run_6.Preds.R$Kriging$mean)),
#                    sort(abs(Run_7.Preds.R$Y - Run_7.Preds.R$Kriging$mean)),
#                    sort(abs(Run_8.Preds.R$Y - Run_8.Preds.R$Kriging$mean)),
#                    sort(abs(Run_9.Preds.R$Y - Run_9.Preds.R$Kriging$mean)),
#                    sort(abs(Run_0.Preds.R$Y - Run_0.Preds.R$Kriging$mean))))
# names(Test) <- c("M1","M2","M3","M4","M5","M6","M7","M8","M9","M0")

max(abs(Run_1.Preds.R$Y - Run_1.Preds.R$Kriging$mean))
max(abs(Run_2.Preds.R$Y - Run_2.Preds.R$Kriging$mean))
max(abs(Run_3.Preds.R$Y - Run_3.Preds.R$Kriging$mean))
max(abs(Run_4.Preds.R$Y - Run_4.Preds.R$Kriging$mean))
max(abs(Run_5.Preds.R$Y - Run_5.Preds.R$Kriging$mean))
max(abs(Run_6.Preds.R$Y - Run_6.Preds.R$Kriging$mean))
max(abs(Run_7.Preds.R$Y - Run_7.Preds.R$Kriging$mean))
max(abs(Run_8.Preds.R$Y - Run_8.Preds.R$Kriging$mean))
max(abs(Run_9.Preds.R$Y - Run_9.Preds.R$Kriging$mean))
max(abs(Run_0.Preds.R$Y - Run_0.Preds.R$Kriging$mean))

# Backre:
Run_1.Results <- list(Run_1,Run_1.Preds.R)
Run_2.Results <- list(Run_2,Run_2.Preds.R)
Run_3.Results <- list(Run_3,Run_3.Preds.R)
Run_4.Results <- list(Run_4,Run_4.Preds.R)
Run_5.Results <- list(Run_5,Run_5.Preds.R)
Run_6.Results <- list(Run_6,Run_6.Preds.R)
Run_7.Results <- list(Run_7,Run_7.Preds.R)
Run_8.Results <- list(Run_8,Run_8.Preds.R)
Run_9.Results <- list(Run_9,Run_9.Preds.R)
Run_0.Results <- list(Run_0,Run_0.Preds.R)