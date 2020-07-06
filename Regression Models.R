install.packages('e1071', dependencies=TRUE)
install.packages('DMwR', dependencies=TRUE)
install.packages('dataPreparation', dependencies=TRUE)

require(caTools)
require(randomForest)
library(rpart)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(caret)
library(Matrix)
library(xgboost)
library(e1071) 
library(Metrics)
library(dataPreparation)
library(randomForest)
require(caTools)

set.seed(999) 

directory <- "C:/Users/user/Desktop/Ovgu Educational/Data Science with R/Repo"
fileName <- "Asteroid_Diameter_Preprocessed.csv"
filePath <- file.path(directory, fileName)

data <- read.csv(filePath,header <- TRUE)

head(data)

preprocessdata <- function(data){
  
  df <- subset(data, select = -c(name,extent,X,spec_B, spec_T, neo, class,n_obs_used,pha))
  neo <- model.matrix(~neo-1,data)
  neo <- subset(neo,select = c(neoY))
  pha <- model.matrix(~pha-1,data)
  pha <- subset(pha,select = c(phaY))
  df <- cbind(df,neo,pha)
  names(df)[names(df) == "phaY"] <- "pha"
  names(df)[names(df) == "neoY"] <- "neo"
  bijections_cols <- whichAreBijection(df)
  df <- subset(df, select = -c(bijections_cols))
  return(df)  
  
}
df <- preprocessdata(data)

# Get indices based on train size
trainSize = 0.75
indices <- sort(sample(nrow(df), nrow(df)*trainSize))

# Use indices to create train and test data
train <- df[indices, ]
test <- df[-indices, ]

## Let's check the count of unique value in the target variable
as.data.frame(table(train$pha))

# Build X_train, y_train, X_test, y_test
X_train <- subset(train, select = -c(diameter))
y_train <- subset(train, select = c(diameter))

X_test <- subset(test, select = -c(diameter))
y_test <- subset(test, select = c(diameter))


#### Linear Regression ####

runLinearModel <- function(train, test){
  
  # Linear Regression
  # . represents all columns
  linearModel = lm(diameter ~ ., data = train)
  summary(linearModel)
  
  preds_lm = predict(linearModel, X_test)
  
  mse_lm = with(test, mean( (diameter - preds_lm)^2))
  rmse_lm = sqrt(mse_lm)
  mae_lm = with(test, mae(diameter, preds_lm))
  
  errors <- c(mse_lm, rmse_lm, mae_lm)
  
  return(errors)
}

runLinearModel_CV <- function(train, test){
  
  # Using Cross Validation for Linear Model
  # Got same rmse as before
  
  train.control <- trainControl(method="cv", number=10)
  linearModel_cv <- train(diameter ~., data = train, method = "lm",
                          trControl = train.control, tuneLength = 10)
  print(linearModel_cv)
  preds_lm_cv = predict(linearModel_cv, X_test)
  mse_lm_cv = with(test, mean( (diameter - preds_lm_cv)^2))
  rmse_lm_cv = sqrt(mse_lm_cv)
  
  mae_lm_cv = with(test, mae(diameter, preds_lm_cv))
  
  errors <- c(mse_lm, rmse_lm_cv, mae_lm_cv)
  
  return(errors)
}

lm_errors <- runLinearModel(train, test)
lm_cv_errors <- runLinearModel_CV(train, test)

#### end

#### Regression Tree ####

runRTree <- function(train, test){
  
  rTree <- rpart(diameter ~ ., data=train, method="anova")
  
  printcp(rTree)
  plotcp(rTree) 
  summary(rTree) 
  
  preds_rt <- predict(rTree, X_test)
  mse_rt = with(test, mean( (diameter - preds_rt)^2))
  rmse_rt = sqrt(mse_rt)
  
  # plot tree
  # plot(dTree, uniform=TRUE, main="Regression Tree for diameter")
  # text(dTree, use.n=TRUE, all=TRUE, cex=.8)
  
  mae_rt = with(test, mae(diameter, preds_rt))
  
  errors <- c(mse_rt, rmse_rt, mae_rt)
  
  return(errors)
}

runRTree_cv <- function(train, test){
  # Use Cross Validation for R Tree
  train.control <- trainControl(method="cv", number=10)
  rTree_cv <- train(diameter ~., data = train, method = "rpart",
                    trControl = train.control, tuneLength = 10)
  print(rTree_cv)
  preds_rt_cv = predict(rTree_cv, X_test)
  mse_rt_cv = with(test, mean( (diameter - preds_rt_cv)^2))
  rmse_rt_cv = sqrt(mse_rt_cv)
  
  mae_rt_cv = with(test, mae(diameter, preds_rt_cv))
  
  errors <- c(mse_rt_cv, rmse_rt_cv, mae_rt_cv)
  
  return(errors)
}

rt_errors <-runRTree(train, test)
rt_cv_errors <-runRTree_cv(train, test)

#### end


#### SVM ####

runSVM <- function(train, test){
  
  svm = svm(formula = diameter ~ ., data = train)
  
  # SVM takes time to run. Use the saved model
  modelName = "svm"
  modelPath <- file.path(directory, modelName)
  saveRDS(svm, file = modelPath)
  #svm <- readRDS("filename.rds")
  
  preds_svm = predict(svm,X_test)
  
  mse_svm = with(test, mean( (diameter - preds_svm)^2))
  rmse_svm = sqrt(mse_svm)
  
  mae_svm = with(test, mae(diameter, preds_svm))
  
  errors <- c(mse_svm, rmse_svm, mae_svm)
  
  return (errors)
}

svm_errors <-runSVM(train, test)

#### end


#### XGBoost ####

# Convert data to Matrix
df_XGBoost <- subset(df, select = -c(diameter))

datamatrix <- data.matrix(df_XGBoost)
labels <- data.matrix(df$diameter)

# training data
X_train_xgb <- datamatrix[indices,]
y_train_xgb <- labels[indices]

# testing data
X_test_xgb <- datamatrix[-indices,]
y_test_xgb <- labels[-indices, ]


runXGBoost <- function(X_train, y_train, X_test, y_test){
  
  #xgboost for diameter
  dtrain <- xgb.DMatrix(data = X_train, label= y_train)
  dtest <- xgb.DMatrix(data = X_test, label= y_test)
  
  model <- xgboost(data = dtrain, # the data   
                   nround = 10) 
  
  # generate predictions for our held-out testing data
  preds_xgb <- predict(model, dtest)
  mse_xgb = mean((y_test - pred_xgb)^2)
  rmse_xgb = sqrt(mse_lm)
  
  mae_xgb = with(test, mae(diameter, preds_rt))
  
  errors <- c(mse_xgb, rmse_xgb, mae_xgb)
  
  return(errors)  
}

xgb_errors <-runXGBoost(X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb)

#### end

#### Random Forest ####


runRF <- function(train, test){
  
  rf <- randomForest(formula = diameter ~ .,data=train)
  
  preds_rf = predict(rf,X_test) 
  
  mse_rf = with(test, mean( (diameter - preds_rf)^2))
  rmse_rf = sqrt(mse_rf)
  mae_rf = with(test, mae(diameter, preds_rf))
  
  errors <- c(mse_rf, rmse_rf, mae_rf)
  
  return (errors)
}


rf_errors <-runRF(train, test)

#### end

