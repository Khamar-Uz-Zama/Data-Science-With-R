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
library(caTools)
library(tidyr)
library(tidyverse)
library(dplyr)
library(GGally)
library(corrplot)
library(mlbench)
library(caret)

####Data Preprocessing####
preprocess <- function(directory,fileName){
  filePath <- file.path(directory, fileName)
  df <- read.csv(filePath,header <- TRUE)
  sapply(df, function(x) sum(is.na(x)))
  summary(df)
  apply(df, 2, class)
  class(df$diameter)
  x <- colnames(df)
  for (i in length(x)){
    class(df$x[i])}
  
  #convert diameter to numeric
  df$diameter <- as.numeric(as.character(df$diameter))
  
  #drop na's in diameter column
  df_diameter_nadrop <- subset(df, !is.na(df$diameter))
  
  ## Remove columns with more than 50% NA
  df_diameter_nadrop <- df_diameter_nadrop[, which(colMeans(!is.na(df_diameter_nadrop)) > 0.5)]
  
  sapply(df_diameter_nadrop, function(x) sum(is.na(x))/length(x[1]))
  head(df_diameter_nadrop)
  unique(df_diameter_nadrop$data_arc)
  
  which(df_diameter_nadrop$condition_code == '9')
  #change type of condition_code and data_arc to numeric
  df_diameter_nadrop$condition_code <- as.numeric(as.character(df_diameter_nadrop$condition_code))
  df_diameter_nadrop$data_arc <- as.numeric(as.character(df_diameter_nadrop$data_arc))
  #fill data_arc, H, albedo with mean of their respective columns
  cols <- c("data_arc","albedo","H") 
  df_diameter_nadrop[cols] <-  replace_na(df_diameter_nadrop[cols],as.list(colMeans(df_diameter_nadrop[cols],na.rm=T)))
  fileName <- "Asteroid_Diameter_Preprocessed.csv"
  filePath <- file.path(directory, fileName)
  write.csv(df_diameter_nadrop, filePath)
}  

#### Exploratory Data Analysis####
EDA<- function(ast.data){
  
  ## Convert factor variables into numeric 
  ast.data$pha <- ifelse(ast.data$pha =="Y",1,0)
  
  ast.data$neo <- ifelse(ast.data$neo =="Y",1,0)
  
  ## remove first column which is row number
  ast.data <- ast.data[-1]
  nums <- unlist(lapply(ast.data, is.numeric)) 
  ast.data <- ast.data[ , nums]
  
  ### Exploratory Data Analysis
  ## Get all histograms of all numeric variables
  darkcols <- brewer.pal(8, "Dark2")
  
  hist_all <- ast.data %>%
    keep(is.numeric) %>% 
    gather() %>% 
    ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()
  
  ## Data distribution using Histograms
  hist_all
  
  ## Plot density graphs
  density_all <- ast.data %>%
    keep(is.numeric) %>%                     # Keep only numeric columns
    gather() %>%                             # Convert to key-value pairs
    ggplot(aes(value)) +                     # Plot the values
    facet_wrap(~ key, scales = "free") +     # In separate panels
    geom_density()  
  
  density_all
  
  ## Get box plot distribution of variables 
  boxplot(ast.data, outline = FALSE, col = darkcols,
          xlab = "Variables", ylab = "Frequency")         # Hide outliers
  
  ## Get Correlation matrix
  corr_matrix <- cor(ast.data)
  
  # Correlation matrix of all variables in dataset
  # with circles
  corrplot(corr_matrix)
  
  # with numbers and lower
  cor_plot <- corrplot(corr_matrix,
                       method = 'number',
                       type = "lower")
  
  
  # calculate correlation matrix
  correlationMatrix <- cor(ast.data)
  # summarize the correlation matrix
  print(correlationMatrix)
  # find attributes that are highly corrected (ideally >0.75)
  highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.90)
  # print indexes of highly correlated attributes
  print(highlyCorrelated)
}


####Preprocess for model####
preprocessformodel <- function(data){
  
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

####Regression models for Diameter prediction####
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
lm_errors <- runLinearModel(train, test)

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
  
  errors <- c(mse_lm_cv, rmse_lm_cv, mae_lm_cv)
  
  return(errors)
}


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


#### end


#### SVM ####

runSVM <- function(train, test){
  
  svm = svm(formula = diameter ~ ., data = train)
  
  # SVM takes time to run. Use the saved model
  #modelName = "svm"
  #modelPath <- file.path(directory, modelName)
  #saveRDS(svm, file = modelPath)
  #svm <- readRDS("filename.rds")
  
  preds_svm = predict(svm,X_test)
  
  mse_svm = with(test, mean( (diameter - preds_svm)^2))
  rmse_svm = sqrt(mse_svm)
  
  mae_svm = with(test, mae(diameter, preds_svm))
  
  errors <- c(mse_svm, rmse_svm, mae_svm)
  
  return (errors)
}


#### end


#### XGBoost ####

runXGBoostdia <- function(df,indices){
  # Convert data to Matrix
  df_XGBoost <- subset(df, select = -c(diameter))
  
  datamatrix <- data.matrix(df_XGBoost)
  labels <- data.matrix(df$diameter)
  
  # training data
  X_train <- datamatrix[indices,]
  y_train <- labels[indices]
  
  # testing data
  X_test <- datamatrix[-indices,]
  y_test <- labels[-indices, ]
  #xgboost for diameter
  dtrain <- xgb.DMatrix(data = X_train, label= y_train)
  dtest <- xgb.DMatrix(data = X_test, label= y_test)
  
  model <- xgboost(data = dtrain, # the data   
                   nround = 10) 
  
  # generate predictions for our held-out testing data
  preds_xgb <- predict(model, dtest)
  mse_xgb = mean((y_test - preds_xgb)^2)
  rmse_xgb = sqrt(mse_xgb)
  
  mae_xgb = with(test, mae(diameter, preds_xgb))
  
  errors <- c(mse_xgb, rmse_xgb, mae_xgb)
  
  return(errors)  
}


#### end

#### Random Forest ####


runRFdia <- function(train, test){
  
  rf <- randomForest(diameter~.,data=train)
  
  preds_rf = predict(rf,X_test) 
  
  mse_rf = with(test, mean( (diameter - preds_rf)^2))
  rmse_rf = sqrt(mse_rf)
  mae_rf = with(test, mae(diameter, preds_rf))
  
  errors <- c(mse_rf, rmse_rf, mae_rf)
  
  return (errors)
}



#### end

####Classification models for pha prediction####
####Logistic Regression Without CV####

runLogisticModelpha <- function(train, test,X_train, X_test,y_train, y_test){
  LogisticModel = glm(pha ~ . , family="binomial", data = train, control = list(maxit = 500))
  summary(LogisticModel)
  pred = predict(LogisticModel, newdata = X_test, type = "response")
  pred[(pred>0.5)] = 1
  pred[(pred<=0.5)] = 0
  plot(LogisticModel)
  return(pred)
}
####Random Forest Without CV####
runrandomforestModelpha <- function(train, test,X_train, X_test,y_train, y_test){
  
  rf = randomForest(pha~.,  
                    ntree = 100,
                    data = train)
  predicted.response <- predict(rf, X_test)
  plot(rf) 
  varImp(rf)
  varImpPlot(rf,  
             sort = T,
             n.var=ncol(df)-1,
             main="Variable Importance")
  print(rf)
  plot(rf) 
  return(predicted.response)
  
}


####XGBoost Without CV####

runXGBoostpha <- function(df,indices){
  # Convert data to Matrix
  labelpha <- data.matrix(df$pha)
  df_XGB <- subset(df, select = -c(pha))
  datamatrix <- data.matrix(df_XGB)
  # training data
  train_data <- datamatrix[indices,]
  train_labels <- labelpha[indices]
  # testing data
  test_data <- datamatrix[-indices,]
  test_labels <- labelpha[-indices]
  #xgboost for pha
  xgb.train <- xgb.DMatrix(data = train_data, label= train_labels)
  xgb.test <- xgb.DMatrix(data = test_data, label= test_labels)
  # train a model using our training data fro pha classifying
  model <- xgboost(data = xgb.train, # the data   
                   nround =10)
  pred <- predict(model, test_data)
  print(model)
  pred[(pred>0.5)] = 1
  pred[(pred<=0.5)] = 0
  cm_rf <- confusionMatrix(factor(test_labels),factor(pred))
  return(cm_rf)
}
####Default Random Forest With CV####
defaultRFModelpha <- function(X_train, X_test,y_train, y_test){
  control <- trainControl(method="repeatedcv", number=10, repeats=3)
  mtry <- sqrt(ncol(X_train))
  metric <- "Kappa"
  tunegrid <- expand.grid(.mtry=mtry)
  rf_default <- train(pha~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
  print(rf_default)
  trellis.par.set(caretTheme())
  basePred <- predict(rf_default,X_test)
  cm<-confusionMatrix(factor(y_test$pha),factor(basePred))
  return(cm)
}

####RandomSearch Random Forest With CV####

RandomSearchRFModelpha <- function(X_train, X_test,y_train, y_test){
  control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
  mtry <- sqrt(ncol(X_train))
  metric <- "Kappa"
  rf_random <- train(pha~., data=train, method="rf", metric=metric, tuneLength=15, trControl=control)
  print(rf_random)
  plot(rf_random)
  basePred <- predict(rf_random,X_test)
  cm<-confusionMatrix(factor(y_test$pha),factor(basePred))
  return(cm)
} 

####GridSearch Random Forest With CV####

GridSearchRFModelpha <- function(X_train, X_test,y_train, y_test){
  control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
  tunegrid <- expand.grid(.mtry=c(1:15))
  mtry <- sqrt(ncol(X_train))
  metric <- "Kappa"
  rf_gridsearch <- train(pha~., data=train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
  print(rf_gridsearch)
  plot(rf_gridsearch)
  basePred <- predict(rf_gridsearch,X_test)
  cm<-confusionMatrix(factor(y_test$pha),factor(basePred))
  return(cm)
} 


####Default XGBoost With CV####

defaultXGBModelpha <- function(X_train, X_test,y_train, y_test){
  
  grid_default <- expand.grid(
    nrounds = 100,
    max_depth = 6,
    eta = 0.3,
    gamma = 0,
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
  
  train_control <- caret::trainControl(
    method = "none",
    verboseIter = FALSE, # no training log
    allowParallel = TRUE 
  )
  
  xgb_base <- caret::train(
    x = X_train,
    y = y_train$pha,
    trControl = train_control,
    tuneGrid = grid_default,
    method = "xgbTree",
    verbose = TRUE,
    nthreads = 4
  )
  basePred <- predict(xgb_base,X_test)
  cm<-confusionMatrix(factor(y_test$pha),factor(basePred))
  return(cm)
}

####TuneXGBoost With CV####
tuneXGBModelpha <- function(X_train, X_test,y_train, y_test){
  
  tune_grid <- expand.grid(
    nrounds = 100,
    eta = c(0.01, 0.1, 0.3),
    max_depth = c(2, 3, 5, 10),
    gamma = c(0),
    colsample_bytree = 1,
    min_child_weight = 1,
    subsample = 1
  )
  
  tune_control <- caret::trainControl(
    method = "cv", # cross-validation
    number = 3, # with n folds 
    verboseIter = FALSE, # no training log
    allowParallel = TRUE #
  )
  
  xgb_tune <- caret::train(
    x = X_train,
    y = y_train$pha,
    trControl = tune_control,
    tuneGrid = tune_grid,
    method = "xgbTree",
    verbose = TRUE,
    nthreads = 4
  )
  plot(xgb_tune)
  tunePred <- predict(xgb_tune,X_test)
  cm<-confusionMatrix(factor(y_test$pha),factor(tunePred))
  return(cm)
}
####Export Classification Metrics####

ExportMetrics <- function(cm_lm,cm_rf,cm_xgb,cm_defaultRFModelpha,cm_RandomSearchRFModelpha,cm_GridSearchRFModelpha,cm_defaultXGBModelpha,cm_tuneXGBModelpha){
  cm_lm <- data.frame(cbind('Logistic Regression Without CV',t(cm_lm$overall),t(cm_lm$byClass)))
  cm_rf <- data.frame(cbind('Random Forest Without CV',t(cm_rf$overall),t(cm_rf$byClass)))
  cm_xgb <- data.frame(cbind('XGBoost Without CV',t(cm_xgb$overall),t(cm_xgb$byClass)))
  cm_defaultXGBModelpha <- data.frame(cbind('Default XGBoost With CV',t(cm_defaultXGBModelpha$overall),t(cm_defaultXGBModelpha$byClass)))
  cm_tuneXGBModelpha <- data.frame(cbind('TuneXGBoost With CV',t(cm_tuneXGBModelpha$overall),t(cm_tuneXGBModelpha$byClass)))
  cm_defaultRFModelpha <- data.frame(cbind('Default Random Forest With CV',t(cm_defaultRFModelpha$overall),t(cm_defaultRFModelpha$byClass)))
  cm_RandomSearchRFModelpha <- data.frame(cbind('RandomSearch Random Forest With CV',t(cm_RandomSearchRFModelpha$overall),t(cm_RandomSearchRFModelpha$byClass)))
  cm_GridSearchRFModelpha <- data.frame(cbind('GridSearch Random Forest With CV',t(cm_GridSearchRFModelpha$overall),t(cm_GridSearchRFModelpha$byClass)))
  tocsv <- data.frame(rbind(cm_lm,cm_rf,cm_xgb,cm_defaultRFModelpha,cm_RandomSearchRFModelpha,cm_GridSearchRFModelpha,cm_defaultXGBModelpha,cm_tuneXGBModelpha))
  write.csv(tocsv,file="Metricspha.csv")
}
plotClassificationmetric <- function(directory,fileName){
  filePath <- file.path(directory, fileName)
  metric <- read.csv(filePath,header <- TRUE)
  label=metric$V1
  label=str_replace(label,"Without CV","\nWithout CV")
  label=str_replace(label,"With CV","\nWith CV")
  op <- par(mar = c(8,0,-5,0) + 5)
  plot(metric$Kappa,xaxt = 'n')
  axis(side=1,at=c(1,2,3,4,5,6,7,8),labels=c(label),las=2)
  op <- par(mar = c(8,0,-5,0) + 5)
  plot(metric$Balanced.Accuracy,xaxt = 'n')
  axis(side=1,at=c(1,2,3,4,5,6,7,8),labels=c(label),las=2)
  op <- par(mar = c(8,0,-5,0) + 5)
  plot(metric$Neg.Pred.Value,xaxt = 'n')
  axis(side=1,at=c(1,2,3,4,5,6,7,8),labels=c(label),las=2)
}


#### MAIN ####
set.seed(999) 
#directory <- getwd()
directory <- "C:/Users/Joe/Documents/DKE/Sem 2/DSR"
fileName <- "Asteroid_Updated.csv"
filePath <- file.path(directory, fileName)

#Preprocessing the Data
# preprocess(directory,fileName)

fileName <- "Asteroid_Diameter_Preprocessed.csv"
filePath <- file.path(directory, fileName)
data <- read.csv(filePath,header <- TRUE)

#Exploratory Data Analysis
# EDA(data)

df <- preprocessformodel(data)
# Get indices based on train size
trainSize = 0.6
indices <- sort(sample(nrow(df), nrow(df)*trainSize))
# Use indices to create train and test data
train <- df[indices, ]
test <- df[-indices, ]
# Build X_train, y_train, X_test, y_test for diameter prediction
X_train <- subset(train, select = -c(diameter))
y_train <- subset(train, select = c(diameter))
X_test <- subset(test, select = -c(diameter))
y_test <- subset(test, select = c(diameter))

####Run Regression Models####
# lm_errors <- runLinearModel(train, test)
# lm_cv_errors <- runLinearModel_CV(train, test)
# rt_errors <-runRTree(train, test)
# rt_cv_errors <-runRTree_cv(train, test)
# svm_errors <-runSVM(train, test)
# xgb_errors <-runXGBoostdia(df,indices)
# rf_errors <-runRFdia(train, test)

# Build X_train, y_train, X_test, y_test for diameter prediction
# final <- df[!(is.na(df$pha)),]
# final$pha <- factor(final$pha)
# split <- sample.split(final$pha, SplitRatio = trainSize)
# train <- subset(final, split == TRUE)
# test <- subset(final, split == FALSE)
# X_train <- subset(train, select = -c(pha))
# y_train <- subset(train, select = c(pha))
# X_test <- subset(test, select = -c(pha))
# y_test <- subset(test, select = c(pha))

####Run Classification Models####
# pred_lm=runLogisticModelpha(train, test,X_train, X_test,y_train, y_test)
# cm_lm <- confusionMatrix(factor(y_test$pha),factor(pred_lm))
# pred_rf=runrandomforestModelpha(train, test,X_train, X_test,y_train, y_test)
# cm_rf <- confusionMatrix(factor(y_test$pha),factor(pred_rf))
# cm_xgb <- runXGBoostpha(df,indices)

####Run Classification Models with cross validation and Hyper Parameters####
# cm_defaultXGBModelpha=defaultXGBModelpha(X_train, X_test,y_train, y_test)
# cm_tuneXGBModelpha=tuneXGBModelpha(X_train, X_test,y_train, y_test)
# cm_defaultRFModelpha=defaultRFModelpha(X_train, X_test,y_train, y_test)
# cm_RandomSearchRFModelpha=RandomSearchRFModelpha(X_train, X_test,y_train, y_test)
# cm_GridSearchRFModelpha=GridSearchRFModelpha(X_train, X_test,y_train, y_test)

# #Export Classification Metrics
# ExportMetrics(cm_lm,cm_rf,cm_xgb,cm_defaultRFModelpha,cm_RandomSearchRFModelpha,cm_GridSearchRFModelpha,cm_defaultXGBModelpha,cm_tuneXGBModelpha)

# #plot Classification Metrics
# fileName <- "Metricspha.csv"
# plotClassificationmetric(directory,fileName)