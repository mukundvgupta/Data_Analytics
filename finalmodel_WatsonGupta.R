library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(ranger)
library(caret)
library(data.table)
library(caTools)
library(rpart)
library(rpart.plot)
library(gbm, quietly=TRUE)
library(pROC)
library(ggplot2)
library(reshape2)
library(leaps)
library(ggcorrplot)
library(FactoMineR)
library(psych)
library(factoextra)

#loading the dataset
energy_data <- read.csv("energy_training.csv")

# Dropping columns with most of the values equal to zero.
energy_data <- energy_data[, -which(names(energy_data) %in% c("DT00", "MXSD", "TSNW", "DX32"))]

# Lasso regression has been used for dimension reduction as well as for removing the high correlation in the data.

# Splitting the data into training and test in the ratio of 70/30
train <-  sample(1:nrow(energy_data), 0.7* nrow(energy_data))
test <- (-train)

# BART model has been chosen as it performs the best.
# Bayesian Additive Regression Trees
library(BART)

x <- energy_data[, -which(names(energy_data) %in% c("res.sales.adj", "year", "EXMP",                                                    "TPCP","MMNT","MNTM", "DP01","DP10",                                                    "MDPT", "VISIB", "LABOR", "GSP", "EMP",                                                    "MWSPD"))]
y <- energy_data[,"res.sales.adj"]
xtrain <- x[train,]
ytrain <- y[train]
xtest <- x[-train,]
ytest <- y[-train]

set.seed(1)
#train
bartfit.train <- gbart(xtrain,ytrain,x.test = xtrain)
yhat.bart.train <- bartfit.train$yhat.train.mean
in_sample_mse_bart <- mean((ytrain-yhat.bart.train )^2)
in_sample_rmse_bart <- sqrt(in_sample_mse_bart)
in_sample_rmse_bart

# Test
bartfit <- gbart(xtrain,ytrain,x.test = xtest)
yhat.bart <- bartfit$yhat.test.mean
out_sample_mse_bart <- mean((ytest-yhat.bart )^2)
out_sample_rmse_bart <- sqrt(out_sample_mse_bart)
out_sample_rmse_bart

finalmodel <- bartfit
save(finalmodel, file="WatsonGupta.RData")