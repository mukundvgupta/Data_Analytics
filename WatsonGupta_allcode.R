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

# check for missing values
colSums(is.na(energy_data)) # no null value is reported

#using the str() function to get an overview of the structure of the dataset:
str(energy_data)

#Using summary function to get a summary statistics of all variables 
summary(energy_data)

#Plotting histogram for all numeric data
energy_data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram() +
  facet_wrap(~variable, scales = "free")

## looking at the histogram we can make the following observations:
table(energy_data$DT00)
table(energy_data$MXSD)
table(energy_data$TSNW)
table(energy_data$DX32)
### DT00 (Number days in a month with minimum temperature ≤ 0 °F has most of it's value as 0, with only 3 unique values. This questions the use of  the variable in our analysis. Same is the case with MXSD and TSNW)

# Dropping columns
energy_data <- energy_data[, -which(names(energy_data) %in% c("DT00", "MXSD", "TSNW", "DX32"))]

#Plotting Box Plot for all numeric data
energy_data %>%
  select_if(is.numeric) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value)) +
  geom_boxplot() +
  facet_wrap(~variable, scales = "free")
## DT32 seems to have a lot of values above the 3rd Quartile. We might have to look at ways to deal with it

# calculate the correlation matrix
cor(energy_data)

# Load the ggcorrplot package

# Create a correlation plot as a heatmap
ggcorrplot(cor(energy_data), 
           hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           colors = c("#6D9EC1", "#FFFFFF", "#E46726"), 
           title = "Correlation Matrix Heatmap",
           ggtheme = ggplot2::theme_gray)


## We can clearly see that there is high correlation between some variables. This will have to be treated before model building

#Checking class of every variable
sapply(energy_data, class)

dim(energy_data)


# Splitting the data into training and test in the ratio of 70/30
train <-  sample(1:nrow(energy_data), 0.7* nrow(energy_data))
test <- (-train)

# Decision tree

library(tree)
tree.energy <- tree(res.sales.adj~month+ res.price+ EMXT+ EMNT+ 
                      MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, energy_data, subset=train)
summary(tree.energy)

plot(tree.energy)
text(tree.energy,pretty =1)

cv.energy <- cv.tree(tree.energy)
plot(cv.energy$size, cv.energy$dev, type="b")

# predict on test set
yhat <- predict(tree.energy, newdata = energy_data[-train,])

# predict on training set
yhat.train <- predict(tree.energy, newdata = energy_data[train,])

energy.train <- energy_data[train, "res.sales.adj"]
energy.test <- energy_data[test, "res.sales.adj"]

plot(yhat,energy.test)
abline(0,1)
# in sample RMSE
out_sample_mse_dt <- mean((yhat-energy.test)^2)
out_sample_rmse_dt <- sqrt(out_sample_mse_dt)
out_sample_rmse_dt

# out of sample RMSE
in_sample_mse_dt <- mean((yhat.train-energy.train)^2)
in_sample_rmse_dt <- sqrt(in_sample_mse_dt)
in_sample_rmse_dt

# Bagging
library(randomForest)
set.seed(1)
bag.energy <- randomForest(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                             MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, data=energy_data, subset=train, mtry=15, importance=TRUE)
bag.energy

yhat.bag <- predict(bag.energy, newdata = energy_data[-train,])
plot(yhat.bag, energy.test)
abline(0,1)
mean((yhat - energy.test)^2)

bag.energy <- randomForest(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                             MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, data=energy_data, subset=train, mtry=15, ntree=1000, importance=TRUE)

# test
yhat.bag <- predict(bag.energy, newdata = energy_data[-train,])
plot(yhat.bag, energy.test)
out_sample_mse_bag <- mean((yhat.bag - energy.test)^2)
out_sample_rmse_bag <- sqrt(out_sample_mse_bag)
out_sample_rmse_bag

#train
yhat.bag.train <- predict(bag.energy, newdata = energy_data[train,])
plot(yhat.bag, energy.test)
in_sample_mse_bag <- mean((yhat.bag.train - energy.train)^2)
in_sample_rmse_bag <- sqrt(in_sample_mse_bag)
in_sample_rmse_bag

# Boosting
library(gbm)
set.seed(1)

boost.energy <- gbm(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                      MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, data = energy_data[train,], distribution = "gaussian", n.trees=1000,cv.folds = 10,
                    interaction.depth = 4)
summary(boost.energy)
#plot(boost.energy, i= "MDPT")
#plot(boost.energy, i="CLDD")

#yhat.boost <- predict(boost.energy, newdata = energy_data[-train,], n.trees=1000, cv.folds=10)
#mean((yhat.boost - energy.test)^2)

boost.energy <- gbm(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                      MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, data = energy_data[train,], distribution = "gaussian", n.trees=1000,
                    interaction.depth = 4, shrinkage = 0.025, verbose = F, cv.folds = 10)
#train
yhat.boost.train <- predict(boost.energy, newdata = energy_data[train,], n.trees=1000)
in_sample_mse_boost <- mean((yhat.boost.train - energy.train)^2)
in_sample_rmse_boost <- sqrt(in_sample_mse_boost)
in_sample_rmse_boost

#test
yhat.boost <- predict(boost.energy, newdata = energy_data[-train,], n.trees=1000)
out_sample_mse_boost<- mean((yhat.boost - energy.test)^2)
out_sample_rmse_boost <- sqrt(out_sample_mse_boost)
out_sample_rmse_boost

# Random Forest
## Random Forest
set.seed(1)
rf.energy <- randomForest(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                            MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME, data=energy_data, subset=train, mtry=5, importance=TRUE, ntree=1000)
rf.energy

# train
yhat.rf.train <- predict(rf.energy, newdata = energy_data[train,])
plot(yhat.rf.train, energy.train)
abline(0,1)
importance(rf.energy)
varImpPlot(rf.energy)
in_sample_mse_rf <- mean((yhat.rf.train - energy.train)^2)
in_sample_rmse_rf <- sqrt(in_sample_mse_rf)
in_sample_rmse_rf 

# test
yhat.rf <- predict(rf.energy, newdata = energy_data[-train,])
plot(yhat.rf, energy.test)
abline(0,1)
importance(rf.energy)
varImpPlot(rf.energy)
out_sample_mse_rf <- mean((yhat.rf - energy.test)^2)
out_sample_rmse_rf <- sqrt(out_sample_mse_rf)
out_sample_rmse_rf

# Linear Model

## Selecting all the variables in energy_data

# In sample RMSE
lm.fit <- lm(res.sales.adj ~ ., data = energy_data, family="binomial", subset = train)
summary(lm.fit)
test_prob <- predict(lm.fit, newdata = energy_data[train,], type = "response")

# Calculate the RMSE value
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[train])
cat("The in-sample RMSE value is:", rmse)

# Out of sample RMSE
lm.fit.test <- lm(res.sales.adj ~ ., data = energy_data, family="binomial", subset = train)
summary(lm.fit)
test_prob <- predict(lm.fit.test, newdata = energy_data[-train,], type = "response")

# Calculate the RMSE value
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[-train])
cat("The out-of-sample RMSE value is:", rmse)


## Selecting variables from lasso to build a linear model

# train
lm.model <- lm(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                 MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME,  
               data = energy_data, subset = train)
summary(lm.model)
test_prob <- predict(lm.model, newdata = energy_data[train,], type = "response")
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[train])
cat("The in-sample RMSE value is:", rmse)

# Test
lm.model.test <- lm(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                      MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME,  
                    data = energy_data, subset = train)

test_prob <- predict(lm.model.test, newdata = energy_data[test,], type = "response")
# Calculate the RMSE value
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[test])
cat("The out of sample RMSE value is:", rmse)

### Checking for interaction terms
lm.model1 <- lm(res.sales.adj ~ month+ res.price+ EMXT+ EMNT+ 
                  MMXT+ DT90+DT32+DP05+WDSP+ GUST+HTDD+CLDD+UNEMP+UNEMPRATE+PCINCOME
                +EMXT*EMNT+EMXT*MMXT+EMXT*DP05+EMXT*UNEMP+EMXT*UNEMPRATE+EMNT*DP05+EMNT*UNEMP+EMNT*UNEMPRATE+
                  MMXT*DP05+MMXT*UNEMP+MMXT*UNEMPRATE+DP05*UNEMP+DP05*UNEMPRATE+UNEMP*UNEMPRATE,  
                data = energy_data, family="binomial", subset = train)
summary(lm.model1)

# train
test_prob <- predict(lm.model1, newdata = energy_data[train,], type = "response")
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[train])
cat("The in-sample RMSE value is:", rmse)

# Test

test_prob <- predict(lm.model1, newdata = energy_data[test,], type = "response")
# Calculate the RMSE value
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[test])
cat("The out of sample RMSE value is:", rmse)

#### Adding interaction terms increases the R-square value of the model

### Trying Polynomial Regression
lm.model_poly <- lm(res.sales.adj ~ month+ res.price+ poly(EMXT,2)+ EMNT+ 
                      MMXT+ DT90+DT32+poly(DP05,2)+WDSP+ GUST+HTDD+CLDD+poly(UNEMP,3)+UNEMPRATE+PCINCOME,  
                    data = energy_data, family="binomial", subset = train)
summary(lm.model_poly)

# train
test_prob <- predict(lm.model_poly, newdata = energy_data[train,], type = "response")
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[train])
cat("The in-sample RMSE value is:", rmse)

# Test

test_prob <- predict(lm.model_poly, newdata = energy_data[test,], type = "response")
# Calculate the RMSE value
rmse <- caret::RMSE(test_prob, energy_data$res.sales.adj[test])
cat("The out of sample RMSE value is:", rmse)


### Using splines and GAM

library(gam)
gam.m3 <- gam(res.sales.adj ~ month+ res.price+ s(EMXT,2)+ EMNT+ 
                MMXT+ DT90+DT32+s(DP05,2)+WDSP+ GUST+HTDD+CLDD+s(UNEMP,3)+UNEMPRATE+PCINCOME,  
              data = energy_data, subset = train)
summary(gam.m3)
# train
preds <- predict(gam.m3, newdata = energy_data[train,])
summary(preds)
preds

rmse <- caret::RMSE(preds, energy_data$res.sales.adj[train])
cat("The in-sample RMSE value is:", rmse)

#test
preds <- predict(gam.m3, newdata = energy_data[test,])
summary(preds)
preds

rmse <- caret::RMSE(preds, energy_data$res.sales.adj[test])
cat("The out-sample RMSE value is:", rmse)

# Bayesian Additive Regression Trees
library(BART)
#x <- energy_data[,1:29]
x <- energy_data[, -which(names(energy_data) %in% c("res.sales.adj", "year", "EXMP", "EMXP",
                                                    "TPCP","MMNT","MNTM", "DP01","DP10",
                                                    "MDPT", "VISIB", "LABOR", "GSP", "EMP",
                                                    "MWSPD"))]
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

# In-Sample Create plot
plot(ytrain, yhat.bart.train, col="blue", pch=20, xlab="res.sales.adj", ylab="Predicted sales")

# Add diagonal line
abline(a=0, b=1, col="red")

# Add title and legend
title("In-sample predicted vs actual")
legend("topleft", legend="BART", col="blue", pch=20)


# Out-of-Sample Create plot
plot(ytest, yhat.bart, col="blue", pch=20, xlab="Actual sales", ylab="Predicted sales")

# Add diagonal line
abline(a=0, b=1, col="red")

# Add title and legend
title("Out-of-sample predicted vs actual")
legend("topleft", legend="BART", col="blue", pch=20)
# Compute variable importance
varimp <- bartVarImp(bartfit.train)


# Ridge and Lasso Regression
## Ridge

set.seed(1)
x <- model.matrix(res.sales.adj ~ ., energy_data) [,-1]
y <- energy_data$res.sales.adj

set.seed(1)
grid <- 10^seq(20,-2, length=300)
train<- sample(1:nrow(x), 0.7*nrow(x))
test <- (-train)
y.train <- y[train]
y.test <- y[test]

library(glmnet)
ridge.mod <- glmnet(x[train,], y[train], alpha=0, lambda = grid)
plot(ridge.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha=0)
plot(cv.out)
bestlam <- cv.out$lambda.min
ridge.pred <- predict(ridge.mod, s=bestlam, newx = x[test,])
ridge.pred.train <- predict(ridge.mod, s=bestlam, newx = x[train,])

mean((ridge.pred - y.test)^2)
mean((ridge.pred.train - y.train)^2)

out <- glmnet(x,y, alpha=0, lambda = grid)
ridge.coef <- predict(out, type = "coefficients", s= bestlam) [1:29,]
ridge.coef
ridge.coef[ridge.coef != 0]

### based on ridge, all variables are important and it doesn't lead to any dimensional reduction.

## Lasso Regression

set.seed(1)
x <- model.matrix(res.sales.adj ~ ., energy_data) [,-1]
y <- energy_data$res.sales.adj

set.seed(1)
grid <- 10^seq(20,-2, length=300)
train<- sample(1:nrow(x), 0.7* nrow(x))
test <- (-train)
y.test <- y[test]

library(glmnet)
lasso.mod <- glmnet(x[train,], y[train], alpha=1, lambda = grid)
plot(lasso.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s=bestlam, newx = x[test,])
lasso.pred.train <- predict(lasso.mod, s=bestlam, newx = x[train,])

mean((lasso.pred.train-y.train)^2)
mean((lasso.pred - y.test)^2)

out <- glmnet(x,y, alpha=1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s= bestlam) [1:29,]
lasso.coef
lasso.coef[lasso.coef != 0]

# Calculating r sqaured value of lasso
library(rsq)
rsq <- 1 - mean((lasso.pred - y.test)^2) / var(y.test)
rsq


#Table for Models used and Error                 
Model <- c('Lasso', 'Ridge', 'Linear Model_all variables', 'LinearModel_Lasso Reduced Var', 'Linear Model_insignif variable interaction', 'Polynomial Regression', 'Splines and GAM', 'Decision Tree', 'Bagging', 'Random Forest', 'Boosting', 'BART', 'Logistical Regression', 'PCR')
Tuning_Parameters <- c('alpha=1, lambda= 10^seq(20,-2, length=300)', 'alpha=0, lambda= 10^seq(20,-2, length=300)', 'Data split 70/30 training-test', 'Data split 70/30 training-test', 'Data split 70/30 training-test', 'Data split 70/30 training-test. poly(EMXT,2), poly(DP05,2), poly(UNEMP,3)', 'Data split 70/30 training-test. s(EMXT,2), s(DP05,2), s(UNEMP,3)', 'Data split 70/30 training-test', 'Data split 70/30 training-test. MTRY=28, ntree=1000, importance=true', 'Data split 70/30 training-test. mtry=14, imprtance=true, ntree=1000', 'Interaction depth=4, shrinkage=0.025, verbose=F, CV folds=10, ntrees=1000', 'Data split 70/30 training-test; variable selection according to Lasso', 'not applicable', 'not applicable')                
In_Sample_RMSE <- c('465.80', '447.70', '405.42', '438.94', '407.15', '419.81', '421.99', '483.09', '205.28', '210.30', '133.96', '205.896', 'Dependent variable not between 0-1', 'Tried but output doesnt look correct')
Out_of_Sample_RMSE <- c('461.84', '467.12', '537.61', '504.61', '515.39', '505.72', '507.16', '541.49', '456.74', '437.67', '465.19', '423.10', 'Dependent variable not between 0-1', 'Tried but output doesnt look correct')


#table output
Table_out <- data.frame(Model,Tuning_Parameters,In_Sample_RMSE,Out_of_Sample_RMSE)
Table_out

#Logistical model is better for categorical variables, which we don't really have so an error kept appearing when trying to run this model
glm.fit <- glm(res.sales.adj ~ ., data = energy_data, family="binomial", subset = train)
summary(glm.fit)
