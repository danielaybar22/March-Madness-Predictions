library(readxl)
library(ggplot2)
library(plyr)
library(reshape2)
library(waffle)
library(googleVis)
library(corrplot)
library(tidyverse)
library(data.table)
library(car)
library(lattice)
library(Hmisc)
library(caret)
library(RWeka)
library(rpart)
library(e1071)
library(partykit)
library(OneR)
library(mlbench)
library(FSelector)
library(tidyverse)
library(rpart)
library(class)
library(dplyr)
library(neuralnet)
library(cluster)
library(dbscan)
library(kohonen)

###### Loading and Cleaning Data #####

setwd("/Users/danielaybar/Downloads/March Madness/Madness 2025")

march16_24 <- read_excel("March Madness 2016-24.xlsx")
march16_24 <- march16_24[,-c(72,73)]
fr25 <- read_excel("2024-2025 madness.xlsx")
fr25 <- fr25[,-72]


ncol(march16_24)

### Correcting Missentered Data ###

for (i in 1:nrow(march16_24)) {
  if(march16_24[i,54] > 200){
    march16_24[i,54] <- 102.6
  } else{
    march16_24[i,54] <- march16_24[i,54] 
  }
}


### Creating New Variables ###

march16_24$fta_marg <- march16_24$FTA - march16_24$OFTA
march16_24$pt_marg <- march16_24$PTS - march16_24$OPTS
march16_24$atr <- march16_24$AST/march16_24$TOV
march16_24$oatr <- march16_24$OAST/march16_24$OTOV
march16_24$ts_marg <- march16_24$`TS%` - march16_24$`opp_TS%`


colnames(fr25) = colnames(march16_24)

# Convert only columns 'a' and 'b' to numeric
fr25[c(2:ncol(fr25))] <- lapply(fr25[c(2:ncol(fr25))], as.numeric)

# View result
print(df)


fr25$fta_marg <- fr25$FTr - fr25$opp_FTr
fr25$pt_marg <- fr25$PTS - fr25$OPTS
fr25$atr <- fr25$AST/fr25$TOV
fr25$oatr <- fr25$OAST/fr25$OTOV
fr25$ts_marg <- fr25$`TS%` - fr25$`opp_TS%`



view(fr25)

### Getting rid of redundant variables ###

essential_subset <- march16_24[,c(12,35,50,52:59,62,66,67,83:87,51)]

### Giving R-friendly column names ###

colnames(essential_subset)[c(1,2,12:14)] <- c('THREEper','O3per','THREEpar','STLper','BLKper')
view(essential_subset)

ncol(essential_subset)
### Creating a secondary subset ###

fr25_sub <- fr25[,c(12,35,50,52:59,62,66,67,83:87)]


corr <- cor(essential_subset)
corrplot(corr, method = 'color')

# Don't need SRS because it is highly correlated with AdjEM
# Same with Seed (AdjEM has stronger correlation with TW)
# Same with AdjO

essential_subset <- essential_subset[,-c(5,10,11)]
fr25_sub <- fr25_sub[,-c(5,10,11)]

view(essential_subset)
view(fr25_sub)
#################################### Kfold cross validation linear regression #########################################

evaluator.lm <- function(subset) {
  # Use k-fold cross validation
  k <- 10
  splits <- runif(nrow(essential_subset))
  results = sapply(1:k, function(i) {
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- essential_subset[test.idx, , drop=FALSE]
    train <- essential_subset[train.idx, , drop=FALSE]
    linear <- lm(as.simple.formula(subset, "TW"), train)
    inv_mse = 1/mean(linear$residuals^2)
    return(inv_mse)
  })
  print(subset)
  print(mean(results))
  return(mean(results))
}

# Forward Selection
subset <- forward.search(names(essential_subset)[-17], evaluator.lm)
f <- as.simple.formula(subset, "TW")  
f

# Backward Selection

subset <- backward.search(names(essential_subset)[-17], evaluator.lm)
b <- as.simple.formula(subset, "TW")  
b

# Hillclimbing Search

subset <- hill.climbing.search(names(essential_subset)[-17], evaluator.lm)
h <- as.simple.formula(subset, "TW") 
h

# Exhaustive Search
subset <- exhaustive.search(names(essential_subset)[-17], evaluator.lm)
e <- as.simple.formula(subset, "TW")
e

################# Standardizing Variables ########################

standardize <- function(x) {
 (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}


march16_24_st <- lapply(essential_subset[,c(1:16)], standardize)
march16_24_st$TW <- essential_subset$TW
fr25_st <- lapply(fr25_sub[,], standardize)

march16_24_st <- as.data.frame(march16_24_st)
fr25_st <- as.data.frame(fr25_st)
colnames(fr25_st)[c(1,2,9:11)] <- c('THREEper','O3per','THREEpar','STLper','BLKper')

view(fr25_st)
view(march16_24_st)

essential_subset %>%
  count(TW) %>%
  mutate(proportion = n/ sum(n))


############################# Train Test Split #################################

### Second Round ###
# Making a binary variable for teams who made it to round 2
for (i in 1:nrow(march16_24_st)) {
  if(march16_24_st[i,17]>= 1){
    march16_24_st[i,18] <- 1
  } else{
    march16_24_st[i,18] <- 0
  }
}
colnames(march16_24_st)[18] <- "rtwo"
march16_24_st$rtwo <- as.factor(march16_24_st$rtwo)

view(march16_24_st)

# Splitting the data 

trainSet32 <- createDataPartition(march16_24_st$rtwo, p = 0.7) [[1]]
train32 <- march16_24_st[trainSet32,]
test32 <-march16_24_st[-trainSet32,]

# Setting Descriptive and Target Variables

train.des32 <- train32[, c(1:16)]
test.des32 <- test32[, c(1:16)]
train.cl32 <- train32$rtwo
test.cl32 <- test32$rtwo



### Sweet16 ###

# Making a binary variable for teams who made it to Sweet 16

for (i in 1:nrow(march16_24_st)) {
  if(march16_24_st[i,17]>= 2){
    march16_24_st[i,19] <- 1
  } else{
    march16_24_st[i,19] <- 0
  }
}
colnames(march16_24_st)[19] <- "sweet16"
march16_24_st$sweet16 <- as.factor(march16_24_st$sweet16)

# Splitting the data 

trainSet16 <- createDataPartition(march16_24_st$sweet16, p = 0.7) [[1]]
train16 <- march16_24_st[trainSet16,]
test16 <- march16_24_st[-trainSet16,]

# Setting Descriptive and Target Variables

train.des16 <- train16[, c(1:16)]
test.des16 <- test16[, c(1:16)]
train.cl16 <- train16$sweet16
test.cl16 <- test16$sweet16


### Elite8 ###

# Making a binary variable for teams who made it to Elite 8

for (i in 1:nrow(march16_24_st)) {
  if(march16_24_st[i,17]>= 3){
    march16_24_st[i,20] <- 1
  } else{
    march16_24_st[i,20] <- 0
  }
}
colnames(march16_24_st)[20] <- "elite8"
march16_24_st$elite8 <- as.factor(march16_24_st$elite8)

#Splitting the data

trainSet8 <- createDataPartition(march16_24_st$elite8, p = 0.8) [[1]]
train8 <- march16_24_st[trainSet8,]
test8 <-march16_24_st[-trainSet8,]

# Setting Descriptive and Target Variables

train.des8 <- train8[, c(1:16)]
test.des8 <- test8[, c(1:16)]
train.cl8 <- train8$elite8
test.cl8 <- test8$elite8

### TW ###

# Splitting the data

trainSetTW <- createDataPartition(march16_24_st$TW, p = 0.7) [[1]]
trainTW <- march16_24_st[trainSetTW,]
testTW <-march16_24_st[-trainSetTW,]

# Setting Descriptive and Target Variables

train.desTW <- trainTW[, c(1:16)]
test.desTW <- testTW[, c(1:16)]
train.clTW <- trainTW$TW
test.clTW <- testTW$TW



############################# ROUND OF 32 KNN ################################################

pred.knn1.32 <- knn(train = train.des32, 
                       test = test.des32,  
                       cl = train.cl32,
                       k = 1)
pred.knn3.32 <- knn(train = train.des32, 
                       test = test.des32, 
                       cl = train.cl32, 
                       k = 3)
pred.knn5.32 <- knn(train = train.des32, 
                       test = test.des32, 
                       cl = train.cl32, 
                       k = 5)
pred.knn11.32 <- knn(train = train.des32, 
                          test = test.des32, 
                          cl = train.cl32, 
                          k = 11)

confusionMatrix(test.cl32, pred.knn11.32)
### KNN 1:
# 64.29% Accuracy
# Good Class Balance (~50/50)
# 65.75% True Negative, 62.96% True Positive

### KNN 3:
# 70.78% Accuracy
# Good Class Balance
# 70.88% True Negative, 70.66% True Positive

### KNN 5:
# Literally the same as 3 but got 1 more negative guess wrong

### KNN 7:
# Slightly worse than 3 (~68.83% Accuracy)

### KNN 11:
# Slightly better than 3: 75.32% Accuracy
# Good Class balance
# 75% True Negative, 75.68% True Positive

# KNN 11 is pretty good

############################### SWEET 16 KNN ##################################################

pred.knn1.16 <- knn(train = train.des16, 
                       test = test.des16,  
                       cl = train.cl16,
                       k = 1)
pred.knn3.16 <- knn(train = train.des16, 
                       test = test.des16, 
                       cl = train.cl16, 
                       k = 3)
pred.knn5.16 <- knn(train = train.des16, 
                       test = test.des16, 
                       cl = train.cl16, 
                       k = 5)

confusionMatrix(test.cl16, pred.knn5.16)

### KNN 1:
# 75.32% Accuracy
# 77.27% Predicted 0 class (About what you want)
# 48.57%(!) True positive, 83.19% True Negative
# Correctly identified 45.95% of class 1 predictions 

### KNN 3:
# 79.87% Accuracy
# 79.22% predicted class 0 (a little more biased)
# 59.38% True positive (improved), 88.89% True negative
# Correctly identified 51.35% of class 1 predictions

### KNN 5:
# 82.47% Accuracy
# 84.42% Predicted class 0 (red flag)
# 70.83% True positive, 84.62% True negative
# Correctly identified 45.95% of class 1 predictions

# Overall, low k models don't do well and high k models over-classify 0

############################### Elite Eight KNN ##############################################

pred.knn1.8 <- knn(train = train.des8, 
                       test = test.des8,  
                       cl = train.cl8,
                       k = 1)
pred.knn3.8 <- knn(train = train.des8, 
                       test = test.des8, 
                       cl = train.cl8, 
                       k = 3)
pred.knn5.8 <- knn(train = train.des8, 
                       test = test.des8, 
                       cl = train.cl8, 
                       k = 5)

confusionMatrix(test.cl8, pred.knn3.8)

### KNN 1:
# 82.35% Accuracy
# 80.39% Predicted as Class 0 (True rate = 88.24%)
# True Positive = 35% (!), True negative = 93.9%
# Correctly identified 58.33% of class 1 predictions

### KNN 3:
# 94.12% Accuracy
# 92.16% Predicted as Class 0 (True rate = 88.24%)
# True Positive = 87.5% , True negative = 94.68%
# Correctly identified 58.33% of class 1 predictions

# KNN 3 is Good


################################ TW KNN #############################################

pred.knn1.TW <- knn(train = train.desTW, 
                         test = test.desTW,  
                         cl = train.clTW,
                         k = 1)
pred.knn3.TW <- knn(train = train.desTW, 
                         test = test.desTW, 
                         cl = train.clTW, 
                         k = 3)
pred.knn5.TW <- knn(train = train.desTW, 
                         test = test.desTW, 
                         cl = train.clTW, 
                         k = 5)

table(test.clTW, pred.knn1.TW)

### KNN 1:
# Seems to overpredict class 1
# bad at identifying 6 win teams (0 and 1 win predicted for winners)
# Not very good at anly level except 50% acc for 5 win teams (only 2 samples)

### KNN 3:
# Seems improved
# Good at finding 0 win teams (71.76% accuracy when predicting 0)
# Mistakes seem more local with a couple outliers (1 win predicted to win 6)

### KNN 5:
# Minority classes are COMPLETELY unrepresented (0 predictions of 4,5,6 wins)

# None are good

############################## PCA Plot and Parallel Coordinates Plot ###############################################
library(ggfortify)

pca_res <- prcomp(train.matchup.des, scale. = TRUE)
autoplot(pca_res, data = train.matchup, colour = 'W', label = TRUE, shape = FALSE, frame = TRUE)

# A LOT of overlap between s16 and nons16 teams. Clear good and bad teams but HUGE middle bunch

library(GGally)
ggparcoord(data = train.matchup,
           columns = 1:6,
           alphaLines = 0.3,
           groupColumn = "W")

?ggparcoord

ncol(train.matchup)
############################### RIPPER for elite eight ##############################################

model.rules <- JRip(elite8 ~., data = train8[,c(1:16,20)])
print(model.rules)
predict.rules <- predict(model.rules, test8)

# Rule Set Evaluation

eval.rules <- confusionMatrix(predict.rules, test8$elite8)
print(eval.rules)
# ~ 76-91% accuracy
# 41.67% True positive, 92.05% True negative
# Correclty identifies 50% of class 1 predictions
# Small overprediction of class 0, but not concerning (90.2% pred, 88.24% actual)

# Accuracy with class 1 is the biggest issue. Not really usable


#################################### Binary RIPPER for sweet sixteen #########################################
view(trainSet16)

model.rules16 <- JRip(sweet16 ~., data = train16[,c(1:16,19)])
print(model.rules16)
predict.rules16 <- predict(model.rules16, test16)
# Only 2 Rules

# Rule Set Evaluation

eval.rules16 <- confusionMatrix(predict.rules16, test16$sweet16)
print(eval.rules16)
# ~ 77% - 89% accuracy, 
# ~ 86.49% true positive, 82.91% true negative
# Correctly identifies 61.54% of class 1 predictions 
# 66.23% predicted class 0 with 75.97% actual class 0
# Class imbalance problem, but other than that, ok.

################################## Binary RIPPER for second round ############################################

model.rules32 <- JRip(rtwo ~., data = train32[,c(1:16,18)])
print(model.rules32)
predict.rules32 <- predict(model.rules32, test32)

# Rule Set Evaluation

eval.rules32 <- confusionMatrix(predict.rules32, test32$rtwo)
print(eval.rules32)
# ~ 62 - 77% Accuracy
# 63.16% true negative rate, 76.92% true positive
# Class imbalance




############################## Naive Bayes Classifier for Second Round ###############################################

nb.model <- naiveBayes(rtwo ~., data = train32[,c(1:16,18)])
print(nb.model)
str(nb.model)

summary(nb.model)

# Evaluating the Model

nb.pred <- predict(nb.model, test.des32)
eval_model(nb.pred, test.cl32)

# 74.03% accuracy
# 70.51% true negative, 77.6% true positive
# Correctly identifies 71.95% of class 1 predictions
# Small class imbalance, but not too concerning

############################## Naive Bayes Classifier for Sweet 16 ###############################################

nb.model16 <- naiveBayes(sweet16 ~., data = train16[,c(1:16,19)])
print(nb.model16)
str(nb.model16)

summary(nb.model16)

# Evaluating the Model

nb.pred16 <- predict(nb.model16, test.des16)
eval_model(nb.pred16, test.cl16)

# 82.47% Accuracy
# True Positive = 83.78%, True Negative = 82.05%
# Correctly identifies 59.62% of class 1 guesses
# Overpredicts class 1 (34% pred vs 24% actual)
# I don't like the class imbalance.

############################## Naive Bayes Classifier for Elite 8 ###############################################

nb.model8 <- naiveBayes(elite8 ~., data = train8[,c(1:16,20)])
print(nb.model)
str(nb.model)

summary(nb.model)

# Evaluating the Model

nb.pred8 <- predict(nb.model8, test.des8)
eval_model(nb.pred8, test.cl8)

# Another class imbalance problem (77% predicted 0 vs 88% actual)
# 83.33% True positive, 82.22% true negative
# Correctly identifies 38.46% (!) of e8 teams
# Class imbalance!!! (25% pred 1 vs 12% actual)

############################## Naive Bayes Classifier for TW ###############################################

nb.modelTW <- naiveBayes(TW ~., data = trainTW[,c(1:17)])
print(nb.modelTW)
str(nb.modelTW)

summary(nb.modelTW)

# Evaluating the Model

nb.pred <- predict(nb.modelTW, test.desTW)
eval_model(nb.pred, test.clTW)

# Predictions do not look TOO bad
# Over predicts class 2, under predicts classes 3 and 4
# Some local mistakes but some big ones as well (5 win team predicted 0)
# Probably usable


################################# Decision Tree TW ############################################

trainTW$TW <- as.factor(trainTW$TW)
testTW$TW <- as.factor(testTW$TW)

model.nom <- J48(TW ~., data = trainTW[,c(1:17)])
summary(model.nom)

### Nominal Model Evaluation ###

predict.nom <- predict(model.nom, testTW[,c(1:17)])
eval.nom <- confusionMatrix(predict.nom, testTW$TW)
print(eval.nom)

# Performs worse than the no information rate

testTW_pred <- testTW[,c(4,5,8,17)]

testTW_pred$pred <- predict.nom
view(testDT)

testTW_pred$pred <- as.numeric(testTW_pred$pred)
testTW_pred$TW <- as.numeric(testTW_pred$TW)

summary(lm(TW ~ pred, data = testTW_pred))

# predicted values explain only 23% of variation of TW. Would ideally be 1:1

testTW_pred %>%
  ggplot(aes(x = pred, y = TW))+
  geom_point()

# Shows almost no correlation at all

testTW_pred$sqr_err <- (testTW_pred$TW - testTW_pred$pred)^2
view(testTW_pred)
mean(testTW_pred$sqr_err)

# mean sq error = 1.922

################################## Decision Tree Round of 32 ###########################################

model.nom32 <- rpart(rtwo ~., data = train32[,c(1:16,18)])
summary(model.nom32)

### Nominal Model Evaluation ###

plot(model.nom32)
plotcp(model.nom32)
printcp(model.nom32)

# need to cut at 0.11

### Pruning ###
model.nom32.pruned <- prune(model.nom32, cp = 0.11)
plot(model.nom32.pruned)
plotcp(model.nom32.pruned)
printcp(model.nom32.pruned)

predict.nom32.pruned <- predict(model.nom32.pruned, test32)

predict.nom32.pruned <- as.data.frame(predict.nom32.pruned) 
for (i in 1:nrow(predict.nom32.pruned)) {
  if(predict.nom32.pruned[i,1] > predict.nom32.pruned[i,2]){
    predict.nom32.pruned[i,3] <- 0
  } else{
    predict.nom32.pruned[i,3] <- 1
  }
}
colnames(predict.nom32.pruned) <- c("pred0", "pred1", "DT_rtwo") 
predict.nom32.pruned$DT_rtwo <- as.factor(predict.nom32.pruned$DT_rtwo)

confusionMatrix(predict.nom32.pruned$DT_rtwo, test.cl32)
# 72.08% accuracy
# 73.68% true positive, 70.51% true negative
# Good class balance
# Doesn't perform overly well

################################# Decision Tree Round of 16 ############################################

model.nom16 <- rpart(sweet16 ~., data = train16[,c(1:16,19)])
summary(model.nom16)

### Nominal Model Evaluation ###

plot(model.nom16)
plotcp(model.nom16)
printcp(model.nom16)

# Cut at 0.16

### Pruning ###
model.nom16.pruned <- prune(model.nom16, cp = 0.16)
plot(model.nom16.pruned)

predict.nom16.pruned <- predict(model.nom16.pruned, test16)

predict.nom16.pruned <- as.data.frame(predict.nom16.pruned) 
for (i in 1:nrow(predict.nom16.pruned)) {
  if(predict.nom16.pruned[i,1] > predict.nom16.pruned[i,2]){
    predict.nom16.pruned[i,3] <- 0
  } else{
    predict.nom16.pruned[i,3] <- 1
  }
}
colnames(predict.nom16.pruned) <- c("pred0", "pred1", "DT_s16") 
predict.nom16.pruned$DT_s16 <- as.factor(predict.nom16.pruned$DT_s16)

confusionMatrix(predict.nom16.pruned$DT_s16, test.cl16)

# 83.77% accuracy (CI between 79.17 and 90.83%)
# 86.49% True positive, 82.91% True negative
# 61.54% accuracy when predicting class 1
# 66.23% predicted 0, actual = 75.97%
# Bad class imbalance

#################################### SVM Round of 32 #########################################

marchsvm <- svm(rtwo ~., data = train32[,c(1:16,18)])
plot(marchsvm, train32[,c(1:16,18)], AdjEM~pt_marg)

### Predict SVM ###

test32_pred <- test32[,c(4,5,8,17,18)]

marchsvm_eval <- predict(marchsvm, test32)
test32_pred$pred <- marchsvm_eval
confusionMatrix(test32_pred$pred, test32$rtwo)

view(test32)

# 73.38% accuracy (CI between 67.05% and 81.33%)
# 73.68% true positive 73.08% true negative
# Correctly identifies 72.73% of class 1 predictions
# Good class balance
# Not overly accurate



################################## SVM Sweet Sixteen ###########################################

marchsvm16 <- svm(sweet16 ~., data = train16[,c(1:16,19)])
plot(marchsvm16, train16, AdjEM~pt_marg)

# Predict SVM

test16_pred <- test16[,c(3,4,8,17,19)]

marchsvm_eval16 <- predict(marchsvm16, test16)
test16_pred$pred <- marchsvm_eval16
confusionMatrix(test16_pred$pred, test16_pred$sweet16)
# ~ 82.47% accuracy (CI between 75.53% and 88.12%)
# ~ 40.54% true positive, 95.73% true negative
# correctly identified 83.58% of class 1 predictions
# Overpredicts class 0 (87.01% pred vs 75.97% actual)
# class imbalance is the main issue


##################### Round Predictions Using 2025 Data ################################

# I chose to proceed with KNN11 R32, KNN3 E8, RIPPER R32, RIPPER S16, Naive Bayes TW
# Naive Bayes TW was the worst performing and had class imbalance issues, but it
# was the best of the TW models and might be worth a shot. Results will be taken
# With a grain of salt.
# RIPPER 16 also had class imbalance issues that might distort predictions.
# The other models performed somewhat similarly to each other accuracy-wise. 
# All ok but not overly exciting.

predmat <- fr25_st[,c(4,5,8)]

### Ripper ###

#predmat$ripper32 <- predict(model.rules32, fr25_st)
predmat$ripper16 <- predict(model.rules16, fr25_st)

### Naive Bayes ###

predmat$nbTW <- predict(nb.modelTW, fr25_st)

### KNN ###

predmat$knn32 <- knn(train = train.des32, 
                     test = fr25_st, 
                     cl = train.cl32, 
                     k = 11)

predmat$knn8 <- knn(train = train.des8, 
                    test = fr25_st, 
                    cl = train.cl8, 
                    k = 3)


### Viewing Results ###

predmat$score <- 1/6*(as.numeric(predmat$knn32)-1) + 
  1/3*(as.numeric(predmat$ripper16)-1) + 
  1/2*(as.numeric(predmat$knn8)-1) + 
  1/6*(as.numeric(predmat$nbTW)-1)

predmat$team <- fr25$Team

view(predmat)

##################################### Matchup Predictions ########################################

matchup <- read_excel("matchup16_24.xlsx")

view(matchup)

firstround25 <- read_excel("First Round 2025.xlsx",sheet = 1)

matchup_sub <- matchup[, c(3:13)]
firstround25_sub <- firstround25[,c(3:12)]

corr <- cor(matchup_sub)
corrplot(corr, method = 'number')

# Taking out irrelevent variables (very low correlation to W)
matchup_sub <- matchup_sub[,-c(5,7,8,10)]
firstround25_sub <- firstround25_sub[,-c(5,7,8,10)]

view(matchup_sub)




###################### Standardizing Variables ###########################

matchup_st <- lapply(matchup_sub[,], standardize)
firstround25_st <- lapply(firstround25_sub[,], standardize)

matchup_st$W <- as.factor(matchup_sub$W)

matchup_st <- as.data.frame(matchup_st)
firstround25_st <- as.data.frame(firstround25_st)

view(matchup_st)

######################## Train Test Split ###################################

### Creating Train, Test, Split ###
set.seed(12345)
trainSet.matchup <- createDataPartition(matchup_st$W, p = 0.7) [[1]]
train.matchup <- matchup_st[trainSet.matchup,]
test.matchup <-matchup_st[-trainSet.matchup,]

### Separating target and descriptive variables ###

train.matchup.des <- train.matchup[, c(1:6)]
test.matchup.des <- test.matchup[, c(1:6)]
train.matchup.cl <- train.matchup$W
test.matchup.cl <- test.matchup$W

################################## Matchup Predictions KNN ###########################################

### KNN 1:5 for Training and testing data ###

pred.matchup.knn1 <- knn(train = train.matchup.des, 
                          test = test.matchup.des,  
                          cl = train.matchup.cl,
                          k = 1)
pred.matchup.knn3 <- knn(train = train.matchup.des, 
                          test = test.matchup.des, 
                          cl = train.matchup.cl, 
                          k = 3)
pred.matchup.knn5 <- knn(train = train.matchup.des, 
                          test = test.matchup.des, 
                          cl = train.matchup.cl, 
                          k = 11)

confusionMatrix(test.matchup.cl, pred.matchup.knn3)

### KNN 1:
# 67.11% accuracy
# Good balance

### KNN 3:
# 73.68% accuracy
# Good balance

### KNN 5:
# 71.71% accuracy
# ok balance

### KNN 7:
# 73.68% accuracy
# Little to no improvement for higher K values

# overall, KNN 3 is pretty good, no improvement for higher k

################################## Matchup Predictions RIPPER ###########################################


matchup.model.rules <- JRip(W ~., data = train.matchup)
print(matchup.model.rules)
matchup.predict.rules <- predict(matchup.model.rules, test.matchup.des)

# Rule Set Evaluation

matchup.eval.rules <- confusionMatrix(matchup.predict.rules, test.matchup.cl)
print(matchup.eval.rules)
# 73.03% accuracy
# Reference ~ 48.03% class 1, prediction ~ 44.74%
# Ok balance


#################################### Naive Bayes Classifier #########################################

matchup.nb.model <- naiveBayes(W ~., data = train.matchup)
print(matchup.nb.model)
str(matchup.nb.model)

summary(matchup.nb.model)

# Evaluating the Model

matchup.nb.pred <- predict(matchup.nb.model, test.matchup.des)
eval_model(matchup.nb.pred, test.matchup.cl)


# 71.05% accuracy, 67.09% true positive, 75.34% true negative 
# Correctly identifies 74.65% of class 1 predictions
# OK balance

############################## Decision Tree ###############################################

matchup.model.nom <- rpart(W ~., data = train.matchup)
summary(matchup.model.nom)

plot(matchup.model.nom)
plotcp(matchup.model.nom)
printcp(matchup.model.nom)

# cut at 0.084

### Pruning

matchup.model.nom.pruned <- prune(matchup.model.nom, cp = 0.084)
plot(matchup.model.nom.pruned)
plotcp(matchup.model.nom.pruned)
printcp(matchup.model.nom.pruned)

matchup.predict.nom.pruned <- predict(matchup.model.nom.pruned, test.matchup)

matchup.predict.nom.pruned <- as.data.frame(matchup.predict.nom.pruned)
for (i in 1:nrow(matchup.predict.nom.pruned)) {
  if(matchup.predict.nom.pruned[i,1] > matchup.predict.nom.pruned[i,2]){
    matchup.predict.nom.pruned[i,3] <- 0
  } else{
    matchup.predict.nom.pruned[i,3] <- 1
  }
}
colnames(matchup.predict.nom.pruned) <- c("pred0", "pred1", "DT") 
matchup.predict.nom.pruned$DT <- as.factor(matchup.predict.nom.pruned$DT)

confusionMatrix(matchup.predict.nom.pruned$DT, test.matchup.cl)

# 71.71% accuracy (CI between 63.84% and 78.71%)
# 72.15% True positive, 71.23% True negative
# Correctly identifies 73.08% of class 1 predictions
# Good balance
# Seems OK



################################ SVM #############################################


# for(i in 1:8){
  #for(j in 1:10){
    #svm_pol <- svm(class ~., kernel = "polynomial", data = train_val, cost = c[i], degree= Degree[j], coef0 = 1)
    #svm_pol_eval = predict(svm_pol, testing[,c(1:4)])
    #pol_acc = confusionMatrix(svm_pol_eval, reference = testing$class)
    #print(c(Degree[j], c[i], pol_acc$overall[1]))
  #}
#}

c = 10^(-4:3)
g = 10^(-4:3)
d = c(1:10)

for(i in 1:8){
  for(j in 1:10){
  matchupsvm <- svm(W ~., data = train.matchup, kernel = "radial", cost = c[i], degree = d[j], coef0 = 1)
  svm_eval <- predict(matchupsvm, test.matchup.des)
  svm_acc = confusionMatrix(svm_eval, reference = test.matchup.cl)
  print(c(d[j], c[i], svm_acc$overall[1]))
  }
}

matchupsvm_c <- svm(W ~., data = train.matchup, type = "C", cost = 1000, gamma = 0.01, coef0 = 1)
svm_eval_c <- predict(matchupsvm_c, test.matchup.des)
svm_acc_c = confusionMatrix(svm_eval_c, reference = test.matchup.cl)
print(svm_acc_c)

?svm

# 76.32% accuracy
# Good class balance
# Performs similarly for predicted 0 and 1

matchupsvm_p <- svm(W ~., data = train.matchup, kernel = "polynomial", cost = 1000, degree = 2, coef0 = 1)
svm_eval_p <- predict(matchupsvm_p, test.matchup.des)
svm_acc_p = confusionMatrix(svm_eval_p, reference = test.matchup.cl)
print(svm_acc_p)

# 77.63% accuracy
# OK class balance

matchupsvm_r <- svm(W ~., data = train.matchup, kernel = "radial", cost = 1, degree = 3, coef0 = 1)
svm_eval_r <- predict(matchupsvm_r, test.matchup.des)
svm_acc_r = confusionMatrix(svm_eval_r, reference = test.matchup.cl)
print(svm_acc_r)

# 75% accuracy
# Good class balance


plot(matchupsvm, test.matchup, AdjEMdiff_st ~ AdjSOSdiff_st)


################################ ANN ############################################# 


matchup.nn.formula <- as.formula(paste("W ~ ", 
                                         paste(names(train.matchup[!names(train.matchup) %in% 'W']), 
                                               collapse = " + "), sep=""))


# Build NN model with default hidden layer (1 hidden layer with 1 node)

matchup.model.nn1 <- neuralnet(W ~., data=train.matchup)


# Plot the network

plot(matchup.model.nn1)

# Simple Neural Network
# NOTE: predicting with the neural network model uses compute() not predict()
matchup.eval.nn1 <- compute(matchup.model.nn1, test.matchup.des)

confusionMatrix(as.factor(round(matchup.eval.nn1$net.result[,2])), 
                as.factor(test.matchup.cl))

# Accuracy: 76.97%, true positive: 75.95%, true negative: 78.08%
# Predicts class 0 50% (48.03% actual)


### Build NN model with 2 hidden layers (3 and 2 nodes)
# Use backpropagation with 0.01 learning rate


matchup.model.nn2 <- neuralnet(W ~., data=train.matchup, 
                                 hidden=c(1,2,3),
                                 algorithm="rprop+",
                                  stepmax = 1000000,
                                  threshold = 0.005,
                                  learningrate.limit = c(0.00001,0.1))

# Plot the network

plot(matchup.model.nn2)

# Evaluate neural network model

matchup.eval.nn2 <- compute(matchup.model.nn2, test.matchup.des)
matchup.eval.nn2$net.result <- as.numeric(matchup.eval.nn2$net.result[,2])
confusionMatrix(factor(round(matchup.eval.nn2$net.result), levels=c("0", "1")), 
  as.factor(test.matchup.cl))

# 70.39% accuracy, 65.75% true negative, 74.68% true positive
# tiny class balance problem
# simpler model is better


############### Predicting 2025 GAMES ########################################
# I am choosing KNN3, NB, all the SVMs, and the Simple ANN

m_predmat <- firstround25[,c(1:3)]


### KNN3 ###

m_predmat$knn3 <- knn(train = matchup_std[,c(1:6)], 
                         test = firstround25_st, 
                         cl = matchup_std$W, 
                         k = 3)

### RIPPER ###

m_predmat$rip <- predict(matchup.model.rules, firstround25_st)

### SVMs ###

#m_predmat$svm_c <- predict(matchupsvm_c, firstround25_st)
m_predmat$svm_r <- predict(matchupsvm_r, firstround25_st)
#m_predmat$svm_p <- predict(matchupsvm_p, firstround25_st)

### Simple ANN ###

matchup.eval.nn25 <- compute(matchup.model.nn1, firstround25_st)

m_predmat$ANN <- as.factor(round(matchup.eval.nn25$net.result[,2]))

view(m_predmat)

# dropping naive bayes, svm_p, and svm_c because they only choose higher seeded teams


################### Predicting the rest of the way #########################


knn_r2 <- read_excel('full16_25.xlsx', sheet = 3)
ann_r2 <- read_excel('First Round 2025.xlsx', sheet = 3)
svm_r_r2 <- read_excel('First Round 2025.xlsx', sheet = 4)


knn_st <- knn_r2[,-c(1,2,7,9,10,12)]
ann_st <- ann_r2[,-c(1,2,7,9,10,12)]
svm_st <- svm_r_r2[,-c(1,2,7,9,10,12)]

view(svm_st)

# Converting data 
df_list <- list(knn_st)
df_list <- lapply(df_list, scale)
df_list <- lapply(df_list, as.data.frame)

view(df_list[[1]])


### KNN ###

knn_predmat <- knn_r2[,c(1:3)]

view(knn_predmat)

knn_predmat$knn3 <- knn(train = train.matchup.des, 
                      test = df_list[[1]], 
                      cl = train.matchup.cl, 
                      k = 3)

view(knn_predmat)

### SVMs ###

svm_predmat <- svm_r_r2[,c(1:3)]

svm_predmat$svm_r <- predict(matchupsvm_r, df_list[[2]])

view(svm_predmat)

### Simple ANN ###

ann_predmat <- ann_r2[,c(1:3)]


r2.eval.nn25 <- compute(matchup.model.nn1, df_list[[3]])

ann_predmat$ANN <- as.factor(round(r2.eval.nn25$net.result[,2]))

view(ann_predmat)



######################### Identifying Nearest Neighbors for Fun #######################

### Matchups ###

results <- list()

matchup_st_train <- matchup_st[,c(1:6)]

view(firstround25_st)
view(matchup_st_train)

matchup_st_train <- matchup_st_train %>% 
  mutate_all(as.numeric)
firstround25_st <- firstround25_st %>% 
  mutate_all(as.numeric)

# Loop through each row in firstround25
for (i in 1:nrow(firstround25_st)) {
  # Extract the numeric part of firstround25 (excluding column 1)
  vec1 <- firstround25_st[i,1:ncol(firstround25_st)]
  
  # Compute Euclidean distances between vec1 and each row in df2 (excluding column 1)
  distances <- apply(matchup_st_train, 1, function(vec2) {
    sqrt(sum((vec1 - vec2)^2))  # Euclidean distance formula
  })
  
  # Find the minimum distance and its index in matchup_std[,c(1:6)]
  top_3_indices <- order(distances)[1:3]  # Sort and select the first 3
  top_3_distances <- distances[top_3_indices]
  
  # Store firstround25[i,1], matchup_std[min_index,1], and min_distance
  results[[i]] <- data.frame(
    firstround25_id = firstround25[i, c(1,2)], 
    match_1_id = matchup[top_3_indices[1], c(1,2,13)],
    distance_1 = top_3_distances[1],
    match_2_id = matchup[top_3_indices[2], c(1,2,13)],
    distance_2 = top_3_distances[2],
    match_3_id = matchup[top_3_indices[3], c(1,2,13)],
    distance_3 = top_3_distances[3]
  )
}


# Convert list to dataframe
df3 <- do.call(rbind, results)

# View result
view(df3)


### Teams ###

view(march16_24)

results2 <- list()

march16_24_st_train <- march16_24_st[,c(1:16)]



march16_24_st_train <- march16_24_st_train %>% 
  mutate_all(as.numeric)
fr25_st <- fr25_st %>% 
  mutate_all(as.numeric)

# Loop through each row in firstround25
for (i in 1:nrow(fr25_st)) {
  # Extract the numeric part of firstround25 (excluding column 1)
  vec1 <- fr25_st[i,1:ncol(fr25_st)]
  
  # Compute Euclidean distances between vec1 and each row in df2 (excluding column 1)
  distances <- apply(march16_24_st_train, 1, function(vec2) {
    sqrt(sum((vec1 - vec2)^2))  # Euclidean distance formula
  })
  
  # Find the minimum distance and its index in matchup_std[,c(1:6)]
  top_3_indices <- order(distances)[1:3]  # Sort and select the first 3
  top_3_distances <- distances[top_3_indices]
  
  # Store firstround25[i,1], matchup_std[min_index,1], and min_distance
  results2[[i]] <- data.frame(
    fr25_id = fr25[i, c(1,2)], 
    match_1_id = march16_24[top_3_indices[1], c(1,51)],
    distance_1 = top_3_distances[1],
    match_2_id = march16_24[top_3_indices[2], c(1,51)],
    distance_2 = top_3_distances[2],
    match_3_id = march16_24[top_3_indices[3], c(1,51)],
    distance_3 = top_3_distances[3]
  )
}


# Convert list to dataframe
df4 <- do.call(rbind, results2)

# View result
view(df4)

df4 <- df4[,-2]

### Similarity Matrix ###

view(matchup_st)
view(similarity_matrix)

similarity_matrix <- lapply(matchup_st,as.numeric)
similarity_matrix$W <- similarity_matrix$W - 1

similarity_matrix <- as.data.frame(similarity_matrix)

similarity_matrix_train <- similarity_matrix[,c(1:6)]

view(firstround25_st)



## Doing the thing


results3 <- list()

# Loop through each row in firstround25
for (i in 1:nrow(firstround25_st)) {
  # Extract the numeric part of firstround25 (excluding column 1)
  vec1 <- firstround25_st[i,1:ncol(firstround25_st)]
  
  # Compute Euclidean distances between vec1 and each row in df2 (excluding column 1)
  distances <- apply(similarity_matrix_train, 1, function(vec2) {
    sqrt(sum(vec1 - vec2)^2)  # Euclidean distance formula
  })
  
  # Find the minimum distance and its index in matchup_std[,c(1:6)]
  top_3_indices <- order(distances)[1:3]  # Sort and select the first 3
  top_3_distances <- distances[top_3_indices]
  
  # Store firstround25[i,1], matchup_std[min_index,1], and min_distance
  results3[[i]] <- data.frame(
    firstround25_id = firstround25[i, c(1,2)], 
    match_1_id = matchup[top_3_indices[1], c(1,2,13)],
    distance_1 = top_3_distances[1],
    match_2_id = matchup[top_3_indices[2], c(1,2,13)],
    distance_2 = top_3_distances[2],
    match_3_id = matchup[top_3_indices[3], c(1,2,13)],
    distance_3 = top_3_distances[3]
  )
}

view(matchup)


# Convert list to dataframe
df5 <- do.call(rbind, results3)

view(df5)










