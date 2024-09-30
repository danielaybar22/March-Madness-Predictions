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


march16_22 <- read_excel("~/Downloads/March Madness/2016-22 march madness.xlsx")
march16_23 <- read_excel("~/Downloads/March Madness/2016-23 march madness.xlsx")
view(march16_23)
march16_22_sub <- march16_22[, c(6, 9,10,12,13,16,17,18,19,20,21,22,24,29,33,35,36,39,40,41,42,43,44,45,47,48,49,50,51,52,54,55,56,59,53)]
march16_22_sub$AdjSOS <- as.numeric(march16_22$AdjSOS)
view(march16_22_sub)
firstround24_full <- read_excel("~/Downloads/March Madness/2024_fullstats.xlsx")

for (i in 1:nrow(march16_23)) {
  if(march16_23[i,55] > 200){
    march16_23[i,55] <- 102.6
  } else{
    march16_23[i,55] <- march16_23[i,55] 
  }
}

march16_23$fta_marg <- march16_23$FTA - march16_23$OFTA
march16_23$pt_marg <- march16_23$PTS - march16_23$OPTS
march16_23$atr <- march16_23$AST/march16_23$TOV
march16_23$oatr <- march16_23$OAST/march16_23$OTOV
march16_23$pm <- march16_23$TOM + march16_23$RBM

firstround24_full$fta_marg <- firstround24_full$FTA - firstround24_full$OFTA
firstround24_full$pt_marg <- firstround24_full$PTS - firstround24_full$OPTS
firstround24_full$atr <- firstround24_full$AST/firstround24_full$TOV
firstround24_full$oatr <- firstround24_full$OAST/firstround24_full$OTOV
firstround24_full$pm <- firstround24_full$TOM + firstround24_full$RBM

essential_subset <- march16_23[,c(52:62,47:51,29,35,47,6,12,24,16,19:21)]
essential_subset$FGper <- essential_subset$`FG%`
essential_subset$THREEper <- essential_subset$`3P%`
essential_subset$OFGper <- essential_subset$`OFG%`
essential_subset$O3per <- essential_subset$`O3P%`
view(essential_subset)

essential_subset <- essential_subset[,-c(17,18,20,21)]

march16_23_sub <- march16_23[,c(52,6,12,19,29,35,51,53:55,58:60,62)]
firstround24_full_sub <- firstround24_full[,c(52,6,12,19,29,35,51,53:55,58:60,62)]


corr <- cor(march16_23_sub)
corrplot(corr, method = 'color')

#############################################################################
# Kfold cross validation linear regression

evaluator.march.lm <- function(subset) {
  # Use k-fold cross validation
  k <- 10
  splits <- runif(nrow(essential_subset))
  results = sapply(1:k, function(i) {
    test.idx <- (splits >= (i - 1) / k) & (splits < i / k)
    train.idx <- !test.idx
    test <- essential_subset[test.idx, , drop=FALSE]
    train <- essential_subset[train.idx, , drop=FALSE]
    linear <- lm(as.simple.formula(subset, "TW"), train)
    inv_mae = 1/mean(abs(linear$residuals))
    return(inv_mae)
  })
  print(subset)
  print(mean(results))
  return(mean(results))
}

# Forward Selection
subset <- forward.search(names(essential_subset)[-1], evaluator.march.lm)
f <- as.simple.formula(subset, "TW")  
f

# Backward Selection

subset <- backward.search(names(essential_subset)[-1], evaluator.march.lm)
b <- as.simple.formula(subset, "TW")  
b

# Hillclimbing Search

subset <- hill.climbing.search(names(essential_subset)[-1], evaluator.march.lm)
h <- as.simple.formula(subset, "TW") 
h

# Exhaustive Search
subset <- exhaustive.search(names(essential_subset)[-1], evaluator.march.lm)
e <- as.simple.formula(subset, "TW")
e

#############################################################################
### KNN

# Standardizing Variables
standardization <- function(data) {
  (data - mean(data)) / sd(data)
}



essential_subset$AdjEM_st <- standardization(essential_subset$AdjEM)
essential_subset$AdjO_st <- standardization(essential_subset$AdjO)
essential_subset$AdjD_st <- standardization(essential_subset$AdjD)
essential_subset$AdjT_st <- standardization(essential_subset$AdjT)
essential_subset$fta_marg_st <- standardization(essential_subset$fta_marg)
essential_subset$pt_marg_st <- standardization(essential_subset$pt_marg)
essential_subset$atr_st <- standardization(essential_subset$atr)
essential_subset$oatr_st <- standardization(essential_subset$oatr)
essential_subset$pm_st <- standardization(essential_subset$pm)
essential_subset$Seed_st <- standardization(essential_subset$Seed)
essential_subset$ORB_st <- standardization(essential_subset$ORB)
essential_subset$BLK_st <- standardization(essential_subset$BLK)
essential_subset$FGper_st <- standardization(essential_subset$FGper)
essential_subset$THREEper_st <- standardization(essential_subset$THREEper)
essential_subset$OFGper_st <- standardization(essential_subset$OFGper)
essential_subset$O3per_st <- standardization(essential_subset$O3per)

essential_subset_st <- essential_subset[,c(1,27:42)]
view(essential_subset_st)

march16_23_sub$FG_perst <- standardization(march16_23_sub$`FG%`)
march16_23_sub$`3_perst` <- standardization(march16_23_sub$`3P%`)
march16_23_sub$OFG_perst <- standardization(march16_23_sub$`OFG%`)
march16_23_sub$`O3_perst` <- standardization(march16_23_sub$`O3P%`)
march16_23_sub$ASTst <- standardization(march16_23_sub$AST)
march16_23_sub$Seedst <- standardization(march16_23_sub$Seed)
march16_23_sub$AdjEMst <- standardization(march16_23_sub$AdjEM)
march16_23_sub$AdjOst <- standardization(march16_23_sub$AdjO)
march16_23_sub$AdjDst <- standardization(march16_23_sub$AdjD)
march16_23_sub$pt_margst <- standardization(march16_23_sub$pt_marg)
march16_23_sub$fta_margst <- standardization(march16_23_sub$fta_marg)
march16_23_sub$ATRst <- standardization(march16_23_sub$atr)
march16_23_sub$PMst <- standardization(march16_23_sub$pm)

march16_23_st <- march16_23_sub[,c(15:27)]
march16_23_st$TW <- march16_23$TW

firstround24_full_sub$FG_perst <- standardization(firstround24_full_sub$`FG%`)
firstround24_full_sub$`3_perst` <- standardization(firstround24_full_sub$`3P%`)
firstround24_full_sub$OFG_perst <- standardization(firstround24_full_sub$`OFG%`)
firstround24_full_sub$`O3_perst` <- standardization(firstround24_full_sub$`O3P%`)
firstround24_full_sub$ASTst <- standardization(firstround24_full_sub$AST)
firstround24_full_sub$Seedst <- standardization(firstround24_full_sub$Seed)
firstround24_full_sub$AdjEMst <- standardization(firstround24_full_sub$AdjEM)
firstround24_full_sub$AdjOst <- standardization(firstround24_full_sub$AdjO)
firstround24_full_sub$AdjDst <- standardization(firstround24_full_sub$AdjD)
firstround24_full_sub$pt_margst <- standardization(firstround24_full_sub$pt_marg)
firstround24_full_sub$fta_margst <- standardization(firstround24_full_sub$fta_marg)
firstround24_full_sub$ATRst <- standardization(firstround24_full_sub$atr)
firstround24_full_sub$PMst <- standardization(firstround24_full_sub$pm)

firstround24_full_st <- firstround24_full_sub[,c(15:27)]










view(firstround24_full_st)

#############################################################################
# ROUND OF 32 KNN 

for (i in 1:nrow(march16_23_st)) {
  if(march16_23_st[i,14]>= 1){
    march16_23_st[i,15] <- 1
  } else{
    march16_23_st[i,15] <- 0
  }
}
colnames(march16_23_st)[15] <- "rtwo"
view(march16_23_st)
march16_23_st$rtwo <- as.factor(march16_23_st$rtwo)

for (i in 1:nrow(essential_subset_st)) {
  if(essential_subset_st[i,1]>= 1){
    essential_subset_st[i,18] <- 1
  } else{
    essential_subset_st[i,18] <- 0
  }
}
colnames(essential_subset_st)[18] <- "rtwo"
view(essential_subset_st)
essential_subset_st$rtwo <- as.factor(essential_subset_st$rtwo)





trainSet32 <- createDataPartition(march16_23_st$rtwo, p = 0.7) [[1]]
march.train32 <- march16_23_st[trainSet32,]
march.test32 <-march16_23_st[-trainSet32,]

march.train.knn32 <- march.train32[, c(1:13)]
march.test.knn32 <- march.test32[, c(1:13)]
march.train.knncl32 <- march.train32$rtwo
march.test.knncl32 <- march.test32$rtwo

march.pred.knn1.32 <- knn(train = march.train.knn32, 
                       test = march.test.knn32,  
                       cl = march.train.knncl32,
                       k = 1)
march.pred.knn3.32 <- knn(train = march.train.knn32, 
                       test = march.test.knn32, 
                       cl = march.train.knncl32, 
                       k = 3)
march.pred.knn5.32 <- knn(train = march.train.knn32, 
                       test = march.test.knn32, 
                       cl = march.train.knncl32, 
                       k = 5)
march.pred.knn13.32 <- knn(train = march.train.knn32, 
                          test = march.test.knn32, 
                          cl = march.train.knncl32, 
                          k = 13)

table(march.test.knncl32, march.pred.knn13.32)
#FOR MARCH_ST Roughly 75% correct classification rate, 
# 70.1% true positive,  80.3% true negative

# march.test.knn32$pred <- march.pred.knn7.32
# march.test.knn32$rtwo <- march.test.knncl32


#################################################################################
# SWEET 16 KNN

for (i in 1:nrow(march16_23_st)) {
  if(march16_23_st[i,14]>= 2){
    march16_23_st[i,16] <- 1
  } else{
    march16_23_st[i,16] <- 0
  }
}
colnames(march16_23_st)[16] <- "sweet16"
march16_23_st$sweet16 <- as.factor(march16_23_st$sweet16)



trainSet16 <- createDataPartition(march16_23_st$sweet16, p = 0.6) [[1]]
march.train16 <- march16_23_st[trainSet16,]
march.test16 <- march16_23_st[-trainSet16,]

march.train.knn16 <- march.train16[, c(1:13)]
march.test.knn16 <- march.test16[, c(1:13)]
march.train.knncl16 <- march.train16$sweet16
march.test.knncl16 <- march.test16$sweet16

march.pred.knn1.16 <- knn(train = march.train.knn16, 
                       test = march.test.knn16,  
                       cl = march.train.knncl16,
                       k = 1)
march.pred.knn3.16 <- knn(train = march.train.knn16, 
                       test = march.test.knn16, 
                       cl = march.train.knncl16, 
                       k = 3)
march.pred.knn5.16 <- knn(train = march.train.knn16, 
                       test = march.test.knn16, 
                       cl = march.train.knncl16, 
                       k = 7)

confusionMatrix(march.test.knncl16, march.pred.knn5.16)
# Roughly 82.2% correct classification rate
# 84.1% true positive rate, 70% true negative rate
# Sample ~ 24% class 1, pred ~ 15% class 1

#############################################################################
# Elite Eight Prediction

for (i in 1:nrow(march16_23_st)) {
  if(march16_23_st[i,14]>= 3){
    march16_23_st[i,17] <- 1
  } else{
    march16_23_st[i,17] <- 0
  }
}
colnames(march16_23_st)[17] <- "elite8"
march16_23_st$selite8 <- as.factor(march16_23_st$elite8)



trainSet8 <- createDataPartition(march16_23_st$elite8, p = 0.7) [[1]]
march.train8 <- march16_23_st[trainSet8,]
march.test8 <-march16_23_st[-trainSet8,]

march.train.knn8 <- march.train8[, c(1:13)]
march.test.knn8 <- march.test8[, c(1:13)]
march.train.knncl8 <- march.train8$elite8
march.test.knncl8 <- march.test8$elite8

march.pred.knn1.8 <- knn(train = march.train.knn8, 
                       test = march.test.knn8,  
                       cl = march.train.knncl8,
                       k = 1)
march.pred.knn3.8 <- knn(train = march.train.knn8, 
                       test = march.test.knn8, 
                       cl = march.train.knncl8, 
                       k = 3)
march.pred.knn5.8 <- knn(train = march.train.knn8, 
                       test = march.test.knn8, 
                       cl = march.train.knncl8, 
                       k = 5)

table(march.test.knncl8, march.pred.knn5.8)
# Rougly 86.5% correct classification rate
# 86.6% true positive rate
# 71% true negative rate, wide range depending on test set (~30%-100%)

#############################################################################
# TW Prediction

trainSetTW <- createDataPartition(march16_23_st$TW, p = 0.7) [[1]]
march.trainTW <- march16_23_st[trainSetTW,]
march.testTW <-march16_23_st[-trainSetTW,]

march.train.knnTW <- march.trainTW[, c(1:13)]
march.test.knnTW <- march.testTW[, c(1:13)]
march.train.knnclTW <- march.trainTW$TW
march.test.knnclTW <- march.testTW$TW


march.pred.knn1.TW <- knn(train = march.train.knnTW, 
                         test = march.test.knnTW,  
                         cl = march.train.knnclTW,
                         k = 1)
march.pred.knn3.TW <- knn(train = march.train.knnTW, 
                         test = march.test.knnTW, 
                         cl = march.train.knnclTW, 
                         k = 3)
march.pred.knn5.TW <- knn(train = march.train.knnTW, 
                         test = march.test.knnTW, 
                         cl = march.train.knnclTW, 
                         k = 5)

table(march.test.knnclTW, march.pred.knn5.TW)

# Not very good

#############################################################################
# PCA Plot and Parallel Coordinates Plot
library(ggfortify)
pca_res <- prcomp(march.train.knn16, scale. = TRUE)
autoplot(pca_res, data = march.train16, colour = 'dancing', label = TRUE, shape = FALSE, frame = TRUE)

library(GGally)
ggparcoord(data = sweetsixteen,
           columns = 19:35,
           alphaLines = 0.3,
           groupColumn = "eliteight")

march16_22_sub4 <- march16_22_sub3[, c(18:35)]

#############################################################################
# RIPPER for elite eight

set.seed(1)
trainSet <- createDataPartition(march16_23_st$elite8, p=.6)[[1]]
march16_23_st.train <- march16_23_st[trainSet,]
march16_23_st.test <- march16_23_st[-trainSet,]

march.model.rules <- JRip(selite8 ~., data = march16_23_st.train[,c(1:13,18)])
print(march.model.rules)
march.predict.rules <- predict(march.model.rules, march16_23_st.test)

# Rule Set Evaluation

march.eval.rules <- confusionMatrix(march.predict.rules, march16_23_st.test$selite8)
print(march.eval.rules)
# ~ 80-90% accuracy
# ~ 90.4% true positive rate -> 90-93.4% (most likely 92%)
# ~ 33% true negative rate -> 40%-80% (most likely 60%)
# Not good at evaluating elite 8 teams. most likely need larger test set.
# Much better with 60% training data

#############################################################################
# Binary RIPPER for sweet sixteen

trainSet16 <- createDataPartition(march16_23_st$sweet16, p = 0.6) [[1]]
march.train16 <- march16_23_st[trainSet16,]
march.test16 <- march16_23_st[-trainSet16,]

march.model.rules16 <- JRip(sweet16 ~., data = march.train16[,c(1:13,16)])
print(march.model.rules16)
march.predict.rules16 <- predict(march.model.rules16, march.test16)

# Rule Set Evaluation

march.eval.rules16 <- confusionMatrix(march.predict.rules16, march.test16$sweet16)
print(march.eval.rules16)
# ~ 78.1 - 81% accuracy, 
# ~ 87.5% - 90.8% true positive, 
# ~ 54.0% - 61.9% true negative

#############################################################################
# Binary RIPPER for second round

trainSet32 <- createDataPartition(march16_23_st$rtwo, p = 0.67) [[1]]
march.train32 <- march16_23_st[trainSet32,]
march.test32 <-march16_23_st[-trainSet32,]

march.model.rules32 <- JRip(rtwo ~., data = march.train32[,c(1:13,15)])
print(march.model.rules32)
march.predict.rules32 <- predict(march.model.rules32, march.test32)

# Rule Set Evaluation

march.eval.rules32 <- confusionMatrix(march.predict.rules32, march.test32$rtwo)
print(march.eval.rules32)
# 72.9% Accuracy
# 74.7% true negative rate
# 86.9% true positive
# disproportionately select 1 or 0 ( ~ 60% in pred vs 49.6% in actual)
## never actaully gets the proportions right

#############################################################################
# Naive Bayes Classifier for Sweet 16

trainSetNB <- createDataPartition(march16_23_st$sweet16, p = 0.67) [[1]]
march.trainNB <- march16_23_st[trainSetNB,]
march.testNB <-march16_23_st[-trainSetNB,]

march.nb.model <- naiveBayes(sweet16 ~., data = march.trainNB[,c(1:13,16)])
print(march.nb.model)
str(march.nb.model)

summary(march.nb.model)

# Evaluating the Model

march.nb.pred <- predict(march.nb.model, march.testNB)
eval_model(march.nb.pred, march.testNB$sweet16)

# Essentially 0 error reduction. As good as random guessing given the apriori

#############################################################################
# Decision Tree TW

trainSetDT <- createDataPartition(march16_23_st$TW, p=.67)[[1]]
march.trainDT <- march16_23_st[trainSetDT,]
march.testDT <- march16_23_st[-trainSetDT,]



march.model.nom <- J48(TW ~., data = march.trainDT[,c(1:14)])
summary(march.model.nom)

# Nominal Model Evaluation

march.predict.nom <- predict(march.model.nom, march.testDT)
march.eval.nom <- confusionMatrix(march.predict.nom, march.testDT$TW)
print(march.eval.nom)

march.testDT$pred <- march.predict.nom
view(march.testDT)

march.testDT$pred <- as.numeric(march.testDT$pred)
march.testDT$TW <- as.numeric(march.testDT$TW)

summary(lm(TW ~ pred, data = march.testDT))

march.testDT %>%
  ggplot(aes(x = pred, y = TW))+
  geom_point()

march.testDT$abs_err <- abs(march.testDT$TW - march.testDT$pred)
view(march.testDT)

nom_eval_mat <- march.testDT[,c(10,18,37,38)]
view(nom_eval_mat)

# BAD

#############################################################################
# Decision Tree Round of 32

trainSetDT32 <- createDataPartition(march16_23_st$rtwo, p=.67)[[1]]
march.trainDT32 <- march16_23_st[trainSetDT32,]
march.testDT32 <- march16_23_st[-trainSetDT32,]
march16_23_st$rtwo <- as.factor(march16_23_st$rtwo)

march.model.nom32 <- rpart(rtwo ~., data = march16_23_st[,c(1:13,15)])
summary(march.model.nom32)

# Nominal Model Evaluation

march.predict.nom32 <- predict(march.model.nom32, march.testDT32)
march.testDT32$p1 <- march.predict.nom32
win <- subset(march.testDT32, p1[,"1"] > p1[, "0"])
lose <- subset(march.testDT32, p1[,"0"] > p1[,"1"])
win$winner <- 1
lose$winner <- 0

march.testDT32 <- rbind(win, lose)

table(march.testDT32$winner, march.testDT32$rtwo)

plot(march.model.nom32)
plotcp(march.model.nom32)
printcp(march.model.nom32)

# Pruning
march.model.nom32.pruned <- prune(march.model.nom32, cp = 0.015696)
plot(march.model.nom32.pruned)
plotcp(march.model.nom32.pruned)
printcp(march.model.nom32.pruned)

march.predict.nom32.pruned <- predict(march.model.nom32.pruned, march.testDT32)

march.predict.nom32.pruned <- as.data.frame(march.predict.nom32.pruned) 
for (i in 1:nrow(march.predict.nom32.pruned)) {
  if(march.predict.nom32.pruned[i,1] > march.predict.nom32.pruned[i,2]){
    march.predict.nom32.pruned[i,3] <- 0
  } else{
    march.predict.nom32.pruned[i,3] <- 1
  }
}
colnames(march.predict.nom32.pruned) <- c("pred0", "pred1", "DT_rtwo") 
march.predict.nom32.pruned$DT_rtwo <- as.factor(march.predict.nom32.pruned$DT_rtwo)

confusionMatrix(march.predict.nom32.pruned$DT_rtwo, march.testDT32$rtwo)
# 81% accuracy
# 81.1% true negative
# 80.8% true positive

# Predicting 2024 data 

DTrtwo_24 <- predict(march.model.nom32.pruned, firstround24_full_st)

DTrtwo_24 <- as.data.frame(DTrtwo_24) 
for (i in 1:nrow(DTrtwo_24)) {
  if(DTrtwo_24[i,1] > DTrtwo_24[i,2]){
    DTrtwo_24[i,3] <- 0
  } else{
    DTrtwo_24[i,3] <- 1
  }
}
colnames(DTrtwo_24) <- c("pred0", "pred1", "DT_rtwo")
view(DTrtwo_24)

full_pred_mat <- cbind(firstround24_full[,c(1,53)], DTrtwo_24[,c(2,3)])
view(full_pred_mat)

full_pred_mat2 <- full_pred_mat[c(1:32,29:32,33:36),1]
full_pred_mat2 <- as.data.frame(full_pred_mat2)
full_pred_mat2[,2] <- full_pred_mat[c(37:64,34,36,33,35,66,68,65,67,65,66,67,68),1]
full_pred_mat2[,3] <- full_pred_mat[c(1:32,29:32,33:36),3]
full_pred_mat2[,4] <- full_pred_mat[c(37:64,34,36,33,35,66,68,65,67,65,66,67,68),3]
full_pred_mat2[,5] <- full_pred_mat[c(1:32,29:32,33:36),4]
full_pred_mat2[,6] <- full_pred_mat[c(37:64,34,36,33,35,66,68,65,67,65,66,67,68),4]
full_pred_mat2[,7] <- full_pred_mat2[,5] - full_pred_mat2[,6]
view(full_pred_mat2)
nrow(test)
#############################################################################
# Decision Tree Round of 16

trainSetDT16 <- createDataPartition(march16_23_st$sweet16, p=.6)[[1]]
march.trainDT16 <- march16_23_st[trainSetDT,]
march.testDT16 <- march16_23_st[-trainSetDT,]

march.model.nom16 <- rpart(sweet16 ~., data = march16_23_st[,c(1:13,16)])
summary(march.model.nom16)
plot(march.model.nom16)
plotcp(march.model.nom16)
printcp(march.model.nom16)


# Nominal Model Evaluation

march.predict.nom16 <- predict(march.model.nom16, march.testDT16)
march.testDT16$p1 <- march.predict.nom16
win <- subset(march.testDT16, p1[,"1"] > p1[, "0"])
lose <- subset(march.testDT16, p1[,"0"] > p1[,"1"])
win$winner <- 1
lose$winner <- 0

march.testDT16 <- rbind(win, lose)

table(march.testDT16$winner, march.testDT16$dancing)

plot(march.model.nom16)
plotcp(march.model.nom16)
printcp(march.model.nom16)

# Pruning
march.model.nom16.pruned <- prune(march.model.nom32, cp = 0.1)
plot(march.model.nom16.pruned)

march.predict.nom16.pruned <- predict(march.model.nom16.pruned, march.testDT16)
march.testDT16$p2 <- march.predict.nom16.pruned
win <- subset(march.testDT16, p2[,"1"] > p2[, "0"])
lose <- subset(march.testDT16, p2[,"0"] > p2[,"1"])
win$winner.prune <- 1
lose$winner.prune <- 0

march.testDT16 <- rbind(win, lose)
march.testDT16$winner.prune <- as.factor(march.testDT16$winner.prune)

confusionMatrix(march.testDT16$winner.prune, march.testDT16$sweet16)
# BEFORE PRUNING:
## 83.75% accuracy, 86% true negative, 81.1% true positive
# AFTER PRUNING:
## 57.9% accuracy, 98.1% true negative, 35.5% true positive
## HUGE bias towards predicting 1 ( ~ 23.4% in sample vs ~ 64% in pred)

#############################################################################
# SVM Round of 32
colnames(march.train32)[2] <- "THREE_perst"

marchsvm <- svm(rtwo ~., data = march.train32[,c(1:13,15)])
plot(marchsvm, march.train32[,c(1:13,15)], AdjEMst~pt_margst)

# Predict SVM
colnames(march.test32)[2] <- "THREE_perst"

marchsvm_eval <- predict(marchsvm, march.test32)
march.test32$pred <- marchsvm_eval
confusionMatrix(march.test32$pred, march.test32$rtwo)

# ~ 75% accuracy
# ~ 76% true positive
# ~ 74% true negative


#############################################################################
# SVM Sweet Sixteen
colnames(march.train16)[2] <- "THREE_perst"

marchsvm16 <- svm(sweet16 ~., data = march.train16[,c(1:13,16)])
plot(marchsvm16, march.train16, AdjOst~AdjDst)

# Predict SVM
colnames(march.test16)[2] <- "THREE_perst"

marchsvm_eval16 <- predict(marchsvm16, march.test16)
march.test16$pred <- marchsvm_eval16
confusionMatrix(march.test16$pred, march.test16$sweet16)
# ~ 79% accuracy
# ~ 83% true positive
# ~ 60% true negative (small sample, probably not very good)
# overpicked class 0, but that is to be expected

##################################### Matchup Predictions ########################################

matchup <- read_excel("~/Downloads/March Madness/march_full_matchup_data16_23.xlsx")
firstround24 <- read_excel("~/Downloads/March Madness/2024 first round & first four.xlsx")
colnames(matchup)[1] <- "Team"
matchup_sub <- matchup[, c(3:10)]

corr <- cor(matchup_sub)
corrplot(corr, method = 'color')

################################## Matchup Predictions KNN ###########################################

matchup_sub$AdjEMdiff_st <- standardization(matchup_sub$AdjEMdiff)
matchup_sub$AdjOdiff_st <- standardization(matchup_sub$AdjOdiff)
matchup_sub$AdjDdiff_st <- standardization(matchup_sub$AdjDdiff)
matchup_sub$SeedDiff_st <- standardization(matchup_sub$SeedDiff)
matchup_sub$pmDIff_st <- standardization(matchup_sub$pmDIff)
matchup_sub$AdjSOSdiff_st <- standardization(matchup_sub$AdjSOSdiff)
matchup_sub$P6_diff_st <- standardization(matchup_sub$P6_diff)

firstround24$AdjEMdiff_st <- standardization(firstround24$AdjEMdiff)
firstround24$AdjOdiff_st <- standardization(firstround24$AdjOdiff)
firstround24$AdjDdiff_st <- standardization(firstround24$AdjDdiff)
firstround24$SeedDiff_st <- standardization(firstround24$SeedDiff)
firstround24$pmDIff_st <- standardization(firstround24$pmDIff)
firstround24$AdjSOSdiff_st <- standardization(firstround24$AdjSOSdiff)
firstround24$P6_diff_st <- standardization(firstround24$P6_diff)

matchup_std <- matchup_sub[,c(8:15)]
matchup_std$W <- as.factor(matchup_std$W)
firstround24_std <- firstround24[,c(10:16)]
view(firstround24_std)

trainSet.matchup <- createDataPartition(matchup_std$W, p = 0.7) [[1]]
march.train.matchup <- matchup_std[trainSet.matchup,]
march.test.matchup <-matchup_std[-trainSet.matchup,]
#march.train.matchup <- march.train.matchup[c(1:228),]

march.train.matchup.knn <- march.train.matchup[, c(2:8)]
march.test.matchup.knn <- march.test.matchup[, c(2:8)]
march.train.matchup.cl <- march.train.matchup$W
march.test.matchup.cl <- march.test.matchup$W

march.pred.matchup.knn1 <- knn(train = march.train.matchup.knn, 
                          test = march.test.matchup.knn,  
                          cl = march.train.matchup.cl,
                          k = 1)
march.pred.matchup.knn3 <- knn(train = march.train.matchup.knn, 
                          test = march.test.matchup.knn, 
                          cl = march.train.matchup.cl, 
                          k = 3)
march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                          test = march.test.matchup.knn, 
                          cl = march.train.matchup.cl, 
                          k = 5)

table(march.test.matchup.cl, march.pred.matchup.knn5)

pred24_knn1 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                             test = firstround24_std, 
                                             cl = march.train.matchup.cl, 
                                             k = 1)

pred24_knn9 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                              test = firstround24_std, 
                                              cl = march.train.matchup.cl, 
                                              k = 9)

pred24_knn5 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                              test = firstround24_std, 
                                              cl = march.train.matchup.cl, 
                                              k = 5)


firstround24$knn1_pred <- pred24_knn1
pred_mat <- firstround24[,c(1,2,3,17)]
pred_mat$knn9_pred <- pred24_knn9
pred_mat$knn5_pred <- pred24_knn5
view(pred_mat)

# With k = 9
# 76.8% accuracy, 77.9% true negative, 75.9% true positive
# Gets the proportions more or less accurately

################################## Matchup Predictions RIPPER ###########################################



matchup.model.rules <- JRip(W ~., data = march.train.matchup)
print(matchup.model.rules)
matchup.predict.rules <- predict(matchup.model.rules, march.test.matchup.knn)

# Rule Set Evaluation

matchup.eval.rules <- confusionMatrix(matchup.predict.rules, march.test.matchup.cl)
print(matchup.eval.rules)
# 71.6% accuracy, 69.47% true negative, 75% true positive, positive class = 0
# Reference ~ 53% class 1, prediction ~ 61.3%


#################################### Naive Bayes Classifier #########################################

matchup.nb.model <- naiveBayes(W ~., data = march.train.matchup)
print(matchup.nb.model)
str(matchup.nb.model)

summary(matchup.nb.model)

# Evaluating the Model

matchup.nb.pred <- predict(matchup.nb.model, march.test.matchup.knn)
eval_model(matchup.nb.pred, march.test.matchup.cl)
# 78.8% accuracy, 74.6% true positive, 83.6% true negative
# Gets the proportions more less accurately

matchup.nb.pred <- predict(matchup.nb.model, firstround24_std)
pred_mat$nb_pred <- matchup.nb.pred
view(pred_mat)

############################## Decision Tree ###############################################


matchup.model.nom <- rpart(W ~., data = march.train.matchup)
summary(matchup.model.nom)

# Nominal Model Evaluation

matchup.predict.nom <- predict(matchup.model.nom, march.test.matchup.knn)
march.test.matchup.knn$p1 <- matchup.predict.nom
win <- subset(march.test.matchup.knn, p1[,"1"] > p1[, "0"])
lose <- subset(march.test.matchup.knn, p1[,"0"] > p1[,"1"])
win$winner <- 1
lose$winner <- 0

march.test.matchup.DT <- rbind(win, lose)

table(march.test.matchup.DT$winner, march.test.matchup.cl)

plot(matchup.model.nom)
plotcp(matchup.model.nom)
printcp(matchup.model.nom)

# Pruning
matchup.model.nom.pruned <- prune(matchup.model.nom, cp = 0.01)
plot(matchup.model.nom.pruned)

matchup.predict.nom.pruned <- predict(matchup.model.nom.pruned, march.test.matchup.DT)
march.test.matchup.DT$p2 <- matchup.predict.nom.pruned
win <- subset(march.test.matchup.DT, p2[,"1"] > p2[, "0"])
lose <- subset(march.test.matchup.DT, p2[,"0"] > p2[,"1"])
win$winner.prune <- 1
lose$winner.prune <- 0

march.test.matchup.DT <- rbind(win, lose)

table(march.test.matchup.DT$winner.prune, march.test.matchup.cl)
#BAD

################################ SVM #############################################

march.test.matchup.knn <- march.test.matchup.knn[,c(1:7)]



# for(i in 1:8){
  #for(j in 1:10){
    #svm_pol <- svm(class ~., kernel = "polynomial", data = train_val, cost = c[i], degree= Degree[j], coef0 = 1)
    #svm_pol_eval = predict(svm_pol, testing[,c(1:4)])
    #pol_acc = confusionMatrix(svm_pol_eval, reference = testing$class)
    #print(c(Degree[j], c[i], pol_acc$overall[1]))
  #}
#}

c = 10^(-4:3)
g = 10^(-3:3)
d = c(1:10)

for(i in 1:8){
  for(j in 1:7){
  matchupsvm <- svm(W ~., data = march.train.matchup, type = "C", cost = c[i], gamma = g[j], coef0 = 1)
  svm_eval <- predict(matchupsvm, march.test.matchup.knn)
  svm_acc = confusionMatrix(svm_eval, reference = march.test.matchup.cl)
  print(c(g[j], c[i], svm_acc$overall[1]))
  }
}

matchupsvm <- svm(W ~., data = march.train.matchup, type = "C", cost = 1, gamma = 10, coef0 = 1)
svm_eval <- predict(matchupsvm, march.test.matchup.knn)
svm_acc = confusionMatrix(svm_eval, reference = march.test.matchup.cl)
print(svm_acc)
march.test.matchup$svm_eval <- svm_eval

matchupsvm <- svm(W ~., data = march.train.matchup, kernel = "polynomial", cost = 1, degree = 10, coef0 = 1)
svm_eval <- predict(matchupsvm, march.test.matchup.knn)
svm_acc = confusionMatrix(svm_eval, reference = march.test.matchup.cl)
print(svm_acc)
march.test.matchup$svm_eval <- svm_eval
# Predict SVM

matchupsvm_eval <- predict(matchupsvm, march.test.matchup.knn)
march.test.matchup.knn$pred <- matchupsvm_eval
table(march.test.matchup.knn$pred, march.test.matchup.cl)



matchupsvm <- svm(W ~., data = march.train.matchup, kernel = "radial", cost = 1, degree = 10, coef0 = 1)
svm_eval2 <- predict(matchupsvm, march.test.matchup.knn)
svm_acc2 = confusionMatrix(svm_eval2, reference = march.test.matchup.cl)
print(svm_acc2)
march.test.matchup$svm_eval <- svm_eval2

plot(matchupsvm, march.test.matchup, AdjEMdiff_st ~ AdjSOSdiff_st)

# 74.2% accuracy, 77.42% true positive, 72.04% true negative
# Reference data ~ 53% class 1, Prediction data ~ 63.2% class 1

pred_mat$svm_pred1 <- predict(matchupsvm, firstround24)
view(pred_mat)

################################ ANN ############################################# 

matchup_std$W <- as.factor(matchup_std$W)
view(matchup_std)

# Run if W factor levels are 1 and 2
#for (i in 1:nrow(matchup_std)) {
  #if(matchup_std[i,1] == 2){
   # matchup_std[i,1] <- 1}else
   #   {
  #  matchup_std[i,1] <- 0
#  }
#}
#view(matchup_std)

trainSet.matchup <- createDataPartition(matchup_std$W, p = 0.67) [[1]]
march.train.matchup <- matchup_std[trainSet.matchup,]
march.test.matchup <-matchup_std[-trainSet.matchup,]
#march.train.matchup <- march.train.matchup[c(1:228),]

march.train.matchup.knn <- march.train.matchup[, c(2:8)]
march.test.matchup.knn <- march.test.matchup[, c(2:8)]
march.train.matchup.cl <- march.train.matchup$W
march.test.matchup.cl <- march.test.matchup$W

# iris.cont.nn.formula <- as.formula(paste("species ~ ", 
                                         #paste(names(iris.cont.train[!names(iris.cont.train) %in% 'species']), 
                                              #collapse = " + "), sep=""))


matchup.nn.formula <- as.formula(paste("W ~ ", 
                                         paste(names(march.train.matchup[!names(march.train.matchup) %in% 'W']), 
                                               collapse = " + "), sep=""))


# Build NN model with default hidden layer (1 hidden layer with 1 node)

# iris.cont.model.nn1 <- neuralnet(iris.cont.nn.formula, data=iris.cont.train)

matchup.model.nn1 <- neuralnet(W ~., data=march.train.matchup)

# Plot the network

# plot(iris.cont.model.nn1)

plot(matchup.model.nn1)

# Build NN model with 2 hidden layers (3 and 4 nodes)
# Use backpropagation with 0.01 learning rate

#iris.cont.model.nn2 <- neuralnet(iris.cont.nn.formula, 
                                 #data=iris.cont.train, 
                                 #hidden=c(3,4), 
                                 #algorithm="backprop", 
                                 #learningrate=0.02)

matchup.model.nn2 <- neuralnet(W ~., data=march.train.matchup, 
                                 hidden=c(3,3),
                                 algorithm="rprop+",
                                  stepmax = 1000000,
                                  threshold = 0.005,
                                  learningrate.limit = c(0.00001,0.1))

 # Plot the network

#plot(iris.cont.model.nn2)

plot(matchup.model.nn2)

# Evaluate neural network model
#
# NOTE: predicting with the neural network model uses compute() not predict()
####
iris.cont.eval.nn1 <- compute(iris.cont.model.nn1, iris.cont.test)
iris.cont.eval.nn1.conMat <- confusionMatrix(
  as.factor(round(iris.cont.eval.nn1$net.result[,1])), 
  as.factor(iris.cont.test$species))
print(iris.cont.eval.nn1.conMat$table)
str(iris.cont.test)



matchup.eval.nn1 <- compute(matchup.model.nn1, march.test.matchup.knn)
matchup.eval.nn1.conMat <- confusionMatrix(
  as.factor(round(matchup.eval.nn1$net.result[,2])), 
  as.factor(march.test.matchup.cl))
print(matchup.eval.nn1.conMat$table)
# Accuracy: 78%, true positive: 83.3%, true negative: 72.2%


iris.cont.eval.nn2 <- compute(iris.cont.model.nn2, iris.cont.test)
iris.cont.eval.nn2.conMat <- confusionMatrix(
  factor(round(iris.cont.eval.nn2$net.result), levels=c("0", "1")), 
  as.factor(iris.cont.test$species))
print(iris.cont.eval.nn2.conMat$table)

str(march.test.matchup)
matchup.eval.nn2 <- compute(matchup.model.nn2, march.test.matchup.knn)
matchup.eval.nn2$net.result <- as.numeric(matchup.eval.nn2$net.result[,2])
matchup.eval.nn2.conMat <- confusionMatrix(
  factor(round(matchup.eval.nn2$net.result), levels=c("0", "1")), 
  as.factor(march.test.matchup.cl))
print(matchup.eval.nn2.conMat$table)
# 75.9% accuracy, 78.6% true negative, 75% true negative
# slight class imbalance favoring 1s

ann_pred <- compute(matchup.model.nn2, firstround24_std)
ann_pred$net.result <- 
  as.factor(round(as.numeric(ann_pred$net.result[,2])))
view(ann_pred$net.result)
pred_mat$ann_pred <- ann_pred$net.result
view(pred_mat)
view(ann_pred)

ann_r2 <- read_excel("~/Downloads/March Madness/ANN_r2.xlsx")
ann_r2$AdjEMdiff_st <- standardization(ann_r2$AdjEMdiff)
ann_r2$AdjOdiff_st <- standardization(ann_r2$AdjOdiff)
ann_r2$AdjDdiff_st <- standardization(ann_r2$AdjDdiff)
ann_r2$SeedDiff_st <- standardization(ann_r2$SeedDiff)
ann_r2$pmDIff_st <- standardization(ann_r2$pmDIff)
ann_r2$AdjSOSdiff_st <- standardization(ann_r2$AdjSOSdiff)
ann_r2$P6_diff_st <- standardization(ann_r2$P6_diff)

ann_pred <- compute(matchup.model.nn2, ann_r2[,c(10:16)])
ann_pred$net.result <- 
  as.factor(round(as.numeric(ann_pred$net.result[,2])))
ann_pred_r2 <- cbind(ann_r2[,c(1,2)],ann_pred$net.result)
view(ann_pred_r2)


############### TESTING ON guess_r2 GAMES ########################################
FirstRound23 <- read_excel("~/Downloads/March Madness/2023 FirstFour_FirstRound.xlsx")
guess_r2 <- read_excel("~/Downloads/March Madness/23Round2_pred.xlsx")

FirstRound23$AdjEMdiff_st <- standardization(FirstRound23$AdjEMdiff)
FirstRound23$AdjOdiff_st <- standardization(FirstRound23$AdjOdiff)
FirstRound23$AdjDdiff_st <- standardization(FirstRound23$AdjDdiff)
FirstRound23$SeedDiff_st <- standardization(FirstRound23$SeedDiff)
FirstRound23$pmDIff_st <- standardization(FirstRound23$pmDIff)
FirstRound23$AdjSOSdiff_st <- standardization(FirstRound23$AdjSOSdiff)
FirstRound23$P6_diff_st <- standardization(FirstRound23$P6_diff)

guess_r2$AdjEMdiff_st <- standardization(guess_r2$AdjEMdiff)
guess_r2$AdjOdiff_st <- standardization(guess_r2$AdjOdiff)
guess_r2$AdjDdiff_st <- standardization(guess_r2$AdjDdiff)
guess_r2$SeedDiff_st <- guess_r2$SeedDiff
guess_r2$pmDIff_st <- standardization(guess_r2$pmDIff)
guess_r2$AdjSOSdiff_st <- standardization(guess_r2$AdjSOSdiff)
guess_r2$P6_diff_st <- standardization(guess_r2$P6_diff)

guess_r2_sub2 <- guess_r2[,c(10:16)]
FirstRound23_sub2 <- FirstRound23[,c(10:16)]
matchup_std <- matchup_sub[,c(8:15)]
matchup_std$W <- as.factor(matchup_std$W)

trainSet.matchup <- createDataPartition(matchup_std$W, p = 0.6) [[1]]
march.train.matchup <- matchup_std[trainSet32,]
march.test.matchup <-matchup_std[-trainSet32,]
march.train.matchup <- march.train.matchup[c(1:228),]
march.test.guess_r2 <- guess_r2_sub2
march.test.FirstRound23 <- FirstRound23_sub2

march.train.matchup.knn <- march.train.matchup[, c(2:7)]
march.test.matchup.knn <- march.test.matchup[, c(2:8)]
march.train.matchup.cl <- march.train.matchup$W
march.test.matchup.cl <- march.test.matchup$W

march.pred.matchup.knn1 <- knn(train = march.train.matchup.knn, 
                               test = march.test.guess_r2,  
                               cl = march.train.matchup.cl,
                               k = 1)
march.pred.matchup.knn3 <- knn(train = march.train.matchup.knn, 
                               test = march.test.guess_r2, 
                               cl = march.train.matchup.cl, 
                               k = 3)
march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                               test = march.test.FirstRound23, 
                               cl = march.train.matchup.cl, 
                               k = 5)
march.pred.matchup.knn7 <- knn(train = march.train.matchup.knn, 
                               test = march.test.FirstRound23, 
                               cl = march.train.matchup.cl, 
                               k = 7)
march.pred.matchup.knn11 <- knn(train = march.train.matchup.knn, 
                               test = march.test.FirstRound23, 
                               cl = march.train.matchup.cl, 
                               k = 11)


FirstRound23_sub2$pred7 <- march.pred.matchup.knn7
FirstRound23_sub2$pred5 <- march.pred.matchup.knn5
FirstRound23_sub2$pred11 <- march.pred.matchup.knn11
view(guess_r2_sub2)



view(march.test.FirstRound23)
matchupsvm_eval <- predict(matchupsvm, march.test.FirstRound23)
FirstRound23_sub2$pred_svm <- matchupsvm_eval

r2svm_eval <- predict(matchupsvm, march.test.guess_r2)
guess_r2_sub2$pred_svm <- r2svm_eval



matchupsvm_eval2 <- predict(matchupsvm, march.test.FirstRound23)
FirstRound23_sub2$pred_svm2 <- matchupsvm_eval2
FirstRound23_sub2$Team <- FirstRound23$...1
FirstRound23_sub2$Opp <- FirstRound23$OPP1
view(FirstRound23_sub2)

matchup.nb.pred.23 <- predict(matchup.nb.model, march.test.guess_r2)
guess_r2_sub2$nb_pred <- matchup.nb.pred.23
guess_r2_sub2$Team <- guess_r2$...1
guess_r2_sub2$Opp <- guess_r2$OPP1
view(guess_r2_sub2)


##### Pred_mat #####

# CREATING FIRST ROUND PRED MAT

firstround24$knn1_pred <- pred24_knn1
pred_mat <- firstround24[,c(1,2,3,17)]
pred_mat$knn9_pred <- pred24_knn9
pred_mat$knn5_pred <- pred24_knn5
view(pred_mat)

matchup.nb.pred <- predict(matchup.nb.model, firstround24_std)
pred_mat$nb_pred <- matchup.nb.pred
view(pred_mat)

pred_mat$svm_pred1 <- predict(matchupsvm, firstround24)
view(pred_mat)

ann_pred <- compute(matchup.model.nn2, firstround24_std)
ann_pred$net.result <- 
  as.factor(round(as.numeric(ann_pred$net.result[,2])))
view(ann_pred$net.result)
pred_mat$ann_pred <- ann_pred$net.result
view(pred_mat)

pred_mat$DT_pred <- full_pred_mat2$V7

for (i in 1:nrow(pred_mat)) {
  if(pred_mat[i,9] == 0){
    pred_mat[i,9] <- 0.5
  } else{
    if(pred_mat[i,9] == -1){
      pred_mat[i,9] <- 0
    }else{
      pred_mat[i,9] <- pred_mat[i,9]
    }
  }
}
view(pred_mat)

pred_mat$knn1_pred <- as.numeric(pred_mat$knn1_pred)
pred_mat$knn9_pred <- as.numeric(pred_mat$knn9_pred)
pred_mat$knn5_pred <- as.numeric(pred_mat$knn5_pred)
pred_mat$nb_pred<- as.numeric(pred_mat$nb_pred)
pred_mat$svm_pred1 <- as.numeric(pred_mat$svm_pred1)
pred_mat$ann_pred <- as.numeric(pred_mat$ann_pred)
pred_mat$DT_pred <- as.numeric(pred_mat$DT_pred)


for (i in 1:nrow(pred_mat)) {
  for(j in 3:8)
  if(pred_mat[i,j] == 2){
  pred_mat[i,j] <- 1}else
   {
  pred_mat[i,j] <- 0
  }
}
view(pred_mat)

for (i in 1:nrow(pred_mat)) {
  pred_mat[i,10] <- 1/12 * pred_mat[i,3] + 1/12 * pred_mat[i,4] +
    1/12 * pred_mat[i,5] + 1/6*pred_mat[i,6] + 1/6*pred_mat[i,7] +
    1/4*pred_mat[i,8] + 1/6*pred_mat[i,9]
}
colnames(pred_mat)[10] <- "dan_pred"
view(pred_mat$dan_pred)

# LOADING R2 PREDICTION DATA
r2_2024 <- read_excel("~/Downloads/March Madness/2024_r2_pred.xlsx")

r2_2024$AdjEMdiff_st <- standardization(r2_2024$AdjEMdiff)
r2_2024$AdjOdiff_st <- standardization(r2_2024$AdjOdiff)
r2_2024$AdjDdiff_st <- standardization(r2_2024$AdjDdiff)
r2_2024$SeedDiff_st <- standardization(r2_2024$SeedDiff)
r2_2024$pmDIff_st <- standardization(r2_2024$pmDIff)
r2_2024$AdjSOSdiff_st <- standardization(r2_2024$AdjSOSdiff)
r2_2024$P6_diff_st <- standardization(r2_2024$P6_diff)

r2_24_std <- r2_2024[,c(10:16)]


# CREATING PRED MAT FOR R2

# Use 100% training data for knn, ~ 70% for others.
pred24_knn1 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                              test = r2_24_std, 
                                              cl = march.train.matchup.cl, 
                                              k = 1)

pred24_knn9 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                              test = r2_24_std, 
                                              cl = march.train.matchup.cl, 
                                              k = 9)

pred24_knn5 <- march.pred.matchup.knn5 <- knn(train = march.train.matchup.knn, 
                                              test = r2_24_std, 
                                              cl = march.train.matchup.cl, 
                                              k = 5)

r2_2024$knn1_pred <- pred24_knn1
pred_mat <- r2_2024[,c(1,2,3,17)]
pred_mat$knn9_pred <- pred24_knn9
pred_mat$knn5_pred <- pred24_knn5
view(pred_mat)

matchup.nb.pred <- predict(matchup.nb.model, r2_24_std)
pred_mat$nb_pred <- matchup.nb.pred
view(pred_mat)

pred_mat$svm_pred1 <- predict(matchupsvm, r2_24_std)
view(pred_mat)

ann_pred <- compute(matchup.model.nn2, r2_24_std)
ann_pred$net.result <- 
  as.factor(round(as.numeric(ann_pred$net.result[,2])))
view(ann_pred$net.result)
pred_mat$ann_pred <- ann_pred$net.result
view(pred_mat)




pred_mat$knn1_pred <- as.numeric(pred_mat$knn1_pred)
pred_mat$knn9_pred <- as.numeric(pred_mat$knn9_pred)
pred_mat$knn5_pred <- as.numeric(pred_mat$knn5_pred)
pred_mat$nb_pred <- as.numeric(pred_mat$nb_pred)
pred_mat$svm_pred1 <- as.numeric(pred_mat$svm_pred1)
pred_mat$ann_pred <- as.numeric(pred_mat$ann_pred)
view(pred_mat)
pred_mat <- pred_mat[-3]

for (i in 1:nrow(pred_mat)) {
  for(j in 3:8)
    if(pred_mat[i,j] == 2){
      pred_mat[i,j] <- 1}else
      {
        pred_mat[i,j] <- 0
      }
}
view(pred_mat)

for (i in 1:nrow(pred_mat)) {
  pred_mat[i,9] <- 1/12 * pred_mat[i,3] + 1/12 * pred_mat[i,4] +
    1/12 * pred_mat[i,5] + 5/24*pred_mat[i,6] + 5/24*pred_mat[i,7] +
    1/3*pred_mat[i,8]
}
colnames(pred_mat)[9] <- "dan_pred"
view(pred_mat)



