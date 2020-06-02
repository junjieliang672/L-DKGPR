rm(list=ls())
library(PGEE)
library(Metrics)
library(rmatio)
##loc <- 'F:/onedrive/pythonCaseStudy/Nonparametric Deep Kernel Learning for Longitudinal Data Analysis/' ####
loc <- '/Users/jul672/Desktop/OneDrive/pythonCaseStudy/Nonparametric Deep Kernel Learning for Longitudinal Data Analysis/'
##loc <- 'C:/Users/jokit/OneDrive/pythonCaseStudy/Nonparametric Deep Kernel Learning for Longitudinal Data Analysis/'
file <- 'tadpole_0.mat'
filepath <- paste0(loc,file)
######## read and format data ##############
data = read.mat(filepath)
trainX <- data[['trainX']]
testX <- data[['testX']]
trainY <- data[['trainY']]
testY <- data[['testY']]
corr <- data[['corr']]
trainIid <- data[['trainId']]
trainOid <- data[['trainOid']]
testIid <- data[['testId']]
testOid <- data[['testOid']]
allX <- data[['data']]
allIid <- data[['iid']]
allOid <- data[['oid']]
allY <- data[['y']]
kmeans <- data[['kmeans']]

train <- as.data.frame(trainX)
train$iid <- trainIid
train$oid <- trainOid
train$label <- trainY
test <- as.data.frame(testX)
test$iid <- testIid
test$oid <- testOid
test$label <- testY
allData <- as.data.frame(allX)
allData$kmeans <- kmeans
allData$iid <- allIid
allData$oid <- allOid
allData$label <- allY

r_squared <- function(actual,pred){
  square_error <- sum((actual - pred)^2)
  m <- mean(actual)
  var <- sum((actual - m)^2)
  1 - square_error/var
}

lambda <- .01 # 0.2  # this is chosen by the cv
formula <- "label~.-iid+0"
family <- gaussian(link = "identity")
# family <- binomial()

# analyze the data through penalized generalized estimating equations
s = Sys.time()
myfit1 <- PGEE(formula = formula, id = iid, data = train, na.action = NULL, 
               family = family, corstr = "AR-1", scale.fix = F, beta_int = rep(0,ncol(train)-2),
               lambda = lambda, eps = 10^-3, maxiter = 200, 
               tol = 10^-3, silent = F)
e = Sys.time()
print(e-s)

# compute the prediction on the test data -> gaussian case
testFeat = test[,names(myfit1$coefficients)]
pred <- apply(testFeat,1,function(r){
  r %*% myfit1$coefficients
})
loss <- r_squared(test$label,pred)
print(paste('pgee r2 loss:',round(loss,3)))
