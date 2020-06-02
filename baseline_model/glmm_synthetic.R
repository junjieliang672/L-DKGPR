rm(list=ls())
library(rmatio)
library(lme4)
library(nloptr)
# onedriveLoc <- 'F:/onedrive/'
onedriveLoc <- 'C:/Users/jokit/OneDrive/'
# onedriveLoc <- '/Users/jul672/Desktop/OneDrive/'
loc <- paste0(onedriveLoc,'pythonCaseStudy/Nonparametric Deep Kernel Learning for Longitudinal Data Analysis/simulation/')
file_base <- 'cluster5'
seed <- 5
file <- paste0(file_base,'_',seed,'.mat')
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


####### define helper functions #############
r_squared <- function(actual,pred){
  square_error <- sum((actual - pred)^2)
  m <- mean(actual)
  var <- sum((actual - m)^2)
  1 - square_error/var
}

nlopt <- function(par, fn, lower, upper, control) {
  .nloptr <<- res <- nloptr(par, fn, lb = lower, ub = upper, 
                            opts = list(algorithm = "NLOPT_LN_BOBYQA", print_level = 1,
                                        maxeval = 200, xtol_abs = 1e-2, ftol_abs = 1e-2))
  list(par = res$solution,
       fval = res$objective,
       conv = if (res$status > 0) 0 else res$status,
       message = res$message
  )
}

rescov <- function(model, data) {
  var.d <- crossprod(getME(model,"Lambdat"))
  Zt <- getME(model,"Zt")
  vr <- sigma(model)^2
  var.b <- vr*(t(Zt) %*% var.d %*% Zt)
  sI <- vr * Diagonal(nrow(data))
  var.y <- var.b + sI
  invisible(var.y)
}

fixedEffects <- names(train)[1:(ncol(train)-4)]
featuresWithRandomEffects <- fixedEffects[1:length(fixedEffects)]
random_formula <- paste0('(0+',featuresWithRandomEffects,'|iid/oid)',collapse = '+')
random_formula <- paste0(random_formula,'+',paste0('(0+',featuresWithRandomEffects,'|oid)',collapse = '+'))
fixed_formula <- paste0(fixedEffects,collapse = '+')
formula <- paste0('label~',random_formula,' + ',fixed_formula ,'+0')


s=Sys.time()
myfit <- lmer(formula = formula, data = train,control=lmerControl(check.nobs.vs.nlev = "ignore",
                                                                  check.nobs.vs.rankZ = "ignore",
                                                                  check.nobs.vs.nRE="ignore",
                                                                  calc.derivs = FALSE,
                                                                  optimizer = "nloptwrap"),verbose = T)

e=Sys.time()
print(e-s)

pred <- predict(myfit,test,allow.new.levels = T)
r2 <- r_squared(test$label,pred)
print(paste0('r2 score: ',round(r2,3)))

#myfit <- lmer(formula = formula, data = allData,control=lmerControl(check.nobs.vs.nlev = "ignore",
#                                                                  check.nobs.vs.rankZ = "ignore",
#                                                                  check.nobs.vs.nRE="ignore",
#                                                                  calc.derivs = FALSE,
#                                                                  optimizer = "nloptwrap"))
#rc1 <- rescov(myfit, allData)
#write.mat(list('corr'=as.matrix(rc1)),paste0(loc,'../synthetic_corr/','glmm_',file))
#image(rc1)

