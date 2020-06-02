rm(list=ls())
library(rmatio)
require(rstan)
require(lgpr)
require(repr)
require(ggplot2)
#onedriveLoc <- 'F:/onedrive/'
onedriveLoc <- '/Users/jul672/Desktop/OneDrive/'
loc <- paste0(onedriveLoc,'pythonCaseStudy/Nonparametric Deep Kernel Learning for Longitudinal Data Analysis/')
file <- 'tadpoleLGPR_0.mat'
filepath <- paste0(loc,file)
######## read and format data ##############
data = read.mat(filepath)
trainX <- data[['trainX']]
testX <- data[['testX']]
trainY <- data[['trainY']]
testY <- data[['testY']]
corr <- data[['corr']]
trainIid <- data[['trainId']]+1
trainOid <- data[['trainOid']]
testIid <- data[['testId']]+1
testOid <- data[['testOid']]
allIid <- c(trainIid,testIid)
allOid <- c(trainOid,testOid)

reId <- function(a){
  a <- as.character(a)
  dic <- list()
  ct <- 1
  for(each in a){
    v <- dic[[each]]
    if(is.null(v)){
      dic[[each]] <- ct
      ct <- ct + 1
    }
  }
  dic
}
l <- reId(allIid)
ind <- 0
for(i in as.character(trainIid)){
  trainIid[ind] <- l[[i]]
  ind <- ind + 1
}
ind <- 0
for(i in as.character(testIid)){
  testIid[ind] <- l[[i]]
  ind <- ind + 1
}

train <- data.frame(id=factor(trainIid),age=as.numeric(trainOid))
train <- cbind(train,as.data.frame(trainX))
train$y <- as.numeric(trainY)
test <- data.frame(id=factor(testIid),age=as.numeric(testOid))
test <- cbind(test,as.data.frame(testX))
# test$y <- as.numeric(testY)
# X_data <- data.frame(id=factor(allIid),age=as.numeric(allOid))
# X_data <- cbind(X_data,as.data.frame(allX))


####### define helper functions #############
r_squared <- function(actual,pred){
  square_error <- sum((actual - pred)^2)
  m <- mean(actual)
  var <- sum((actual - m)^2)
  1 - square_error/var
}

fixedEffects <- names(train)[1:(ncol(train)-1)]
fixed_formula <- paste0(fixedEffects,collapse = '+')
formula <- paste0('y~',fixed_formula)
f <- as.formula(formula)
s = Sys.time()
fit <- lgp(formula  = f,
           data     = train,
           equal_effect = FALSE,
           iter     = 2000, 
           chains   = 5,
           refresh  = 200,
           verbose  = T,
           parallel     = TRUE)
e = Sys.time()
print(e - s)

PRED   <- lgp_predict(fit, test, samples = 'map')
pred <- PRED$LIST[[1]]$mu_f
r_squared(testY,pred)

# curLoc <- paste0(loc,'R/lgpr/R/')
# setwd(curLoc)
# fs <- list.files(curLoc)
# for(fp in paste0(curLoc,fs)){
#   source(fp)
# }
# 
# samples <- 'map'
# PP <- predict_preproc(fit,X_data,samples)
# info    <- PP$info
# D       <- PP$D
# cnames  <- PP$cnames
# LIST    <- list()
# TSCL    <- fit@model@scalings$TSCL
# params    <- hyperparam_estimate(fit, samples)
# nam   <- names(params)
# alpha <- params[which(grepl("alpha_", nam))]
# ell   <- params[which(grepl("ell_", nam))]
# if(D[3]){
#   stp <- params[which(grepl("warp_steepness", nam))]
#   if(info$HMGNS==0){
#     beta <- params[which(grepl("beta", nam))]
#   }else{
#     beta <- NULL
#   }
#   if(info$UNCRT==1){
#     t_ons <- params[which(grepl("T_effect", nam))]
#     t_ons <- rbind(info$case_ids, t_ons)
#     rownames(t_ons) <- c("case id", "effect time")
#   }else{
#     t_ons <- NULL
#   }
# }else{
#   stp  <- NULL
#   beta <- NULL
#   t_ons <- NULL
# }
# sigma_n <- params[which(grepl("sigma_n", nam))]
# KERNEL_INFO <- list(D       = D, 
#                     alpha   = alpha, 
#                     ell     = ell, 
#                     stp     = stp, 
#                     beta    = beta,
#                     t_ons   = t_ons,
#                     info    = info,
#                     TSCL    = TSCL,
#                     NCAT    = info$NCAT)
# kk <- compute_kernel_matrices(X_data,X_data,KERNEL_INFO)
# k_sum <- apply(kk, c(1,2), sum)
# rc1 <- (k_sum + sigma_n^2*diag(nrow(X_data)) + info$DELTA*diag(nrow(X_data)))[nrow(X_data):1,]
# # rc1 <- rc1 ^ (1/5)
# write.mat(list('corr'=as.matrix(rc1)),paste0(loc,'synthetic_corr/','lgpr_',file))
# image(t(rc1))

cols <- strsplit('id, age, V5, V6, V7, V8, V9, V10, V11, V1, V2, V3, V4',split=',')[[1]]
cols <- gsub(' ','',cols, fixed=T)
tmp <- test[cols]
tmp$id <- as.numeric(tmp$id)
PRED   <- lgp_predict(fit, tmp, samples = 'map')
pred <- PRED$LIST[[1]]$mu_f
r_squared(testY,pred)
