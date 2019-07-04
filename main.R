library(data.table)  # package for load files
library(ggplot2)  # package for nice plots
library(gridExtra)  # package for arranging ggplots on same page
library(corrplot)  # package for visualizing correlation coefficients matrix
library(keras)  # package for deep learning, version 1.12.0 is recommendated
library(tfruns) # package for hyperparameter tuning



## _____________________________________________________________________________
## _____________________________________________________________________________
## prepocessing the data

# set the path  of the file containing our data
setwd("~/Documents/UoE/1_courses/dissertation/subproject01/stage02/NearestPoints")
# load all csv files
files = list.files(pattern="*.csv")
# combine data in all files into one dataframe
rawdata <- do.call(rbind, lapply(files, fread))
# extract features
mydata <- rawdata[,c('Coh_Swath', 'Coh_SwathOverPoca', 'DayInYear_Swath', 
                    'Dist_SwathToPoca', 'Heading_Swath', 'LeadEdgeS_Poca',
                    'LeadEdgeW_Poca', 'PhaseConfidence_Swath', 
                    'PhaseSSegment_Swath', 'Phase_Swath', 'Phase_SwathOverPoca', 
                    'PowerScaled_Swath', 'PowerScaled_SwathOverPoca', 
                    'PowerWatt_Swath', 'SampleNb_Swath',
                    'SampleNb_SwathMinusLeadEdgeS')]
# extract targets
mydata$diffO_S <- rawdata$Elev_Oib - rawdata$Elev_Swath

# set the path of the script file
setwd("~/Documents/UoE/1_courses/dissertation/subproject01/stage02")

## _____________________________________________________________________________
## _____________________________________________________________________________
## exploratory data analysis

# check if there exist missing data
anyNA(mydata)
# get the summary information of the data
summary(mydata)
sd(mydata$diffO_S)
# plot the histogram of Elev_Oib - Elev_Swath
ggplot(mydata) + 
  geom_histogram(aes(x=diffO_S, y=..density..), binwidth = 3, 
                 color='dodgerblue3', 
                 fill='dodgerblue') + 
  geom_density(aes(x=diffO_S,y=..density..), col ='dodgerblue3') +
  xlab('Elev_Oib - Elev_Swath')
# plot the scatter plots for all explanatory variables against response
splots <- list()
for (name in colnames(mydata[,-17])){
  splots[[name]] <- ggplot(mydata, aes_string(x=name, y='diffO_S')) + 
    geom_point(col='gold', shape=1) +
    geom_smooth(method=lm , color="gold3", se=FALSE)
}
grid.arrange(grobs=splots, nrow=4, ncol=4)
# plot correlation matrix
corr <- cor(mydata[,-17])
corrplot(corr, type="upper", order="hclust", col=c("black", "white"),
         bg="lightblue", tl.col="black", tl.cex=0.6)

## _____________________________________________________________________________
## _____________________________________________________________________________
## fit simple linear regression

# split data for training and test
set.seed(1)
ind <- sample(nrow(mydata), 200000)
traindata <- mydata[ind, ]
testdata <- mydata[-ind, ]
## _____________________________________________________________________________

# fitting step-wise linear regression
lr <- step(lm(diffO_S~., data=traindata),direction="both")
# get the results of the selected linear model
summary(lr)

# residual analysis
par(mfrow=c(1,3))
plot(residuals(lr),col='dodgerblue')  # plot residuals v.s. index
# plot residuals v.s. values and qq plot
plot(lr, which=c(1,2), col='dodgerblue', caption='')
dev.off()

# calculate rmse of test data
preds_lr <- predict(lr, testdata)
rmse_lr <- sqrt(mean((preds_lr - testdata$diffO_S)^2))
rmse_lr

## _____________________________________________________________________________
## _____________________________________________________________________________
# perform feedforward neural networks

# normalizing features to speed up the convergence of neural network
mean_train <- apply(traindata[, -17], 2, mean)
sd_train <-  apply(traindata[, -17], 2, sd)
features_train <- scale(traindata[, -17], center=mean_train, scale=sd_train)
features_test <- scale(testdata[, -17], center=mean_train, scale=sd_train)
targets_train <- traindata$diffO_S
targets_test <- testdata$diffO_S
## _____________________________________________________________________________
# build a baseline model which overfit

# random search the # hidden neurons of baseline model
runs_base <- tuning_run("FnnBase.R",   # the model to be tuned
                        runs_dir = "base_tuning",   # the disc to store runs
                        flags = list(
                          units1 = c(32, 64, 128, 256, 512),
                          units2 = c(32, 64, 128, 256, 512))
                        )
# view the results of random search by the order of validation mse
View(ls_runs(order = metric_mean_squared_error, runs_dir = "base_tuning", 
             decreasing = F))

# build model topology
build_modelBase <- function(u1,u2) {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = u1, activation = "relu",
                input_shape = dim(features_train)[2]) %>%
    layer_dense(units = u2, activation = "relu") %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'adam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# get one (256,256) models
modelBase1 <- build_modelBase(256,256)
# get one (256,512) models
modelBase2 <- build_modelBase(256,512)
# get one (512,256) models
modelBase3 <- build_modelBase(512,256)
# get one (512,512) models
modelBase4 <- build_modelBase(512,512)

# fit model and store training stats
historyBase1 <- modelBase1 %>% fit( 
  features_train, targets_train,
  epochs = 50, 
  batch_size = 256, 
  verbose = 0,
  validation_split = 0.2
)
historyBase2 <- modelBase2 %>% fit( 
  features_train, targets_train,
  epochs = 50, 
  batch_size = 256, 
  verbose = 0,
  validation_split = 0.2
)
historyBase3 <- modelBase3 %>% fit( 
  features_train, targets_train,
  epochs = 50, 
  batch_size = 256, 
  verbose = 0,
  validation_split = 0.2
)
historyBase4 <- modelBase4 %>% fit( 
  features_train, targets_train,
  epochs = 50, 
  batch_size = 256, 
  verbose = 0,
  validation_split = 0.2
)
# plot training mse and test mse
plot(historyBase1, metrics = 'loss')
plot(historyBase2, metrics = 'loss')
plot(historyBase3, metrics = 'loss')
plot(historyBase4, metrics = 'loss')

## _____________________________________________________________________________

# random search the hyperparameter configuration
runs <- tuning_run("FnnFinal.R", sample = 1e-3, runs_dir = "final_tuning",
                   flags = list(units1 = c(256,512), units2 = c(256,512),
                                dropout1 = seq(0, 0.9, 0.1),
                                dropout2 = seq(0, 0.9, 0.1),
                                batch_size = c(16, 64, 128, 256),
                                activation = c('relu','softmax','elu','selu',
                                               'softsign','tanh'),
                                kernel_initializer = c('lecun_uniform',
                                                       'glorot_uniform',
                                                       'he_uniform'),
                                optimizer = c('rmsprop','Adagrad','Adadelta',
                                              'Adam','Adamax','Nadam')
))



# list the results of random search by the order of validation mse
View(ls_runs(order = metric_val_mean_squared_error, runs_dir = "final_tuning", 
             decreasing = F))

# view the 'optimal' model we found
view_run('final_tuning/2019-07-03T17-19-54Z')

## _____________________________________________________________________________
# train full model

# build the 'optimal' nerual network
build_model <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform',
                input_shape = dim(features_train)[2]) %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'Nadam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# Set early_stop crteria
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      min_delta = 1e-2,
                                      patience = 5,
                                      mode = 'min')

# Tune epochs using early stopping
model <- build_model()
history <- model %>% fit( 
  features_train, targets_train,
  epochs = 300, 
  batch_size = 256, 
  verbose = 1,
  validation_split = 0.2,
  callbacks = early_stop
)
# result: stops at 109 epoch
plot(history, metrics = 'loss')  # plot results

# train on all data
model_all <- build_model()
history_all <- model_all %>% fit( 
  features_train, targets_train,
  epochs = 109, 
  batch_size = 256, 
  verbose = 1
)
model_all %>% save_model_hdf5("my_model_all.h5")  # save model
model_all %>% evaluate(features_test, targets_test,verbose = 0)  # get test loss

## _____________________________________________________________________________
# train reduced model

# delete PowerWatt_Smith and  and SampleNb_SwathMinusLeadEdgeS
rd_features_train <- features_train[,c(-14,-16)]
rd_features_test <- features_test[,c(-14,-16)]

# build the 'optimal' nerual network
build_model_rd <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform',
                input_shape = dim(rd_features_train)[2]) %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'Nadam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# Set early_stop crteria
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      min_delta = 1e-2,
                                      patience = 5,
                                      mode = 'min')

# tune epochs using early stopping
model_rd <- build_model_rd()
history_rd <- model_rd %>% fit( 
  rd_features_train, targets_train,
  epochs = 300, 
  batch_size = 256, 
  verbose = 1,
  validation_split = 0.2,
  callbacks = early_stop
)
# result: stops at 89 epoch
plot(history_rd, metrics = 'loss')  # plot results

# train on all data
model_rd_all <- build_model_rd()
history_rd_all <- model_rd_all %>% fit( 
  rd_features_train, targets_train,
  epochs = 89, 
  batch_size = 256, 
  verbose = 1
)

model_rd_all %>% save_model_hdf5("my_model_rd_all.h5")  # save model
model_rd_all %>% evaluate(rd_features_test, targets_test,verbose = 0)  # get test loss
## _____________________________________________________________________________
# train reduced model

# delete SampleNb_SwathMinusLeadEdgeS
rd1_features_train <- features_train[,-16]
rd1_features_test <- features_test[,-16]

# build the 'optimal' nerual network
build_model_rd1 <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform',
                input_shape = dim(rd1_features_train)[2]) %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'Nadam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# Set early_stop crteria
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      min_delta = 1e-2,
                                      patience = 5,
                                      mode = 'min')

# tune epochs using early stop
model_rd1 <- build_model_rd1()
history_rd1 <- model_rd1 %>% fit( 
  rd1_features_train, targets_train,
  epochs = 300, 
  batch_size = 256, 
  verbose = 1,
  validation_split = 0.2,
  callbacks = early_stop
)
# result: stops at 128 epoch
plot(history_rd1, metrics = 'loss')  # plot results

# train on all data
model_rd1_all <- build_model_rd1()
history_rd1_all <- model_rd1_all %>% fit( 
  rd1_features_train, targets_train,
  epochs = 128, 
  batch_size = 256, 
  verbose = 1
)

model_rd1_all %>% save_model_hdf5("my_model_rd1_all.h5")  # save model
model_rd1_all %>% evaluate(rd1_features_test, targets_test,verbose = 0)  # get test loss
## _____________________________________________________________________________
# train reduced model

# delete PowerWatt_Smith
rd2_features_train <- features_train[,-14]
rd2_features_test <- features_test[,-14]

# build the 'optimal' nerual network
build_model_rd2 <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform',
                input_shape = dim(rd2_features_train)[2]) %>%
    layer_dense(units = 512, activation = "tanh",
                kernel_initializer = 'lecun_uniform') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'Nadam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# Set early_stop crteria
early_stop <- callback_early_stopping(monitor = "val_loss",
                                      min_delta = 1e-2,
                                      patience = 5,
                                      mode = 'min')

# tune epochs using early stopping
model_rd2 <- build_model_rd2()
history_rd2 <- model_rd2 %>% fit( 
  rd2_features_train, targets_train,
  epochs = 300, 
  batch_size = 256, 
  verbose = 1,
  validation_split = 0.2,
  callbacks = early_stop
)
# result: stops at 94 epoch
plot(history_rd2, metrics = 'loss')  # plot results

# train on all data
model_rd2_all <- build_model_rd2()
history_rd2_all <- model_rd2_all %>% fit( 
  rd2_features_train, targets_train,
  epochs = 94, 
  batch_size = 256, 
  verbose = 1
)

model_rd2_all %>% save_model_hdf5("my_model_rd2_all.h5")  # save model
model_rd2_all %>% evaluate(rd2_features_test, targets_test,verbose = 0)  # get test loss
