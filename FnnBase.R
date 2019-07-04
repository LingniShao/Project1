## This is the model for random search

myflags <- flags(
  flag_integer('units1', 32),
  flag_integer('units2', 32)
)

# build the feedforward nerual network in a function
build_model <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = myflags$units1, activation = 'relu',
                input_shape = dim(features_train)[2]) %>%
    layer_dense(units = myflags$units2, activation = 'relu') %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = 'adam',
    loss = "mse",
    metrics = "mse"
  )
  model
}

# get one topology of the fnn
model <- build_model() 

# fit model
history <- model %>% fit( 
  features_train, targets_train,
  epochs = 10, 
  batch_size = 256, 
  verbose = 0,  # learning scilently without printing anything
  validation_split = 0.2  # split out 20% training data for validation
)





