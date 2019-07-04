## This is the model for random search

# optimize hyperparameters
myflags <- flags(
  flag_integer('units1', 256),
  flag_integer('units2', 256),
  flag_numeric('dropout1', 0.5),
  flag_numeric('dropout2', 0.5),
  flag_integer('batch_size', 32),
  flag_string('activation', 'relu'),
  flag_string('kernel_initializer','lecun_uniform'),
  flag_string('optimizer', 'rmsprop')
)

# build the feedforward nerual network in a function
build_model <- function() {
  
  # set up the architectures of neural network
  model <- keras_model_sequential() %>%
    layer_dense(units = myflags$units1, activation = myflags$activation,
                kernel_initializer = myflags$kernel_initializer,
                input_shape = dim(features_train)[2]) %>%
    layer_dropout(rate = myflags$dropout1) %>%
    layer_dense(units = myflags$units2, activation = myflags$activation,
                kernel_initializer = myflags$kernel_initializer) %>%
    layer_dropout(rate = myflags$dropout2) %>%
    layer_dense(units = 1) 
  # compile model
  model %>% compile(
    optimizer = myflags$optimizer,
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
  epochs = 5, 
  batch_size = myflags$batch_size, 
  verbose = 0,  # learning scilently without printing anything
  validation_split = 0.2  # split out 20% training data for validation
)





