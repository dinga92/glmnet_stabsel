library(glmnet)
library(caret)


run_cv <- function(data.X, data.Y, n_folds, alpha, family) {
  predictions <- list()
  labels <- list()
  folds <- createFolds(data.Y, n_folds, list = FALSE)

  for (fold in 1:n_folds) {
    # print(fold)
    train.data.X <- as.matrix(data.X[folds != fold,])
    train.data.Y <- as.matrix(data.Y[folds != fold])
    test.data.X <- as.matrix(data.X[folds == fold,])
    test.data.Y <- as.matrix(data.Y[folds == fold])

    # impute missing data without leaking information from test set
    # to keep the independence of train and test set
    # preProc <- preProcess(method = "knnImpute", train.data.X)
    # train.data.X <- predict(preProc, train.data.X)
    # test.data.X <- predict(preProc, test.data.X)

    # find the optimal lambda using cross validation and fit the model
    fit <- cv.glmnet(x = train.data.X,
                     y = train.data.Y,
                     family = family,
                     type.measure = "mse",
                     parallel = T,
                     alpha = alpha)
    # get predictions for the test set
    pred <- predict(fit, test.data.X, s = "lambda.min",
                                  type = 'response')
    # add additional info to predictions
    pred.names <- colnames(pred)
    pred <- as.data.frame(pred)
    names(pred) <- pred.names
    # pred$Y <- test.data.Y
    # pred$fold <- fold
    # save the predictions, for later use
    predictions[[fold]] <- pred
    labels[[fold]] <- test.data.Y
  }
  return(list(predictions, labels))
}

run_repeated_cv <-  function(data.X, data.Y, n_folds, repeats, alpha, family) {
  predictions <- list()
  labels <- list()
  for (rep in 1:repeats) {
    results <- run_cv(data.X, data.Y, n_folds, alpha, family)
    predictions[[rep]] <- results[[1]]
    labels[[rep]] <- results[[2]]
  }
  predictions <- unlist(predictions, recursive = FALSE)
  labels <- unlist(labels, recursive = FALSE)
  return(list(predictions, labels))
}

results_to_measures_binomial <- function(pred_scores, pred_labels){
  library(ROCR)
  scores <- list()
  labels <- list()
  # if they are columns, make them vectors
  if (!is.null(dim(pred_scores[[1]]))) {
    for (i in 1:length(pred_scores)) {
      scores[[i]] <- pred_scores[[i]][,1]
    }
  } else {
    scores <- pred_scores
  }
  if (!is.null(dim(pred_labels[[1]]))) {
    for (i in 1:length(pred_scores)) {
      labels[[i]] <- pred_labels[[i]][,1]
    }
  } else {
    labels <- pred_labels
  }

  # make target values 0-1
  labels <- lapply(labels, function(x) {x - min(x)})
  pred <- prediction(scores, labels, label.ordering = c('0','1'))
  perf_auc <- performance(pred, 'auc')
  mean_auc <- mean(unlist(perf_auc@y.values))
  sd_auc <- sd(unlist(perf_auc@y.values))
  threshold <-  mean(unlist(labels))
  thresholded_scores <- ifelse(unlist(scores) < threshold, 0, 1)
  thresholded_pred <- prediction(thresholded_scores, unlist(labels))
  mean_sens <- performance(thresholded_pred, 'sens')@y.values[[1]][2]
  mean_spec <- performance(thresholded_pred, 'spec')@y.values[[1]][2]
  mean_balanced_acc <- (mean_sens + mean_spec) / 2
  c('mean_auc' = mean_auc, 'sd_auc' = sd_auc, 'mean_sens' = mean_sens,
    'mean_spec' = mean_spec, 'mean_balanced_acc' = mean_balanced_acc)
}

results_to_measures <- function(pred_scores, pred_labels){
  n_groups <- ncol(pred_scores[[1]])
  if (n_groups == 1) {
    results <- results_to_measures_binomial(pred_scores, pred_labels)
  } else {
    names_groups <- names(pred_scores[[1]])
    results <- list()
    for (target_group in 1:n_groups) {
      scores <- lapply(pred_scores, function(x){return(x[,target_group])})
      labels <- lapply(pred_labels, function(x){return(ifelse(x == target_group, 1, 0))})
      results[[target_group]] <- results_to_measures_binomial(scores, labels)
    }
    names(results) <- names_groups
  }
  return(results)
}

p_val <- function(null, measured) {
  1 - (length(null[null <= measured])/(length(null) + 1))
}


# allow multicore processing
library(doMC)
registerDoMC(cores = 5)

# Load example data
data(dhfr) # Dihydrofolate Reductase Inhibitors Data, from caret package
data.X <- dhfr[,-1]
data.Y <- dhfr[,1]
data.Y <- as.numeric(data.Y)

# create third class from randomly selected samples to test multinomial prediction
# since it's random, we don't expect to be able to predict this group
data.Y.multinomial <- data.Y
data.Y.multinomial[sample(1:length(data.Y), 60)] <- 3



# predictions -------------------------------------------------------------

# get results (predicted scores and original labels) from repeated cv
# normaly we would use 10 folds with 10 repeats
results <- run_repeated_cv(data.X, data.Y,
                           n_folds = 2, repeats = 2,
                           alpha = 0.5, family = 'binomial')
# same for multionomial
results_multinomial <- run_repeated_cv(data.X, data.Y.multinomial,
                                       n_folds = 2, repeats = 2,
                                       alpha = 0.5, family = 'multinomial')
# get the performance measures from the computed results
results_to_measures(results[[1]], results[[2]])

# in the case of multinomial prediction, one vs. all measures are computed
# for each group
results_to_measures(results_multinomial[[1]], results_multinomial[[2]])


# statistical signifficance of the whole model ----------------------------

# We will use permutation test how well would are model predicts if there is
# no relationship between X and Y. There are other ways to do this, but this
# one takes most time

n_perm <- 100 # normally this will be between 1000-10000
measures_null_dist <- list()

for (i in 1:n_perm) {
  print(i)
  Y.shuffled <- sample(data.Y)
  res <- run_repeated_cv(data.X, Y.shuffled,
                         n_folds = 2, repeats = 2,
                         alpha = 0.5, family = 'binomial')
  measures_null_dist[[i]] <- results_to_measures(res[[1]], res[[2]])
}
measures_null_dist <- do.call(rbind, measures_null_dist)
# we car only about signifficance of AUC
measured_auc <- results_to_measures(results[[1]], results[[2]])['mean_auc']
null_auc <- measures_null_dist[,'mean_auc']

# p-values cannot be 0, so in the situation where are measured AUC is higher
# than all AUC's from the null distribution, the smalles p-value we can get
# is number of permutations/number of permutations + 1
p_val(null_auc, measured_auc)

# slight change for multinomial predictions, to get a p-value for each contrast
n_perm <- 100
measures_null_dist_multinomial <- list()
for (i in 1:n_perm) {
  print(i)
  Y.shuffled <- sample(data.Y.multinomial)
  res <- run_repeated_cv(data.X, Y.shuffled,
                         n_folds = 2, repeats = 2,
                         alpha = 0.5, family = 'multinomial')
  measures_null_dist_multinomial[[i]] <- results_to_measures(res[[1]], res[[2]])
}

measured_auc_multinomial <- results_to_measures(results_multinomial[[1]],
                                                results_multinomial[[2]])

# we can see that predictions for gorup 1 and two are signifficant, but not
# for group three, which is expected, since it contains random samples
for (group in 1:length(measures_null_dist_multinomial[[1]])) {
  group_null <- lapply(measures_null_dist_multinomial, function(x){return(x[[group]])})
  group_null <-  do.call(rbind, group_null)[,'mean_auc']
  group_measured_auc <- measured_auc_multinomial[[group]]['mean_auc']
  print(c(group, p_val(group_null, group_measured_auc)))
}



# signifficance of predictors ---------------------------------------------

# we will use stability selection to find statistically signifficant set of
# predictors controlling for false discovery rate.
library(c060)
stabpath.obj <- stabpath(as.matrix(data.Y), as.matrix(data.X),
                         family = "binomial", mc.cores = 5, alpha = 0.5,
                         weakness = 1, steps = 1000, standardize = T)
stabsel.obj <- stabsel(stabpath.obj, error = 0.05, pi_thr = 0.75, type = 'pfer')

# list of signifficant variables
stabsel.obj$stable

# stability paths can also be plotted
plot(stabpath.obj, error = 0.05, pi_thr = 0.75, type = 'pfer')

# rerun cv using only the selected variables
data.X.stable <- data.X[,names(stabsel.obj$stable)]
results.stable.vars <- run_repeated_cv(data.X.stable, data.Y,
                           n_folds = 5, repeats = 2,
                           alpha = 0.5, family = 'binomial')

# the method is conservative, therefore it does not identify all important
# variables, which can be see here as a decrease of model performance
results_to_measures(results.stable.vars[[1]], results.stable.vars[[2]])

# the same method can be used for multionomial prediction changing
# the family parameter to 'multinomial'



