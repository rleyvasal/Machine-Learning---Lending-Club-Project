## Rigoberto Leyva Salmeron
## Machine Learning - Prediction of Loan Default

library(pacman)
p_load(tidyverse, arrow, fs, tictoc, lubridate, beepr, DataExplorer, tidymodels, C50, skimr)

lending_club_data_2012_2014 <- arrow::read_parquet("data/lending_club_data_2012_2014.parquet")

# Drop variables and rows

missing.5 <- map_df(lending_club_data_2012_2014, function(x) mean(is.na(x)) > 0.5) |>       # map to get proportion of nas  on each column>.5
  select_if(~.x == TRUE)         # Select only variables with TRUE value ( >.5 missing)              

vars <-c(names(missing.5)) #variable names to be dropped inside a vector

lending2 <- lending_club_data_2012_2014 |> select(-all_of(vars)) #Drop all the variable names identified

#  select model variables 

#  Did not select debt_settlement_flag even though it was identified by the Boruta algorithm because it is variable that identifies people who have defaulted and settle their debt.

lending2 <- lending2 |> select(loan_status, loan_amnt, funded_amnt , funded_amnt_inv , term , 
                                int_rate , installment , grade , sub_grade , annual_inc , 
                                verification_status, dti , fico_range_low , 
                                fico_range_high , open_acc , revol_bal , revol_util , total_acc , 
                                initial_list_status , out_prncp , out_prncp_inv , total_pymnt , 
                                total_pymnt_inv , total_rec_prncp , total_rec_int , total_rec_late_fee , 
                                recoveries , collection_recovery_fee , last_pymnt_d , last_pymnt_amnt , 
                                last_credit_pull_d , last_fico_range_high , last_fico_range_low , 
                                tot_cur_bal , total_rev_hi_lim , avg_cur_bal , bc_open_to_buy , 
                                num_actv_bc_tl , num_actv_rev_tl , num_bc_sats , 
                                num_bc_tl , num_op_rev_tl , num_rev_accts , num_rev_tl_bal_gt_0 , 
                                num_sats , tot_hi_cred_lim , total_bal_ex_mort , total_bc_limit ,  year)

#Drop Missing values on rows.


lending2 <- lending2 |> drop_na()



#Create and indicator variable for default as a factor.


lending2 <- lending2 |> mutate(default = as.factor(ifelse(loan_status %in% c("Charged Off"), "yes", "no")))

lending2 <- lending2 |> mutate(default = fct_relevel(default, c("yes", "no"))) #change levels of factors to have yes as 1st prediction

levels(lending2$default) 


## Drop loan_status: variable was used for creating causing problems at the end when calculating metrics 

lending2 <- lending2 |> select(-loan_status)

# split the data


lending_split <- lending2 |> initial_split()

train <- lending_split |> training()

test <- lending_split |> testing()



# Check proportions of default in original and splits


lending2 |> count(default) |> 
  mutate(prop = n/sum(n))

train |> count(default) |> 
  mutate(prop = n/sum(n))

test |> count(default) |> 
  mutate(prop = n/sum(n))

# proportion defaulted on loans


train |> group_by(default) |>
  summarize(n = n()) |>
  mutate(freq = n / sum(n))


# Null model 

default_null_model <- logistic_reg(mode = "classification") |>
  set_engine("glm") |>
  fit(default ~ 1, data = train) # Figure out the percentage on each class of y variable

# Predict default with the null model 

default_null_model |> predict(new_data = train, type = "class")


# Use the null model to predict default and then bind columns of train dataset 


default_null_model |>
  predict(new_data = train, type = "class") |>
  bind_cols(train)


# Accuracy of null model: predicting everything as default, use the null model and predict the train dataset, bind columns of the train dataset and calculate accuracy with predict() by comparing the observed value for **default** and the predicted value **.pred_class** calculated by predict()


default_null_model |>
  predict(new_data = train, type = "class") |>
  bind_cols(train) |>
  accuracy(truth = default, estimate =.pred_class) # comparing observed(truth) and predicted value(.pred_class)


# Confusion matrix verifies only "no" default has been predicted correctly. 


default_null_model |>
  predict(new_data = train, type = "class") |>
  bind_cols(train) |>
  conf_mat(default, .pred_class)


# ## Model C5.0


### Preprocessing with recipe

sample1 <- train |> sample_n(size = 1000) # for testing

set.seed(345)

c50_recipe <- sample1 |> recipe(default ~ .) |>         # Recipe  without formula y = default x's = all other variables can it work withut recipy and prep/
  step_nzv(all_predictors()) |>                       # Remove variables with little variance     
  step_scale(all_numeric())|>                         # Re-scale all numeric variables
  prep()                                               # apply the steps above


### Specify the C5.0 model 


c50_model <- boost_tree() |>  
  set_engine("C5.0") |>                          # C5.0 package
  set_mode("classification")

### cross validattion


c50_cv <- vfold_cv(sample1, folds = 10)               #create 10 folds to test on each

c50_grid <- expand.grid(mtry = c(3,7), trees = tune())  #create a grid with tuning parameters

c50_workflow_cv  <- workflow() |>                      #create the workflow
  add_recipe(c50_recipe) |>
  add_model(c50_model)


c50_tune_results <- c50_workflow_cv |>               # Use the workflow and tune it with different parameters
  tune_grid(resamples = c50_cv,                       # use the  resamples created with cross validation
            grid = c50_grid,                         # use the grid created above 
            metrics = metric_set(accuracy, roc_auc, kap) )   # Specify metrics needed 

c50_tune_results |> collect_metrics()                    #collect metrics from output of tune results

c50_tune_results |> show_best("accuracy")                # show the best tree model with the highest accuracy

c_50_best <- select_best(c50_tune_results)                # select the model with best accuracy to be used to finalize  the workflow

finalized_results <- c50_workflow_cv |> finalize_workflow(c_50_best)  #Use workflow pipe to finalize workflow with best c50 extracted above

final_c50 <- 
  finalized_results |>
  fit(sample1)                # Fit  the best model  identified  to train data 

library(vip)      # Library to get the most important variables

final_c50 |>                 # importance of variables
  pull_workflow_fit() |> 
  vip()



final_fit <- last_fit(c50_workflow_cv, split = lending_split) # Fit finalized model on full training dataset and evaluates finalized model on testing data 


final_fit |> 
  collect_metrics()     #get accuracy on the test data

final_fit |>
  collect_predictions() |> # extract the probabilities form the final fit
  roc_curve(truth = default, estimate = .pred_yes) |>   # get the predictions probability values   default  and predicted  probability .pred_yes
  autoplot()  # plot the porbabiliteis


c50_resample_results <- fit_resamples(c50_workflow_cv, resamples=c50_cv)

c50_resample_results |> collect_metrics

c50_resample_results |> show_best("accuracy")

### Setup the workflow


c50_wflow <-                               
  workflow() |>
  add_recipe(c50_recipe) |>                   # Add recipe created above to workflow
  add_model(c50_model) |>    # Add the model created above default ~ .
  fit(data=train)             # Fit the model


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate the metrics


c50_wflow |> 
  predict(test) |>  # Predict the test dataset based on workflow 
  bind_cols(test)|> # Bind the columns of the test dataset
  metrics(truth = default, estimate = .pred_class) # Compare observed value (default) vs predicted value (.pred_class)


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate confusion matrix


c50_wflow |> 
  predict(test) |>  # Predict based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  conf_mat(truth = default, estimate = .pred_class) # Compare observed default vs predicted value in a confusion matrix


# ROC curve plot


c50_wflow |> 
  predict(test, type = "prob") |>  # Predict probabilities "pred_yes" , "pred_no" based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  roc_curve(truth = default, estimate = .pred_yes) |>  # Compare observed default vs predicted value in a confusion matrix
  autoplot()

### Specify h2o model
library(h2o)
library(h2oparsnip)

h2o.init()

h2o_model <- boost_tree(mode = "classification", trees =  tune(), min_n = tune()) |>
  set_engine("h2o")

h2o_grid <- expand.grid(trees =  c(20), min_n = c(5))

h20_workflow_cv  <- workflow() |>
  add_recipe(c50_recipe) |>
  add_model(h2o_model)

h20_tune_results <-  h20_workflow_cv |>
  tune_grid(resamples = c50_cv, 
            grid = h2o_grid, 
            metrics = metric_set(accuracy) )

### Specify the Random Forest model 


rf_model <- rand_forest(trees = 20) |>           # 20 trees 
  set_engine("ranger") |>                          # C5.0 package
  set_mode("classification")



### Setup the workflow


rf_wflow <-                                    
  workflow() |>
  add_recipe(c50_recipe) |>                   # Add recipe created above to workflow
  add_model(rf_model) |>    # Add the model created above default ~ .
  fit(data=train)             # Fit the model


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate the metrics


rf_wflow |> 
  predict(test) |>  # Predict the test dataset based on workflow 
  bind_cols(test)|> # Bind the columns of the test dataset
  metrics(truth = default, estimate = .pred_class) # Compare observed value (default) vs predicted value (.pred_class)


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate confusion matrix


rf_wflow |> 
  predict(test) |>  # Predict based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  conf_mat(truth = default, estimate = .pred_class) # Compare observed default vs predicted value in a confusion matrix


# ROC curve plot


rf_wflow |> 
  predict(test, type = "prob") |>  # Predict probabilities "pred_yes" , "pred_no" based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  roc_curve(truth = default, estimate = .pred_yes) |>  # Compare observed default vs predicted value in a confusion matrix
  autoplot()

#### Specify XGBoost recipe 

xg_recipe <- train |> recipe(default ~ .) |>         # Recipe  without formula y = default x's = all other variables can it work withut recipy and prep/
  step_nzv(all_predictors()) |>                       # Remove variables with little variance     
  step_scale(all_numeric()) |>                        # Re-scale all numeric variables
  step_dummy(c(term,grade, sub_grade, 
               verification_status,initial_list_status,  # Select all the categorical varialbes
               last_pymnt_d,last_credit_pull_d), 
             one_hot = TRUE) |>                        # Convert them to dummy variables: columns or each value with 1's and 0s
  prep() 

### Specify the XGBoost model 


xg_model <- boost_tree(trees = 20) |>           # 20 trees 
  set_engine("xgboost") |>                          # C5.0 package
  set_mode("classification")



### Setup the workflow


xg_wflow <-                                    
  workflow() |>
  add_recipe(xg_recipe) |>                   # Add recipe created above to workflow
  add_model(xg_model) |>    # Add the model created above default ~ .
  fit(data=sample1)             # Fit the model


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate the metrics


xg_wflow |> 
  predict(test) |>  # Predict the test dataset based on workflow 
  bind_cols(test)|> # Bind the columns of the test dataset
  metrics(truth = default, estimate = .pred_class) # Compare observed value (default) vs predicted value (.pred_class)


### Predict probabilities on test dataset and bind columns of test dataset to compare them and calculate confusion matrix


xg_wflow |> 
  predict(test) |>  # Predict based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  conf_mat(truth = default, estimate = .pred_class) # Compare observed default vs predicted value in a confusion matrix


# ROC curve plot


xg_wflow |> 
  predict(test, type = "prob") |>  # Predict probabilities "pred_yes" , "pred_no" based on workflow the test dataset
  bind_cols(test) |> # Bind the columns of the test dataset
  roc_curve(truth = default, estimate = .pred_yes) |>  # Compare observed default vs predicted value in a confusion matrix
  autoplot()


us

