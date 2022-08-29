
# SIP trunk and CX customer related churn work; all real data and customer references replaced with alternative non-SIP customer records

#****************************************************************************************************************************************************
## I've selected KNN because taking non-parametric algorithmic approach for churn is extremely effective when under 100,000 transaction examples.
## I'm using a modeling approach built on data transformation, feature engineering, modeling metrics, and predictions using TIDYVERSE and TIDYMODELS
## This means the packages to be used will include rsample, recipes, workflows, parsnip, tune and yardstick, and of course, kknn for KNN.
## I have install code here, but also using RStudio Package Manger for convenience
#****************************************************************************************************************************************************

library(tidyverse)
library(tidymodels)
install.packages("kknn") ## for KNN, referenced by parsnip in set_engine()
packages = c('kknn', 'visdat', 'ggcorrplot', 'ggpubr', 'forcats')  ## other packages
for (p in packages){
  if(!require(p, character.only = T)){
    install.packages(p)
  }
  library(p,character.only = T)
}
## Prefer tidymodels
tidymodels_prefer() ## to resolve preference of functions.

options(scipen=999) ## To minimize scientific notation presentation issue use while using ggplot, then change back to 0 default when needed.
options(scipen=0)  ## default

## *************************************************************************************************************************************************
### Data loading, preparation, identification and grouping of Continuous and Categorical predictor types plus special type class, EDA, visualization
## *************************************************************************************************************************************************
## Getting started: Read target Churn data set from csv file into dataframe from project working directory
## tchurndf is the clean read from the csv file that we will evolve in the Data Preparation and Feature engineering process.
## Because I'm using recipe(), I will also have a special type 'id variable' to track non-used vars in the KNN Churn modeling process.

tchurndf <- read.csv('churn_clean.csv') %>% 
  ## change character to factors
  mutate_if(is.character, as.factor)  ##  data preparation before imbalance check 

anyNA(tchurndf) #check for missing values; verified there are none as it returns FALSE

# get numeric columns for continuous variable identification             
continuous_ident <- head(select_if(tchurndf, is.numeric))  
                    
                    
## I'll be removing CaseOrder below from continuous though it is an int, and will manage it's utility in modeling as an 'id variable' rather than Recipe predictor, and as separate from Categorical or Continuous types.
id_vars_in_dataset <- data.frame(ColumnName=character(), Type=character() )
id_vars_in_dataset <- id_vars_in_dataset %>% add_row(tibble_row(ColumnName = 'CaseOrder', Type = 'id variable'))

## Many Items are surveys; that isn't really Continuous, so remove the columns from continuous_indent, and change them to factors and include in list of categoricals.
## Zip is also really a categorical even though it is an int, so we will include that as a categorical and Factor as well.
continuous_ident <-  select(continuous_ident, -c(Item1, Item2, Item3, Item4, Item5, Item6, Item7, Item8, Zip, CaseOrder ))
continuous_in_datset <- tibble(colnames(continuous_ident)) %>% mutate(Type = "Continuous")  ## 13 variables minus Items, Zip, and CaseOrder, which is an int but an 'id variable', and Lat Lng
names(continuous_in_datset)[1] = "ColumnName" ## I confirmed in Environment

## I check Continuous feature candidates for distribution breadth and skewness. I won't look at CaseOrder, as I designate it in the recipe() later as an 'id variable' column
## I'm graphically representing the following more interesting Continuous Univariate columns to evaluate distribution.
Population <- ggplot(data=tchurndf, aes(x= `Population`)) +
  geom_histogram(bins=20, color="black", fill="blue") +
  ggtitle ("Population: 1 Mile Radius")

Children <- ggplot(data=tchurndf, aes(x= `Children`)) +
  geom_histogram(bins=20, color="black", fill="cornflowerblue") +
  labs(x = "Children", title = "Customer: Children in Household")

Age <- ggplot(data=tchurndf, aes(x= `Age`)) +
  geom_histogram(bins=20, color="black", fill="orange") +
  labs(x = "Age", title = "Customer: Age")

Income <- ggplot(data=tchurndf, aes(x= `Income`)) +
  geom_histogram(bins=20, color="black", fill="green") +
  labs(x = "Customer Income", title = "Customer:Income")
##new
SystemOut <- ggplot(data=tchurndf, aes(x= `Outage_sec_perweek`)) +
  geom_histogram(bins=20, color="black", fill="red") +
  labs(x = "Outages in Secs/Week", title = "Customer: Service Outages")

YrlyFailures <- ggplot(data=tchurndf, aes(x= `Yearly_equip_failure`)) +
  geom_histogram(bins=20, color="black", fill="red") +
  labs(x = "Equipment Failures", title = "Customer: Yrly Equipment Failures")

Email2 <- ggplot(data=tchurndf, aes(x= `Email`)) +
  geom_histogram(bins=20, color="black", fill="darkblue") +
  labs(x = "Emails to Customer in last Yr", title = "Customer: Emails")

Contacts <- ggplot(data=tchurndf, aes(x= `Contacts`)) +
  geom_histogram(bins=20, color="black", fill="black") +
  labs(x = "Customer contacted Support", title = "Customer: Support Contacts")

Tenure <- ggplot(data=tchurndf, aes(x= `Tenure`)) +
  geom_histogram(bins=20, color="black", fill="darkgreen") +
  labs(x = "", title = "Customer: Tenure")

MoCharge <- ggplot(data=tchurndf, aes(x= `MonthlyCharge`)) +
  geom_histogram(bins=20, color="black", fill="green") +
  labs(x = "Average Monthly Charge", title = "Customer: Avg Monthly Charge")

Gbyear <- ggplot(data=tchurndf, aes(x= `Bandwidth_GB_Year`)) +
  geom_histogram(bins=20, color="black", fill="purple") +
  labs(x = "Gigabytes per Year", title = "Customer: Average GB Usage/Yr")
## Simplifying presentation by grouping continuous ggplot objects
ggarrange(Population, Children, Age, Income, SystemOut,YrlyFailures, Email2, Contacts, Tenure, MoCharge, Gbyear, ncol = 2, nrow = 6)

## I can see from Plots that skewness is the rule in our continuous predictors, rather than the exception, so I'll deal with the skewness in the Recipe steps

# Per EDA work, Item1 to Item8 values, while integers, are in fact ordinals representing categorical levels, and Zip is unique categories, so to make tracking easy I'll change to Factors
tchurndf <- tchurndf %>%  mutate(across (c(Item1, Item2, Item3, Item4, Item5, Item6, Item7, Item8, Zip), as.factor)) ## add
                          
# get factor columns for categorical identification; will make adjustments for $Customer_id, $Interaction, and UID b/c are character values that are identifiers for Customers or observations. Will assign to role of 'id variable'             
categorical_ident <- head(select_if(tchurndf, is.factor)) %>% 
                    select(-c(Customer_id, Interaction, UID))
glimpse(categorical_ident)

categoricals_in_dataset <- tibble(colnames(categorical_ident)) %>% mutate(Type = "Categorical") 
names(categoricals_in_dataset)[1] = "ColumnName" ## I confirmed in Environment

id_vars_in_dataset <- id_vars_in_dataset %>% add_row(tibble_row(ColumnName = 'Customer_id', Type = 'id variable')) %>% 
                                     add_row(tibble_row(ColumnName = 'Interaction', Type = 'id variable')) %>%
                                     add_row(tibble_row(ColumnName = 'UID', Type = 'id variable'))
## Check for data set imbalance... 
## Churn is the Outcome Variable, so we look at Yes and Nos for Churn in the data set to check and see if there is too much data set imbalance.
summary(tchurndf$Churn) ## a look at Churn var 
levels(tchurndf$Churn)
## The data is OK, as the imbalance is manageable as KNN creates a decision surface that adapts well to the shape of the data, 
## so we should still get good accuracy rates since we will have a big training set
## I won't use any techniques like SMOTE, undersampling of active customers or oversampling of Churn(s)
## as mentioned previously when I define the formula in the recipe, I'll also deal with other data issues and Feature engineering using available functions.
## I've checked and prepped using dataset and Data Dictionary plus exploratory data analysis now and from before, and considered best Features for KNN. I know how I'll define my formula and recipe():
## - the 'step_'s in recipe() will include functions to deal with skewness, normalization, dummies, etc. One special step, update_role() will designate variables to 'id variable' rather than continuous or categorical. 
## - Based on the characteristics of KNN, I plan for the 1st time in Churn work to include some aspect of location; however, my approach will be to
## drop longitude (aka $Lng), City, State, County, and Zip. Instead I'll engineer Features for location using Lat with TimeZone changes using fct_collapse() in code below. 
## - Exploratory data analysis also suggest dropping numerical values Population, Income, Outage_sec_perweek, Email, and Yearly_equip_failure
## and not use categoricals Job (b/c high cardinality), Area (unneeded location category), and Gender (not influential) in the Churn KNN formula definition
## - also will not include Item 1 to Item 8 either as numerical or an ordered factor because of proven limited utility

## In my review process, making sure `Yes` is mapped to the first level of 'Churn`, as when using tidymodels it goes to the 1st level for positive class in all performance metrics functions
tchurndf$Churn <- relevel(tchurndf$Churn, "Yes")  ### the releveling for Churn as per above.
levels(tchurndf$Churn)

## I'm going to take a different approach with KNN using features related to location that I have in the past in other classification work.
## I'm going to engineer a Feature relationship that will work with KNN Euclidean distance by using a combination of $TimeZone (for east-west in the U.S.) and $Lat (for north-south latitude)
## To make it work, I'm going to consolidate the existing 25 different TimeZone values down to the 7 key U.S. zones - mapped as Puerto_Rico, Eastern, Central, Mountain, Pacific, Alaska, Hawaii
## Having this east to west representation also means I No longer need $Lng to provide that location information, so I'll drop it as a feature in the formula statement in recipe.
tchurndf %>%  count(TimeZone, sort = TRUE) ## What I start with in TimeZones

tchurndf$TimeZone <- fct_collapse(tchurndf$TimeZone,   ## reducing TimeZone levels to the key 7 U.S. zones ##
                          Eastern = c("America/Detroit", "America/Indiana/Indianapolis", "America/Indiana/Marengo", "America/Indiana/Petersburg", "America/Indiana/Vincennes", 
                         "America/Indiana/Winamac", "America/Kentucky/Louisville", "America/New_York", "America/Toronto"), 
                          Central = c("America/Chicago", "America/Indiana/Knox", "America/Indiana/Tell_City", "America/Menominee", "America/North_Dakota/New_Salem" ),
                          Mountain = c("America/Denver", "America/Boise", "America/Ojinaga", "America/Phoenix"),
                          Pacific = c("America/Los_Angeles"),
                          Alaska = c( "America/Anchorage", "America/Juneau", "America/Nome", "America/Sitka"),
                          Hawaii = "Pacific/Honolulu",
                          Puerto_Rico = "America/Puerto_Rico")
levels(tchurndf$TimeZone) 


set.seed(201) # BAT This is a psuedo-random number seed for randomization of data sampling on splits, etc. I'm using a consistent seed for generation

## Given it is KNN, I'll take the "train" data to use for neighbor evaluations from the data set.
## I'll be doing a split of the data, then also performing Cross Validation.
## The most simple Cross validation would be just using the split off Test and Train. But churn_clean is a reasonably large data set, so I'm doing something more sophisticated...
## I'll use the resample package for splitting and folds in the modeling process, and I'll use a 2 for the folds.

tchurndf_split <- initial_split(tchurndf, prop = 0.75, ## 75%/25% is usually the default proportions of the split, and our data set is a good size.
                                    strata = Churn) ## indicates the Outcome

tchurn_train <- training(tchurndf_split)

tchurn_test <-  testing(tchurndf_split)
## Write the files for split data tracking
write.csv(tchurn_train, file = "tchurn_train.csv")
write.csv(tchurn_test, file = "tchurn_test.csv")

## Write the files for prepared data CSV submission
write.csv(tchurndf, file = "knn_tchurndf.csv")


## I've chosen to do k-fold cross validation on the training data for the KNN classification on so we can work with the hyperparameters
## The rsample package supports multiple fold options including Leave One Out, and Monte Carlo, but I'm electing to resample using vfold_cv()
## I'll use (little k for v_fold_cv) k=2 to create two(2) folds on tchurn_train because 1) we have a split already, 2) k=2 has worked well in KNN classification
## 3) the KNN (big) K values we are evaluating based on statistical guidance are quite large - at the high end 100 plus

tchurn_cvfold <- vfold_cv(tchurn_train, v = 2) 

## When I originally loaded churndf from the csv file, I did basic work in preparation of the Churn Data before the track of split and v_fold_cv process.  
## In the next section of the code I'll use the tools in the Tidyverse/Tidymodels recipes package and create a recipe() for pre-processing,
## what the statistical machine learning community would refer to as Feature Engineering!
## ************************************************************************************************************************
### Feature and Formula definition using the recipe package from tidymodels
## ************************************************************************************************************************
## Now we can create a Feature and Formula recipe() for this data.
## In order to accomplish this in a functional approach, I'm using the formula heuristic that is part of R here in the recipe() function.
## I'll define the Outcome variable (Churn) and the chosen Predictors to use in my KNN work as part of my recipe() object, and start Feature pre-processing work.
##  - for the reasons I've previously enumerate, dropping the following variables from the KNN formula: Item1 to Item8, PaperlessBilling, TechSupport, DeviceProtection, Tablet,
## Yearly_equip_failure, Contacts, Email, Outage_se_perweek, Gender, Income, Job, Area, Population, Zip, County,State, City
## In the code below, I'll pipe (%>%) all additional "steps_" needed to the tchurn_rec recipe, including:
## - Declaring CaseOrder, Customer)id, Interaction, UID as an 'id variable' using _role which means the variable isn't a predictor, but instead identifies an observation or customer
## - Deal with any correlation issues
## - Remove skewness from numeric predictors
## - Normalize all numeric predictors
## - Create dummy variables for all nominal predictors .

## I'm going to create a Formula object for my KNN classification effort with Churn as Outcome and the select feature variables and assign it to churn_features_knn.
churn_features_knn <- Churn ~  CaseOrder + Customer_id + Interaction + UID + Lat + TimeZone  + Children + Age + Marital + Techie + Contract + Port_modem + InternetService + Phone + Multiple + OnlineBackup + 
  StreamingTV + StreamingMovies + Tenure + PaymentMethod + MonthlyCharge + Bandwidth_GB_Year

## BAT Recipe includes the formula churn_features_knn, and a consistent set of transformations that need to be applied in the modeling process.

tchurn_rec <- recipe(churn_features_knn, data = tchurn_train) %>% ## arguments are the formula from above, and tchurn_train split designated as the starting data 
              ## BAT now the feature/function/"step_" work 
              update_role(CaseOrder, Customer_id, Interaction, UID,  new_role ='id variable') %>% ## now these variables only identify an observation or Customer; aren't used in the recipe formula
              step_corr(all_numeric(), -all_outcomes(), -has_role('id variable')) %>% ## removes variables with large absolute correlations (default .9) with other vars; unsupervised related to model
              step_YeoJohnson(all_numeric(), -all_outcomes(), -has_role('id variable')) %>% ## adjust scale with transformation to where the distribution will approximate a symmetrical one,
              step_center(all_numeric(), -all_outcomes(), -has_role('id variable')) %>%   ## sub function of step_normalize(), but breaking it out here.
              step_scale(all_numeric(), -all_outcomes(), -has_role('id variable')) %>%   ## performs sub function of normalization that scales to a standard deviation of one
              step_dummy(all_nominal(), -all_outcomes(), -has_role('id variable')) 

summary(tchurn_rec)

## I've identified dataset vars by type, and created individual dataframes for Continuous types, Categorical types, and the id variables'. I combine them for reference  
tchurndf_all_types <- rbind(id_vars_in_dataset, continuous_in_datset, categoricals_in_dataset ) ## All dataset vars with types '
churn_formula_types <- tchurndf_all_types[tchurndf_all_types$ColumnName %in% all.vars(churn_features_knn) ,]


## Pre-processing and feature work in a CSV
## with all 10K rows but including formula and recipe
knn_full_recipe <- tchurn_rec %>% ## running the prep() and bake() recipe functions; bake includes a parameter to reuse the "recipe" with a new dataset if needed.
  prep() %>% 
  bake(new_data = tchurndf) 
write.csv(knn_full_recipe, file = "knn_full_recipe.csv")

## I've completed defining the tchurn_rec recipe object; this is a nice part of Feature Engineering with tidymodels interface approach, as the instance can now be used in variety of ways.
tchurn_rec %>% ## running the prep() and bake() recipe functions; bake includes a parameter to reuse the "recipe" with a new dataset if needed.
  prep() %>% 
  bake(new_data = tchurn_train)

## ************************************************************************************************************************************
### KNN Classification model and hyperparameter definition using kknn package as engine and tidymodels parsnip, tune, workflow packages
## ************************************************************************************************************************************
## I'm going to use the nearest_neighbor() function for KNN from the parsnip package in defining my classification model, kknn_class.tc
kknn_class <- nearest_neighbor(neighbors = tune()) %>% ## I'm going to use neighbors hyperparameters from a tune tibble of potential values, not a K default number which would be 5.
              set_engine('kknn') %>% #specifies KNN, specifically from kknn package, and then I set mode for Classification
              set_mode('classification')

## kknn's default distance_power is 2, which equates to Euclidean distance, which is fine for this so I'm taking the default and not setting it.
## Now I'm going to bundle the recipe and model using the workflows package

tchurn_wkflow <- workflow() %>% 
                 add_model(kknn_class) %>% 
                 add_recipe(tchurn_rec)

## I want to consider 31, 33, 43, 51, 75, 99, 123, and 153 as my candidate K hyperparameters so I can determine the best possible "neighbor" value for the model!
## Create a labeled vector neighbors with the candidates, and then make it a tibble so it matches the tune_grid() function parameter requirements
neighbors = c(31, 33, 43, 51, 75, 99, 123, 153)
## Make it a tibble so it matches the tune_grid() function parameter requirements
tchurn_Ks <- tibble(neighbors)

## The evaluations are performed using functions from the tune package, a Tidy Tuning tool. I've already used tune() in the kknn_class code.
## For my next step I'll continue with Tidy Tuning and use  the tune_grid() function to evaluate my tchurn_Ks candidates.

set.seed(201)  ## doing this again using 201 because I sometimes jump around in testing

## I now take the workflow containing the model and recipe, and use it to Tune the model.
## The tune_grid functions gets the specified 2 cross validation folds of data, and the eight K values that are my hyperparameters. It then iterates through combinations.
tcknn_tune <- tchurn_wkflow %>% 
              tune_grid(resamples = tchurn_cvfold, grid = tchurn_Ks )  
print(tcknn_tune)  ## I take a look at the Tuning object results from the evaluation of my tchurn_ks candidates against the folds. This can be seen in RStudio Environment pane as well.

## I use the autoplot() function of tune to see a quick graphical breakdown of how my hyperparameters for K performed.
autoplot(tcknn_tune, type = "marginals")## excellent hyperparameter results with the higher K values; eyeballing tells me 153 is best for AUC.

## **************************************************************************************************************************************************
### Review KNN model and hyperparameter results, ROC/AUC and accuracy metrics, Classification prediction outcomes on Churn including Confusion Matrix
## **************************************************************************************************************************************************
## Use the tune show_best() function to look at the top 5 models, and then use select_best() to identify the best both by accuracy AND the best by AUC...
top5_acc<- tcknn_tune %>% show_best('accuracy')
top5_auc <- tcknn_tune %>% show_best('roc_auc')

topK_acc <- tcknn_tune %>% select_best('accuracy')
topK_auc <- tcknn_tune %>% select_best('roc_auc')

## I'll finalize the effort by using the best K for AUC in the updated workflow
topK_auc_wkflow <- tchurn_wkflow %>% finalize_workflow(topK_auc)

## I now have a tuned hyperparameter and model using the best K based on AUC (153) included in the workflow object topK_auc_wkflow
## I'll go through the process of evaluation again to fit to the entire training set split, which includes the test data to validate performance
last_fit_auc <- topK_auc_wkflow %>% 
                last_fit(split = tchurndf_split ) # split including test data

## Check the metrics of KNN with K = 153 for performance against tchurndf_split
tchurn_knn_metrics <- collect_metrics(last_fit_auc)

tchurn_knn_metrics


## Now take a look at the predictions KNN actually makes on Churn. I'll do this both by looking at the object itself, and then plotting
churn_predictions <- last_fit_auc %>% 
                     collect_predictions()

churn_predictions

## I'll use roc_curve and autoplot functions with the churn_predictions to visually plot the dataframe data ROC curve (and so also AUC)
churn_predictions %>% 
    roc_curve(truth = Churn, estimate = .pred_Yes) %>%  #from columns in Churn_predictions
    autoplot()  #plotting function that is part of workflow/workflowsets

## To make comprehension of Classification prediction results easier, I'll represent the churn_predictions results 
## (ground truth is Churn, prediction is .pre_class column) in a Confusion Matrix
## The conf_mat function is from yardstick, aka "Tidy Characterizations of Model Performance"
tchurn_cmatrix <- conf_mat(churn_predictions, truth = Churn, estimate = .pred_class)
tchurn_cmatrix

## NOTE: I've chose to focus on ROC/AUC, but precision can be relevant at prediction volume
## where taking actions without need is OK when cost effective to prevent churn?

