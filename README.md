# SyriaTel Customer Churn Prediction
## 1. Business Understanding
### 1.1 Business Overview
The telecommunications industry has become very competitive over the years, with customer retention emerging as a critical challenge. One of the
major issues facing telecom providers is customer churn, a scenario where users discontinue their service, either due to dissatisfaction from the
provider or due to the availability of better alternatives. High churn rates can significantly impact a company's overall revenue, and scaling potential.
### 1.2 Problem Statement
SyriaTel, a leading telecom provider, is experiencing a significant loss of customers. To address this challenge, the company seeks to build a robust
predictive model capable of identifying customers who are at risk of churning. By using data driven insights and predictive modeling, SyriaTel aims to
understand the key drivers of customer churning, determing methods of improving long term retention of customers and enhance long term
customer loyalty
## 2. Data Understanding
In this step, we explore the dataset to understand what kind of information it contains. We look at the different features, their data types, and check
for things like missing values or unusual patterns. This helps us get a clear picture of the data before moving on to cleaning and modeling.
### 2.1. Import Libraries
For this project, we will implement the following tools and libraries
Numpy: for numerical computations
Pandas: for data loading, cleaning and manipulation
Seaborn: for data visualization and EDA
Matplotlib: for data visualization and EDA
Scikit-learn: for data preprocessing, predictive modeling and model evaluation.
Imblearn: for dealing with class imbalance.
### 2.2 Load the Dataset
We will load the dataset, check the info and summary statistics of the dataset.
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3333 entries, 0 to 3332
Data columns (total 21 columns):
# Column Non-Null Count Dtype 
--- ------ -------------- ----- 
0 state 3333 non-null object 
1 account length 3333 non-null int64 
2 area code 3333 non-null int64 
3 phone number 3333 non-null object 
4 international plan 3333 non-null object 
5 voice mail plan 3333 non-null object 
6 number vmail messages 3333 non-null int64 
7 total day minutes 3333 non-null float64
8 total day calls 3333 non-null int64 
9 total day charge 3333 non-null float64
10 total eve minutes 3333 non-null float64
11 total eve calls 3333 non-null int64 
12 total eve charge 3333 non-null float64
13 total night minutes 3333 non-null float64
14 total night calls 3333 non-null int64 
15 total night charge 3333 non-null float64
16 total intl minutes 3333 non-null float64
17 total intl calls 3333 non-null int64 
18 total intl charge 3333 non-null float64
19 customer service calls 3333 non-null int64 
20 churn 3333 non-null bool 
dtypes: bool(1), float64(8), int64(8), object(4)
memory usage: 524.2+ KB
From the info() function, we can see the following:
The dataset contains a total of 3333 records, and 21 columns/features.
The numerical features are about 16, while the categorical columns are about 4, excluding the target variable, which is churn.
Next, we want to check the descriptive statistics of the dataset. In this section, we will use the describe() function to check for:
count: The total number of records in each numerical column
mean: The average value in each numerical column
std: The standard deviation
min: The minimum value in each numerical column
25%: The 25th percentile value in each numerical column
50%: The 50th percentile value (median) in each numerical column
75%: The 75th percentile value in each column max: The maximum value in each column
### 2.3. Feature Understanding
Below is a description of all the numerical and categorical features in the dataset: Numerical Features:
account length: The number of days the customer has been an account holder. area code: The area code associated with the customer's phone
number. number vmail messages: The number of voice messages received by the customer. total day minutes: The total number of minutes
used by the customer during the day. total day calls: The total number of calls made by the customer during the day. total day charge: The total
charges incurred by the customer during the day. total eve minutes: The total number of minutes used by the customer in the evening. total eve
calls: The total number of calls made by the customer in the evening. total eve charge: The total charges incurred by the customer in the
evening. total night minutes: The total number of minutes spent by the customer at night. total night calls: The total number of calls made by the
customer at night. total night charge: The total charged incurred by the customer at night. total intl minutes: The total number of minutes spent
by the customer on international calls total intl calls: The total number of international calls made by the customer total intl charge: The total
charge incurred by the customer on international calls. customer service calls: The number of calls made by customer service to customers.
Categorical Features:
state: The customer's state of residence. phone number: The customer's mobile number. international plan: Indicates if the customer has subscribed
to an international plan (Yes/No) voice mail plan: Indicates if the customer has a voice mail plan (Yes/No) Now that we have a rudimentary
understanding of the data, we can proceed to implementing some data preparation techniques.
## 3. Data Preparation
In this section, we will look into data cleaning techniques, Exploratory Data Analysis (EDA) and data preprocessing (data wrangling) for our dataset.
This step is paramount to provide data that will contribute significantly to the performance of the prediction model
### 3.1 Data Cleaning
In this section, we perform some data cleaning techniques on the dataset. These techniques include:
Checking for null values and handling them.
Checking for duplicate rows and dropping them.
Standardizing the columns by adding an underscore between each word in a column, and capitalizing the 1st letter of each word in a column.
We created a function from the file utility.py called clean_nulls_and_duplicates that will perform this task.
### 3.2 Exploratory Data Analysis
In this section, we will perform a systematic investigation of the dataset to extract insights, evaluate feature distributions, assess relationships
between features and the target variables, and identify anomalies, outliers or data quality issues. This helps inform feature engineering decisions
and guides the selection of appropriate modeling techniques.
### 3.2.1. Univariate Analysis
Univariate analysis in EDA aims to explore and analyze each feature in a dataset to understand its distribution, central tendency and spread. It also
seeks to detect presence of outliers, anomalies or inconsistencies present in the data.
Churn Distribution
In this section, we will look at the distribution of the unique values in the Churn column.
<img width="887" height="526" alt="image" src="https://github.com/user-attachments/assets/7b885118-c898-4352-b862-1e744cea62ac" />

From the bar plot, out of the 3333 customers in the dataset, 483 have churned from the company(i.e. terminated their contract), which is about 14.5
% of the total customers. This indicates that the target variable is highly imbalanced. This imbalance can negatively impact the performance of the
prediction model by influencing the model to make false predictions. Therefore, this class imbalance should be handled before modeling.
Area Code Distribution
In this section, we want to see how the distribution of the customers is with regards to the area code. This will aid in determining the area codes that
have the most customers
<img width="778" height="488" alt="image" src="https://github.com/user-attachments/assets/95d25234-4c20-4519-83aa-cb219b6f16a9" />

From the plot, area code 415 has a higher number of customers with about 1655 customers, which accounts for approximately 49.7% of the total
number of customers. Area codes 408 and 510 have a close number of customers, with area code 408 having 838 customers and area code 510
having 840 customers.
The uneven customer distribution suggests that SyriaTel has a larger customer base concentrated in specific regions. This could indicate:
Stronger marketing or network presence in those regions.
Regional preferences for SyriaTel services.
Distribution of categorical features
In this section, we will explore three main categorical features:
State
International_Plan
Voice_Mail_Plan
We created a function plot_categorical_distributions in utility.py that will take in a dataframe and feature, and return the distribution of the input
feature.
We will start with analyzing the State feature.
<img width="1117" height="393" alt="image" src="https://github.com/user-attachments/assets/b3313f45-ec87-4264-98e3-1725b3eb711b" />

From the plot, the top 5 states from where majority of the customers reside are:
West Virginia (WV): The state with the highest number of customers (106)
Minnesota (MN): Has about 84 customers
New York (NY): Has about 83 customers
Alabama (AL): Has about 80 customers
Ohio (OH): Has about 78 customers
Next we will plot the distribution of the International Plan feature
<img width="1120" height="395" alt="image" src="https://github.com/user-attachments/assets/097dac3b-078a-4b53-887b-2a1cfb5e6d49" />

From the plot, about 323 customers, which is about 9.7 % of the total customers have an international plan.
Finally, we will plot the distribution of the Voice_Mail_Plan feature.
<img width="1120" height="393" alt="image" src="https://github.com/user-attachments/assets/f663718c-0402-4f84-accc-ebaf30f3590e" />

From the plot, 922 customers, which is about 27.7% of the total number of customers have a voice mail plan.
Numerical Features Distribution
In this section, we will plot the distributions of all the numerical features, with Kernel Density Estimation (KDE) curves. This will aid in:
Understanding customer usage patterns, with features like Total_Day_Minutes. This will inform the usage of the services by the customers.
Detecting outliers in the dataset. For example, outliers in total charge or minutes could indicate erratic usage linked to churn.
We created a function numerical_distributions in utility.py, that takes in the dataframe, and a list of numerical features, and returns the KDE
distribution plots of each feature.
<img width="1140" height="1428" alt="image" src="https://github.com/user-attachments/assets/ae1eaaed-db30-40e0-bf4b-98a8a2736b07" />

From the plot, all the features, apart from Customer_Service_Calls and Number_Vmail_Messages follow a Normal distribution. Despite the
Total_Intl_Calls feature being skewed to the right, it is still normally distributed.
### 3.2.2. Bivariate Analysis
Bivariate analysis in EDA aids in understanding the relationship between two variables, in this case the relationship between the independent
features and the target variable Churn. This is crucial for identifying the predictive features to implement in modeling.
Categorical features vs Churn
In this section, we will use bar plots to analyze the relationship between the categorical features and Churn. The categorical features we will use are:
State, International_Plan and Voice_Mail_Plan.
This will help us in answering questions like:
Do customers with an international plan or voice mail plan churn more?
We created a function categorical_churn that takes in a dataframe and a feature, and returns the countplot of each feature, with churn as a
comparative variable.
We will first plot the State feature
<img width="608" height="329" alt="image" src="https://github.com/user-attachments/assets/4df6c0cd-1d05-4fc1-b069-d1ecadd967a7" />

From the plot, majority of the customers who churned came from New Jersey, Texas, Maryland, Miami and New York.
Next, we will plot the International_Plan feature.
<img width="620" height="328" alt="image" src="https://github.com/user-attachments/assets/715b2468-cf7a-4809-8515-54266273926f" />

From the plot, most of the customers who churned did not have an international plan.
Finally, we will plot the Voice_Mail_Plan feature
<img width="620" height="328" alt="image" src="https://github.com/user-attachments/assets/82bbc723-8d7f-4a02-a906-6cf1fbe667bf" />

From the plot, most of the customers who churned did not have a voice mail plan.
Customer Service Calls agains Churn
In this section, we want to visualize the variation in churn with the number of customer service calls. This will help us in determining whether
customer service calls are a major contributor towards customer churning.
We will also implement a hue of Area_Code in order to see the area code that had the highest rate of churn
<img width="825" height="333" alt="image" src="https://github.com/user-attachments/assets/cf6962ed-0fc4-4c4e-b86d-b2c5abca96db" />

From the plot, most of the customers who churned came from area code 415 and 510. In addition, customers who churn from the company tend to
have more customer service calls of about 4, than those who do not. It is also evident that there are a number of outliers, which will be dealt with.
Numerical Features vs Churn
In this section, we will investigate the distribution of certain numerical columns with the churn rate. Specifically, we will use: Total_Day_Charge,
Total_Eve_Charge, Total_Night_Charge and Total_Intl_Charge. These 4 plots will help us in understanding the variation of the charging rates with
the churn rate. We will use KDE plots to visualize these distributions.
We created a function kde_plots_with_churn, that will take in the dataframe, feature and the charge type(day, evening, night or international)
<img width="623" height="388" alt="image" src="https://github.com/user-attachments/assets/5bf0966b-0717-43f5-b560-8273b4cad5ad" />

This KDE plot shows the distribution of Total Day Charges for customers who churned as Churn = True vs those who did not as Churn = False. From
the plot, the orange (churned) curve has a longer right tail and maintains density at higher values of day charges. This implies that customers who
churn tend to have higher day charges than those who do not churn.
Next, we will plot the Total_Eve_Charge feature
<img width="617" height="388" alt="image" src="https://github.com/user-attachments/assets/145d3ae2-1d03-4edc-932f-eb2acc65abb0" />

In this KDE plot, The non-churned group (blue) has a tighter and higher peak between 15-20, while the churned group (orange) is lower and flatter,
with a subtle shift towards higher evening charges.
The churned group maintains more density beyond ~25 compared to the non-churned group, similar to the trend seen with the day charges. From
this, customers who churned show a slight tendency to have higher evening charges, but the separation between churned and non-churned is less
distinct.
Next, we will plot the Total_Night_Charge feature
<img width="624" height="388" alt="image" src="https://github.com/user-attachments/assets/081e781a-8a95-45e8-ac10-be4546b2ceed" />

From this KDE plot, The two curves heavily overlap, implying that most customers, whether they churned or not, had similar total night charges. In
this scenario, we cannot say for sure that majority of the customers who churned had a higher night charge.
Finally, we will plot the Total_Intl_Charge feature
<img width="611" height="388" alt="image" src="https://github.com/user-attachments/assets/bfe9c8ed-cc12-48a6-8bec-35b989f22d71" />

From the density plot, The non-churning curve has a tight bell shaped distribution centered around 2.5-3.0 for international charges, which suggests
that most loyal customers have moderate international usage patterns, with relatively little variation.
On the other hand, the churning curve shows a broader, flatter distribution that extends further to the right, with multiple peaks around 3.0-4.0. This
indicates churned customers tend to have higher international charges and more varied usage patterns
Based on this analysis, customers with very low international charges (< 1.0) rarely churn, suggesting basic users tend to stay. The "tail" of high
international charges (> 4.0) is dominated by churned customers, suggesting very high international usage may be a churn risk factor.
### 3.2.3. Feature correlation
In this section, we will use a correlation heatmap to measure the correlation between the features and the target variable.
We created a function correlation_heatmap that takes in a dataframe and returns a correlation heatmap of the various numerical columns with the
target variable.
<img width="686" height="592" alt="image" src="https://github.com/user-attachments/assets/a96e4678-346f-4282-bb09-d49738c8a4e1" />

From the correlation heatmap, some of the features share a perfect correlation of 1.0. They include:
Total_Day_Charge and Total_Day_Minutes features, which are fully positively correlated.
Total_Eve_Charge and Total_Eve_Minutes features, which are fully positively correlated.
Total_Night_Charge and Total_Night_Minutes features, which are fully positively correlated.
Total_Intl_Charge and Total_Intl_Minutes features, which are fully positively correlated.
These perfect correlations indicate a completely linear relationship between usage and charges within each time period. This suggests:
Fixed per-minute rates for each time period
No complexity in pricing.
However, from a modeling perspective, this causes severe multicollinearity because one varibale is perfectly predictable from the other.
### 3.2.4. Multicollinearity check
Multicollinearity occurs when two or more independent variables in a dataset are highly correlated, implying that they provide overlapping
information. This can lead to unreliable model coefficients, and reduced model interpretability. In order to address this challenge, we will compute the
correlation matrix, and drop one feature from each pair of variables with a correlation coefficient greater than 0.9. This threshold will help eliminate
repetition while preserving the most informative features. Removing highly correlated features improves model stability, reduces overfitting risk and
ensures that each remaining variable contributes uniquely and significantly to the model's predictions.
In this section, we created a function drop_highly_correlated_features that takes in a dataframe, and returs a dataframe with dropped features.
### 3.2.5. Handling outliers
Outliers are data points that significantly differ from other observations and can distort machine learning models. They may arise from data entry
errors, meaurement anomlies or genuine but rare events.
To handle them, we first identify outliers using Z-score, which measures how many standard deviations a value is from the mean. In this case, we
use the Z-score method to filter out numerical outliers, removing any row where a feature's Z-score exceeds a defined threshold, which in this case
is 3. This approach ensures a cleaner dataset for more reliable modeling.
How the code works:
Identify numeric columns: We select all columns of numeric dtype (integers and floats)
Compute the Z-scores: Using scipy.stats.zscore() on only the numeric columns. Any NaNs detected in the numeric columns are temporarily
treated in such a way that they don't trigger an outlier removal. We fill the Nans with a 0 for comparison.
Create a boolean mask: For each row, check if any numeric column's absolute Z-score is <= z_threshold. Rows that fail, those that have at least
one numeric value beyond the threshold, get dropped.
Return a filtered DataFrame: We use .loc[mask] to keep only rows that passed the Z-score test and return a fresh copy
### 3.3. Data Preprocessing
Data preprocessing is the process of transforming raw data into a usable format for analysis or modeling. It involves steps like encoding categorical
variables and scaling features. Proper preprocessing ensures that the data is accurate, consistent and suitable for machine learning algorithms.
In this section we will perform the following preprocessing steps: Label Encoding, One-Hot Encoding and Feature Scaling.
### 3.3.1. One-Hot Encoding
This is a technique used to convert categorical variables into a numerical format. It creates binary (0 or 1) columns for each category, indicating the
presence of a category in a given observation. This method allows machine learning models to process categorical data without assuming any
ordinal relationship
### 3.3.2. Label Encoding
Label Encoding is a technique used to convert categorical text data into numerical values. Each unique category is assigned an integer label. In this
section, we will use label encoding on the Churn target variable to encode it to 0 (False) and 1 (True).
### 3.3.3. Data Scaling
This is the process of transforming features so they fall within a similar range. It ensures that no feature dominates another due to its magnitude,
which helps improve model performance and convergence.
In this section, we will implement Min-Max scaling in order to normalize the features to a fixed range of -1 to 1. This will ensure that each feature
contributes equally during model training. This step will be taken in the modeling phase, when we define our X and y variables.
## 4. Modelling
In this section, we will build a prediction model that can predict customer churn based on the features in our dataset. This stage is very critical as it
will help us in knowing the actual customers who churn in the company, and will guide in providing key insights that will necessitate the retention of
customers in the company.
In order to build a robust, effective model, we will train and evaluate five different models, and pick the model that will portray the highest
performance on unseen data. These 5 models include:
Logistic Regression: This will be our baseline model
Decision Tree
Random Forest
K_nearest Neighbor (KNN)
Gradient Boosting Classifier
In evaluating our models, we will use the recall and ROC-AUC metrics to monitor model performance.
First, we will define our X and y variables. The X variable will represent the features in the dataset, while the y variable will represent the target.
Now that we have defined our X and y variables, we can split the data into train and test sets. We will use an 80/20 split, implying that 80% of the
data will go to training, and 20% of the data will go to testing.
We will also Scale the X features using a MinMaxScaler() function. This is because this function scales the numerical values to a range of (0,1),
which is ideal for models such as K-Nearest Neighbors and Logistic Regression models.
### 4.1. Class Imbalance
From our earlier analysis, we observed that the target variable has a high class imbalance. We can show this again using the y_train variable:
The 0 class has a count of 2181, which is about 86.04%, while the 1 class has a count of 354, which is about 13.97%. This shows a very huge
imbalance in the two classes in the target variable. To address this, we will use a technique known as SMOTE(Synthetic Minority Over-Sampling
Technique). This technique is an oversampling method that generates synthetic samples from the minority class by interpolating between existing
minority class samples.
SMOTE works by selecting examples from the minority class, and generating examples by choosing a random point along the line between a
sample and one of its nearest neighbors from the minority class. These new synthetic samples are then used to balance the dataset, which helps ML
models to learn better and avoid bias toward the majority class.
In this section, we will apply SMOTE to our train dataset to obtained a balanced pair of training data.
Now that we have balanced the dataset, we can proceed to the actual modeling. We will start with the LogisticRegression() as our baseline model,
and build other models sequentially.
### 4.2. Logistic Regression.
Logistic Regression model accuracy: 0.7823343848580442
<img width="407" height="348" alt="image" src="https://github.com/user-attachments/assets/13da4899-cb4c-44e1-bae0-dd5261572802" />

### 4.3. Decision Tree Classifier
<img width="407" height="348" alt="image" src="https://github.com/user-attachments/assets/40f13388-f705-42bf-9c3b-c90e1d0be7a3" />

### 4.4. Random Forest Classifier
<img width="407" height="348" alt="image" src="https://github.com/user-attachments/assets/4630ff17-1919-43d0-a56d-10b53d6636e4" />

The Random Forest model has a recall score of about 0.73, which is lower than the recall scores of the Decision Tree and Logistic Regression
models. This implies that out of all the customers who churned from the company, the model was able to predict 73% of them correctly.
From the confusion matrix plot, the true positive and true negative have the highest values, implying that the model is making correct predictions
more frequently than incorrect predictions. This also shows that the model is not overfitting (i.e. performing poorly on unseen test data)
### 4.6. K-Nearest Neighbor Classifier
<img width="407" height="348" alt="image" src="https://github.com/user-attachments/assets/0b71fe1f-b774-4594-a8e9-3a5b4b8dd407" />

The K-Nearest Neighbor model has a recall score of about 0.4, which is much lower than the recall scores of the other models. This implies that out
of all the customers who churned from the company, the model was able to predict 40% of them correctly, which is a very low prediction rate, lower
than average (50%).
From the confusion matrix plot, however, the true positive and true negative have the highest values, implying that the model, despite having a lower
recall, is still making correct predictions more frequently than incorrect predictions.
### 4.7. Gradient Boosting Classifier
<img width="407" height="348" alt="image" src="https://github.com/user-attachments/assets/b188da52-0f08-46b9-9df4-ef5cf73ed370" />

The Gradient Boosting model has a recall score of about 0.81, which is higher than the recall scores of the other models. This implies that out of all
the customers who churned from the company, the model was able to predict 81% of them correctly, which is the best prediction rate thus far.
From the confusion matrix plot, the true positive and true negative have the highest values, implying that the model is making correct predictions
more frequently than incorrect predictions. This also shows that the model is not overfitting (i.e. performing poorly on unseen test data)
## 5. Model Evaluation
In this section, we will evaluate the model performance, and pick the best two, which we will perform hyperparameter tuning on, to select the best
performing model. As I had mentioned before, we will use recall and ROC-AUC Curve, to measure model performance.
### 5.1. ROC Curve
An ROC (Receiver Operating Characteristic) curve is a graphical representation used to evaluate the performance of a binary classification model. It
plots the True Positive Rate (TPR) (Recall) against the False Positive Rate (FPR) at various threshold settings. The curve illustrates how well a
model distinguishes between the two classes.
A model with a good performance will have a curve that bows toward the top-left corner of the plot. The Area Under the Curve (AUC) provides a
single number summary of performance - the closer the AUC is to 1, the better the model. An ROC curve is useful for comparing multiple models'
diagnostic ability.
We will plot the ROC curves of all the 6 models and compare the AUC scores.
<img width="708" height="492" alt="image" src="https://github.com/user-attachments/assets/e2d38154-5501-43a7-8417-1f6bde9e6bb7" />

From the ROC Curve plot, we can see that Random Forest, XGBoost, and Gradient Boosting have the highest AUC scores, with Gradient Boosting
having an AUC score of 0.912, and XGBoost having an AUC score of 0.910. K-Nearest Neighbors has the lowest AUC score of 0.651
### 5.2. Recall Score
Recall is a performance metric that represents the proportion of correctly predicted positive observations out of all actual positive observations. In
other terms:
Of all the actual positive items, how many did the model correctly identify?
A higher recall means very few false negatives, which implies that the model makes correct predictions more than false predictions, which is what
we are after anyway.
In this section, we will create a table that has each model with its recall score. This will help in determining the top performing models on unseen
data.
From the analysis on recall and ROC, we can conclude that the Gradient Boosting is the highest performing model, with a recall score of 0.807, and
an AUC score of 0.912, follwoed by the XGBoost classifier, with a recall score of 0.795, and an AUC score of 0.910. We will perform hyperparameter
tuning on these two models to try and improve the performance of each model, and again compare and see which model performs the best, based
on those two metrics.
But before we go to hyperparameter tuning, we can check the top important features in the Gradient Boosting classifier
<img width="708" height="420" alt="image" src="https://github.com/user-attachments/assets/44b5b016-42b4-4de3-9a34-dac9f54536d6" />

From the plot, the most important features for the model are Customer_Service_Calls, Total_Day_Charge, and International_Plan_Yes
## 6. Model Hyperparameter Tuning.
Model hyperparameter tuning in classification machine learning entails optimizing the external configuration settings (hyperparameters) that guide
the training process of a model. These parameters, such as the regularization strength in logistic regression, the number of neighbors in K-NN, or
the maximum depth in decision trees, are not learned from data, but significantly influence model performance. Proper tuning ensures that the model
generalizes well to unseen data, avoids overfitting, and achieves optimal performance metrics such as accuracy, recall, and f1 score.
Two common methods for hyperparameter tuning are Grid Search and Random Search:
Grid Search exhaustively tests every possible combination of specified hyperparameter values. For example, if tuning two hyperparameters with five
options each, grid search will evaluate all 25 combinations. While this ensures that the optimal combination (within the grid) is found, it becomes
computationally expensive and inefficient as the number of parameters or their value ranges increase.
Random Search on te other hand samples combinations of hyperparameters at random. While this doesn't guarantee evaluating all possibilities, it
often determines a good (or even optimal) combination with fewer iterations, especially when only a few hyperparameters significantly affect model
performance. This makes it faster and more scalable for high-dimensional search spaces.
Comparison
Grid Search is ideal when the hyperparameter space is small and well-defined. Random Search is better suited for large, complex, or lessunderstood search spaces In practice, Random Search if often more efficient, and can yield nearly optimal results in a fraction of the time, making it
more suitable for many real-world classification tasks
In this section, we will mainly use Random Search to obtain the best hyperparameters for training our model: Gradient Boosting
### 6.1. Gradient Boosting Hyperparameter Tuning
We will first tune the Gradient Boosting model. The main parameters we will look out for in the tuning process are:
Learning rate: Controls the contribution of each tree to the final prediction. A smaller value makes the model more robust, but requires more
estimators to achieve high performance. By default, learning_rate = 0.1
n_estimators: Defines the number of boosting iterations (trees) to be added. More estimators usually lead to better model performance, but also
increases the risk of overfitting. By default, n_estimators = 100
max_depth: Specified the maximum depth of each individual tree. Shallow trees might underfit, while deeper trees can overfit. It is crucial to find the
right depth. By default, max_depth = None
From the random search process, the best model hyperparameters we have obtained are:
n_estimators = 300 max_depth = 9 learning_rate = 0.08 We well update our Gradient Boosting model with these hyperparameters and monitor its
performance.
From the classification report, the recall score for the tuned Gradient Boosting model has remained the same from the first Gradient Boosting model,
which had a recall score of 0.81. This means that the model can predict 81% of the actual postive predictions correctly.
In addition, the number of true positives and negatives has increased in the tuned model, implying that the tuned model has a higher rate of making
true predictions compared to the initial model.
We can also compute the AUC score and plot the ROC curve for the tuned model to see whether there performance has improved.
From the ROC-AUC computation, the AUC score for the tuned Gradient Boosting classifier has increased from 0.912 to 0.921 which is a very
significant increase. This increases the model's capability to distinguish between the classes by 0.009, which is 0.9%.
## 7. 📝 Conclusion.
From our prediction modeling analysis, The XGBoost Classifier model had a recall score of 0.82, while the Gradient Boosting model achieved a
recall score of 0.81. However, the Gradient Boosting model had a higher AUC score of 0.921, while the XGBoost model had an AUC score of 0.911.
We were able to meet all our set objectives, which were to build a customer churn prediction model with a recall score of 0.8 and above, and to
identify the key features that contribute significantly to customer churn, which include Customer_Service_Calls, Total_Day_Charge, and
International_Plan. Due to the nature of the project and the prediction problem, I would recommend the XGBoost classifier model with a higher recall
for predicting customer churn rates at SyriaTel Telecommunication company
# 💼 Business Recommendations
1. Targeted Incentives for High-Churn Area Codes Customers in area codes 415 and 510 exhibit higher churn tendencies from my analysis.
Offering specialized discounts, loyalty rewards, or exclusive promotions in these regions can serve as an effective incentive to retain these
customers.
2. Enhance Customer Service Efficiency A high amount of customer service interactions with customers is seen to increase churn. Investing in
comprehensive training sessions for support stuff, and implmenting better issue/conflict resolution frameworks can significantly boost customer
satisfation, and in turn minimize the rate of customer churn.
3. State-Specific Retention Strategies States such as Texas, New Jersey, Maryland, Miami, and New York reported an above-average churn rate.
To mitigate this challenge, developing localized market efforts, personalized engagement strategies, and enhanced customer support in these
regions would aid in strengthening customer loyalty and retention.
4. Review and Optimize Call Rate Plans Majority of the customers who churn experience high day, evening, night and international call rates.
Reassessing the pricing model, and introducing more competitive/bundled plans could make the services more attractive, and cost-effective for
current users.
