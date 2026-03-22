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





