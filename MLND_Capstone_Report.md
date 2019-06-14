# Machine Learning Engineer Nanodegree
## Capstone Project : Predicting Budget Cost Codes Based on Purchase Order Information
Daniel Kelly  
Dan_kelly@telus.net  
June 8th, 2019

## I. Definition

### Project Overview
I currently work for a Construction Management company and one critical function of a Construction Management company is creating and tracking the budget of a construction project during the entire lifecycle of the construction project. 


During the construction of the project, the budget information may be used by developers/clients to track the performance of the architect, construction management team, and subcontractors. It can also be used to identify when a project may go over budget and require additional financing from their lender.

By breaking down the project budget into detailed sections like concrete, appliances, windows, etc. you can identify potential performance issues with individual subcontractors, products, or design elements and take steps to correct or mitigate them moving forward.

Additionally, when a project is complete you can then use the information in the finished budget to estimate the costs of new projects. Again, by breaking the budget into more granular sections, you can get more detailed information on what the costs of new projects are likely to be and how changes to the design or materials may affect that cost.

As a result of these use cases, it is important then to have accurate cost data and ensure that all costs are allocated to their respective budget sections. To do this, you use numerical Cost Codes that are unique and mutually exclusive. One numerical cost code is defined for each line in the budget. Every cost that arises during the construction of a project, from concrete orders to safety equipment, is allocated to a cost code within the project budget. 

For consistency, an organization will define a master list of every possible cost code that can be used in a budget, then a project will use only a subset of these codes that are relevant to the type project.

The projects that my company manages range in budget from between $1,000,000 and $200,000,000 with between 100 and 900 cost codes depending on their size and scope.

In this project I created a machine learning model to predict the budget category (Cost Code) that is associated with purchase orders (POs) created during a construction project. 
This project requires evaluating both text description and accompanying numerical data of a PO  item to predict its category. This classification problem is similar to classifying bank transactions, this paper reviews some techniques for addressing this problem: [Automatic Classification of Bank
Transactions](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2456871/17699_FULLTEXT.pdf?sequence=1&isAllowed=y)

*Source: Olav Eirik Ek Folkestad, Erlend Emil Nøtsund Vollset. "Automatic Classification of Bank Transactions." June 2017. Norwegian University of Science and Technology Department of Computer Science.*

This paper, however, did not include any numerical transaction data in the prediction, nor did it explore using an ensemble of multiple models as I will.

The dataset that I used for this project consisted of an export of all the purchase order items from my company's ERP system SQL database. The raw CSV file can be found [here](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/raw_data/PO_Dataset.csv). I also used an export of the master list of cost codes and their description, found [here](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/raw_data/Code_Master_list.csv). This raw dataset contains over 39,000 records.


### Problem Statement

Currently, Project Coordinators on the construction site must review and assign a cost code to each item on a purchase order so it can be accounted for in the correct location of the construction project's budget. The Project Manager then reviews and confirms or modifies the correct cost code is selected for an item before committing it to the budget.

This is a time consuming and therefore expensive process for both the Project Coordinator and Project Manager that must be completed manually at least once per month for every purchase order item before the supplier of the purchase order can be paid and the costs of the project can be entered to the budget. A completed project will have between 1,000 and 10,000 purchase order items that will have needed to be manually coded.

This process cannot feasibly be accomplished by a one-to-one mapping of products to cost codes because products are continuously added or changed, different projects may use different vendors, vendors are continuously being added, one product may be associated with different cost codes depending on how it is used, and many other factors.

I propose that one way to reduce the amount of time and the expense of choosing the correct cost code for an cost is to implement a predictive model that will use the data that is present on a purchase order (The vendor name, product description, cost, etc.) to predict what the associated cost code should be used for each item on a purchase order.
This solution will use two models. The first model will be used to predict the associated cost code of an item based on the description of the item. Then this prediction made by the first model will be added as another feature to the existing categorical and numerical features (Unit Cost, Vendor name, etc.) The dataset including the new feature from the first model will then be fed into a second model optimized to make predictions on numerical and categorical data. Using two models for the two different types of data will allow me to select the best classifier for each type of data and creates a simple form of ensemble stacking.

The output of this solution will be a prediction of the correct cost code that the purchase order item should be associated with.


### Metrics

This is a supervised classification problem where we are trying to predict what cost code a purchase order item belongs to.
I believe that the best evaluation for this metric is an F1-Score. I based this decision on my findings in the data exploration workbook. 

The conclusion of my findings is that the dataset is very unbalanced. The most used cost code appears 4,157 times, whereas the mean usage of a cost code is 97 times with a median of 5.5. This means that there are a small number of codes that are used a large number of times, but most codes are used relatively rarely.  

This imbalance makes accuracy poor measure of performance because, as discussed in the benchmark model section, the model can achieve an accuracy of almost %15 by always predicting the most common cost code despite being an obviously poor model. This makes F1-score a better metric for hte performance of this model than accuracy.  
The formula for f1-score is:  
![Source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9](https://cdn-images-1.medium.com/max/800/1*T6kVUKxG_Z4V5Fm1UXhEIw.png)  
*Source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9*

By using the f1-score we get a balance between precision and recall that better reflects the performance of the model when compared to accuracy.   

Fbeta-score is another metric that could be useful, however, this metric is used to weight either precision or recall higher than the other. This would be used if one metric was more important than the other. Eg. Recall is more important if the cost of a false negative is higher than the cost of a false positive. In the case of this model, since we are suggesting cost codes to an end-user, the cost of a false positive is the same as a false negative.  

As a result, F1-score is the metric I have used to evaluate the models performance but with recall and precision individually for reference.

## II. Analysis

### Data Exploration

The [dataset](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/raw_data/PO_Dataset.csv) that I will use for this project is a CSV list of purchase order items that I have exported from the SQL server database of my company's ERP system and obtained permission to use. This file contains over 39,000 examples of purchase order items, their accompanying information and their corresponding cost codes. These purchase order items have been previously entered into the company's ERP system and had their cost codes selected manually over the course of over 5 years and several construction projects. This labelled data will be split up and used as the training, validation, and test sets for the model. 

As previously mentioned the raw export contains over 39,000 records, however through some data exploration, documented in [this workbook](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/PO_Dataset_Exploration.ipynb), I found that there were some records that would not be helpful for achieving an accurate model. For example, there were some items with negative cost amounts which would correspond with credits that we had received from a supplier for items. These records are not relevant to the purchase of new items.

As noted in the data exploration workbook, this is an unbalanced dataset with a few cost codes being used much more than others. This creates a problem where just randomly splitting the dataset into training, test, and validation sets could introduce a significant amount of bias. I will mitigate this by using the stratifying feature of sklearn's train_test_split and specifying the Cost Code column. This will maintain the ratio of codes relative to each other in each of the datasets.

Because splitting the data will reduce the number of samples I have for training, I can use SMOTE (Synthetic Minority Oversampling TEchnique) to generate more data points based on the existing information, giving my model more data to train with.

![Smote](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/smote.png)

*Source: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


There are 9 features in this dataset, plus the variable that I want to predict. I will be using 7 of the features:

![Dataset description screenshot](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/Dataset%20Description.png)

- **Company #** This is used internally to identify which internal company the project is associated with. It is not relevant to the prediction and will not be used.  

- **Purchase Order** This is the purchase order number, this is not relevant to the prediction.

- **Item** This feature defines which item this record was on the purchase order, as purchase orders may contain multiple products. It is not useful for this prediction.

- **Vendor** This is the name of the vendor who sold the item on the purchase order. I will use one-hot encoding for this nominal data.

- **Description** This is a text field that contains the description of the item purchased. This should be the most important feature in the data. 

- **Unit of Measure** This feature tells us how the units of each item are measured. Most commonly LS - lump sum or EA - each. I will one-hot encode this feature.

- **Units** How many of these items were ordered. I will use this feature after normalizing it. 

- **Unit Cost** How much each unit ordered costs. I will use this feature after normalizing it.

- **Cost** This is the total cost of the line item (Units * Unit Cost). I will use this feature after normalizing it.

- **Cost Code** This is the variable that my model will predict and will be used for training. There are 905 unique cost codes in the master list, however only 354 are used in the purchase orders in the dataset. 

This dataset is very unbalanced, the average number of times a cost code is used is 111, but the median is 6.5. This means that there are a few cost codes that are used many more times than the others. For example, 03-31-43 concrete material - above grade verticals is used 5570 times.


### Exploratory Visualization

The following graphic shows the ten most used cost codes.
  
<img src="https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Cost%20Code%20Counts.png" width="50%">
  
Furthermore, there are a small number of very high value POs, or POs with a large number of Units that skew the data.
The following table shows, for example, that the Units feature has a maximum value of over 100,000 while the 75th percentile is under 11. 
  
![Units and Costs](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Units%20and%20costs.png)
  
These POs are outliers and will be removed.

The following figure shows the correlation between the numerical and categorical features in the dataset.
  
![Correlation Matrix](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Correlation.png)
  
This shows us that the Vendor is the most clostly correlated variable with the cost code, followed by the overall cost of the item. Intuitively, the vendor having a high correlation with the cost code makes sense. For the most part, vendors each sell a certain type of product related to its function. For example a vendor called "Advanced Safety Supplies" sells mostly safety related equipment that would be budgeted to a "safety supplies" cost code. 

The cost and unit cost being closely correlated also makes sense, because the cost of a line item is just a multiple of the unit cost. 

What I found suprising was that the unit cost of an item was not very closely correlated to the cost code. I would have expected the unit cost to be very closely related to the particular item being purchased, which would then correspond to a particular cost code. It could be that product prices have changed over the years, or with different vendors. It's also possible that many products have similar prices, or simply that end users did not bother to put in the unit cost and just entered the total cost of the line item.

Looking at the text data in the Description feature, we can see that the majority of PO descriptions have between 2 and 6 words in them.
  
![Word Count](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Word%20Count.png)
  
And the most frequently used words are:
  
![Word Frequency](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Word%20Frequency.png)
  
Note that I did not remove stop-words from this dataset in exploration or in the training. This is because there is some research to suggest that removing stopwords can have a negative affect on classification performance. http://www.lrec-conf.org/proceedings/lrec2014/pdf/292_Paper.pdf 


### Algorithms and Techniques

---
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_
---

The first set of algorimths that I will test will be for predicting the cost code of the PO item solely on the description text information in the PO. So I will split out the description text of the training and test datasets into new dataframes.
Then, I will use a pipeline that uses the CountVectorizer and TFIDF feature extraction techniques. The processed text data will then be passed to either a SGDC Classifier, Logistic Regression classifier, or Multinomial Naive Bayes classifier. I chose these classifiers based on recommendations found on the internet. https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/

For all three parts of the pipeline I can use a gridsearch to adjust hyperparameter of the feature extractor or classifier.
After tuning the hyperparameters I will choose the pipeline and classifier that outputs the best F1-score based on the test data.
I will then use this model to predict a cost code for the training set data, and append the result to that dataset. I will also append the test prediction to the test dataset.

With this new dataset that includes the prediction from the previous model, I will then train two additional models, a Random Forest Classifier, and KNeigbors Classifier, with hyperparameters tuned with a gridsearch.

I will then use the SMOTE technique to attempt to minimize the affect of the imbalance in the dataset. SMOTE will synthetically generate more datapoints for the minority classes and increase the size of the training data set. SMOTE will not be applied to the testing dataset. I will then train a second set of Random Forest and KNeighbors classifiers witht he SMOTE enhanced dataset to compare how they perform to the original dataset.

The combination of algorithms that product the highest F1, recall, precision, and accuracy scores will be chosen as the solution to the problem.
    
### Benchmark

Currently, the processes for selecting cost codes for a purchase order items is entirely manual. We do not have statistics for how accurate the initial cost coding is, or how often Project Managers change cost codes when they are reviewing them.
Additionally, we also have no documented data on relationships between the information on a purchase order and the cost codes. 
Therefore, without any additional data on what the correlation is between a cost code and the information in the purchase order, I believe that the most relevant benchmark model would be to use a model that always predicts the most commonly used cost code. After doing data exploration and clean up, the most commonly used cost code was used 4,157 times out of a total of 28,446 records so a model that always predicted this cost code would have an accuracy of almost 15%. 


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
---
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_
---

The dataset that I used for this project required several data processing steps that I identified in the [Data Exploration](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/PO_Dataset_Exploration.ipynb) notebook, and implemented in the project notebook.

The first processing step was to convert the Units column from an object to a float. I'm not sure why ERP system would allow text values to be entered into the units field, which should only be the number of units of an item purchased.  

`#Convert the Units column to float  
df['Units'] = pd.to_numeric(df['Units'], errors='coerce').fillna(0)  
df['Units'] = df['Units'].astype('float64')`  

I then dropped any PO items with null values in any of the columns. There were only 21 of them, so it didn't make sense to include them.  

`df.dropna(inplace=True)`

  
Next, using a master list of valid cost codes exported from the system, I dropped any PO items that had an invalid cost code. This could possibly occur due to incorrect data entry or if a cost code was made invalid and is no-longer used in POs.  

`#Read in Master list of valid cost codes  
df_ml = pd.read_csv('raw_data/Code_Master_list.csv')  

#Drop rows where the cost code is not in the master list  
df = df[df['Cost Code'].isin(df_ml['Cost Code'])].dropna()`

Looking at the numerical fields, there were some negative values in the Units, Unit Cost, and Costs fields. These are likely related to credits back to the company, and are not relevant to predicting PO cost codes, so they needed to be removed.  

`
#Update dataset to exclude rows with Units, Unit Cost, or Costs that are negative.    
df = df[(df[['Units','Unit Cost','Cost']] >= 0).all(axis=1)]  
`

Next, looking at the description of the data in the Units, Unit Cost, and Costs fields, we can see that there are a few outliers with very large values that are skewing the data. So by dropping the the records in the top 10% of these fields, we get a more representational dataset.

`
#Create a new dataframe that takes only the 90th quartile of data from the 3 numerical columns.  
df_90 = df[df['Cost'] < df['Cost'].quantile(.90)]  
df_90 = df_90[df_90['Units'] < df_90['Units'].quantile(.90)]  
df_90 = df_90[df_90['Unit Cost'] < df_90['Unit Cost'].quantile(.90)]  
`

Finally, it is a best practice to scale numerical values between 1 and 0, so I used sklearns MinMaxScaler() to scale the Units, Unit Cost, and Costs features.

`
#It's a good practice to scale numerical data  
#Initialize a scaler, then apply it to the features  
scaler = MinMaxScaler()   
numerical = ['Units','Unit Cost','Cost']  

df_90[numerical] = scaler.fit_transform(df_90[numerical])  
`

We'll need cost codes with atleast 2 examples in the database to have atleast one example in both the training and testing datasets. So drop any codes with a count fewer than 2.

`
#When splitting for training and testing later, we'll need a minimum of 2 examples of each cost code.  
#Assign cost code to a variable  
df_count = df_90['Cost Code'].value_counts()  

#New dataframe only includes lines with cost codes with a count of 2 or greater  
df_90 = df_90[~df_90['Cost Code'].isin(df_count[df_count <= 2].index)]
`

Next, the categorical features Vendor and Unit of Measure need to be dealt with. I'll use one-hot-encoding for these features.  
`
#One Hot Encode categorical features  
categorical = ['Vendor', 'Unit of Measure']  
df_90 = pd.get_dummies(df_90, columns = categorical )  
`  
The target variable, "Cost Code" needs to be encoded as well. Label encoding makes sense here.  
`
#Numerically encode cost codes.  
le = LabelEncoder()  
cost_code = df_90['Cost Code']  
df_90['Cost Code Encoded'] = le.fit_transform(cost_code)  
`

Drop features that are irrelevant to the prediction.  
`
#drop features I won't be using
df_90 = df_90.drop(['Company #','Purchase Order', 'Item'], axis = 1)
`

Separate the target variable from the features that will be used to predict it.

`
#Separate the target variable from the features  
cost_codes = df['Cost Code Encoded']  
features = df.drop(['Cost Code','Cost Code Encoded'], axis=1)  
`

Now I split the data into the the training and test sets using sklearns train test split. 80% of the data will be in the training set, and 20% will be in the testing set. The stratify parameter will ensure that the same ratio of cost codes will be included in each set.
`
#Use sklearn train test split to split the data into training and testing sets.   
#Testing set is 20% of total dataset size.  
#Stratify the data so we don't introduce bias in the sets.  

X_train, X_test, y_train, y_test = train_test_split(features,  
                                                    cost_codes,  
                                                    test_size = 0.2,  
                                                    stratify = cost_codes  
                                                   )  
`

Lastly, for use in the two separate models, I need to extract the Description feature from the training and test sets so they can be used separately from the other features.

`
#Split X_train and X_test text Descriptions for use in sepearate model.  
X_train_desc = X_train['Description'].copy()  
X_train = X_train.drop('Description', axis=1)  

X_test_desc = X_test['Description'].copy()  
X_test = X_test.drop('Description', axis=1)  
`

With those steps, the data pre-processing is complete. Further processing and feature extraction of the text data in the description feature will be done in the pipeline described in the implementation section.

### Implementation
---
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_
---

### Refinement
---
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_
---

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
---
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_
---

### Justification

RF - Random Forest Classifier
KN - K Neighbors Classifer
RF_res - Random Forest Classifier with Smote enhanced Dataset
KN_res - K Neighbors Classifer with Smote enhanced Dataset
![Classifier Comparison](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Classifier_Comparison.png)

The figure above compares the accuracy, F1-score, precision, and recall of the final alogrithms that I used in my solution when run against the test set of data.

From this, we can see that the accuracy of all the models performed significantly better than the benchmark model's accuracy of 15%. 

The final solution of using the Logistic Regression algorithms output combined with a Random Forest classifier produced the best results when looking at the accuracy, precision, and recall metrics, and matched using the KNeighbors classifier for F1-Score. 

When taking into consideration the nature of the problem, which is to suggest a cost code to an end user who will simply accept or reject the suggestion. I think I can safely argue that my solution to the problem provides enough value to be considered to have solved the problem. In addition, due the nature of the data available in a PO there are some instances where there just is not enough information in the PO to more accurately predict the cost code, or the correct code to use may be subjective. 


## V. Conclusion

### Free-Form Visualization

The table below shows some examples of predictions from my model and the actual cost codes.  

![Example predictions](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/figures/Example%20Predictions%20.png)

I think this example demonstrate why getting a high accuracy of the prediction on this dataset is difficult. There are some items and descriptions that could apply to multiple cost codes. For example the Fuel Surchage on a concrete delevery would have the same description but could apply to any of several concrete related cost codes. As I mention later in the improvements section, if each line item is taken out of the context of the PO and evaluated by itself, there are instances where there is not enough information to predict which cost code an item belongs to. And again, the "Polarcon Accelerating - Bronze" is a product that is added to concrete to speed its curing time. This product could be used in multiple concrete related cost codes.
I think these items demonstrate that an above 50% accuracy rate for the model is in-fact impressive, and if it does not give the end-user the exact cost code to use, it suggests one that is close.

### Reflection

In summary, the solution that I arrived at for this problem involved the following:
* The first part of the solution was pre-processing the training data. Several of the features had extreme outliers in the data and had to be trimmed down, I also scaled the numerical features and converted the categorical features using one-hot-encoding. The data also contained codes that did not appear enough times to make a good prediction, so I dropped these instances. There were also some irrelevant columns that I dropped. Lastly, I label encoded the target variable.

* I then split the training at test data, making sure to stratify the split so that there was a representational portion of all the cost codes in both the training and test set. Since the dataset is very unbalanced, this helps to reduce and bias that could have been introduced if the data as split randomly.

* Next I extracted the description text and separated it from the numerical and categorical data and use a pipeline that extracts the features from the text data. The last step in the pipeline is training a logistic regression classifier to predict the cost code value based on the text features. 

* I then add the predicted cost code from the first model as a feature into the remaining categorical and numerical dataset. I then trained a random forest classifier to predict the final cost code based on this dataset. 

As I suspected, I found the most difficult part of this project was figuring out how to deal with text based data and numerical and categorical data. Most of the resources I found for dealing with text based data were sentiment analysis based and used only text information. For example, in this paper which addresses a similar problem - classifying products based on their description and other informaiton - they simply treated features that could be categorical, like brand, as text and included it as a word. http://cs229.stanford.edu/proj2011/LinShankar-Applying%20Machine%20Learning%20to%20Product%20Categorization.pdf 

Again, the most unexpected result was that the classifiers trained on the dataset that had been augmented using SMOTE performed worse than the classifiers trained on the base dataset. I beleive this was due to SMOTE causing overfitting. I noticed that the f1 score of the training set for the classifiers trained on the SMOTE dataset was much higher than the f1 score of the testing dataset indicating overfitting.

When considering this project I was originally hoping to achieve an accuracy close to 75%, and was slightly disapointed to only achieve ~50% accuracy. However when looking more closely at the data I think this is a good result. Some of the cost codes possibly overlap eachother in their use or are confusing and may be frequently miscoded by end-users. One example is the cost codes "01-52-22 field office supplies" and "01-52-23 field supplies" these codes are very similar and possibly mis-used. So taking this into context I think 50% is a good result, and if you consider the cost of suggesting an incorrect cost code is so low, I think that even at 50% accuracy, the prediction still has value. Lastly, my model significantly beats the benchmark model's accuracy of 15%.

### Improvement

There are a few areas where I think I could improve this project:

The first is the way that I handled stacking the two different models. I believe that it is possible to create a pipline and use feature unions to process the text and numeric and categorical data separately then pass them to a final classifier as shown in this post https://www.kaggle.com/metadist/work-like-a-pro-with-pipelines-and-feature-unions. By better using pipeline and stacking functionality I could improve the accuracy of the model. It would also make testing multiple classifiers easier, and tuning hyperparameters. However I did not have the time to fully research and understand this technique enough to be confident implementing it.

The second area that I think could be improved on is the second classifier that I used. My research showed that XGBoost is generally one of the highest performing classifiers. I did manage to get XGBoost working, however I found that the time it took to run was excessive and I could not properly tune its parameters. I believe the problem with using XGBoost is because of the number of features in the dataset that are created when one-hot-encoding the vendors feature. When I tried label encoding the vendors feature XGBoost ran at an acceptible speed, however all of the classifiers prediction performance dropped unacceptably low. With more time, I would have liked to get XGBoost performing better so I could fully evaluate its performance and possibly increase the accruacy of my model. Another option would have been to try using the LightGBM classifier which is similar to XGBoost and is also supposed to produce great results. This post has a good comparison of XGBoost and LightGBM. https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/

Finally, it may be possible to improve the accuracy of the predictions by taking into consideration other items on a PO. One PO may have several items on it that are related. For example, a concrete purchase order may have 3 items: 1 - the concrete material itself, 2 - a fuel surcharge for the delivery of the concrete, 3 - a disposal fee for leftover concrete. The first item, the concrete material, may be a specific mix that has information in its description that indicates it is for a concrete footing and, therefore, should be associated with cost code "03-31-40 concrete material footings". The other two items, the fuel surcharg and disposal fee are generic and could apply to any one of many concrete cost codes, however when taken into context with the other item on the PO should also go to the "03-31-40 concrete material footings" cost code. 

Overall I'm sure there is room for improvement of my final result, however this project demonstrated the proof of concept that a machine learning model can use the information in a purchase order to make useful predictions on what the cost code of a purchase order item should be.
