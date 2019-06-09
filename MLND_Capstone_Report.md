# Machine Learning Engineer Nanodegree
## Capstone Project : Predicting Budget Cost Codes Based on Purchase Order Information
Daniel Kelly  
Dan_kelly@telus.net  
June 8th, 2019

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

I currently work for a Construction Management company and one critical function of a Construction Management company is creating and tracking the budget of a construction project during the entire lifecycle of the construction project. 


During the construction of the project, the budget information may be used by developers/clients to track the performance of the architect, construction management team, and subcontractors. It can also be used to identify when a project may go over budget and require additional financing from their lender.

By breaking down the project budget into detailed sections like concrete, appliances, windows, etc. you can identify potential performance issues with individual subcontractors, products, or design elements and take steps to correct or mitigate them moving forward.

Additionally, when a project is complete you can then use the information in the finished budget to estimate the costs of new projects. Again, by breaking the budget into more granular sections, you can get more detailed information on what the costs of new projects are likely to be and how changes to the design or materials may affect that cost.

As a result of these use cases, it is important then to have accurate cost data and ensure that all costs are allocated to their respective budget sections. To do this, you use numerical Cost Codes that are unique and mutually exclusive. One numerical cost code is defined for each line in the budget. Every cost that arises during the construction of a project, from concrete orders to safety equipment, is allocated to a cost code within the project budget. 

For consistency, an organization will define a master list of every possible cost code that can be used in a budget, then a project will use only a subset of these codes that are relevant to the type project.

The projects that my company manages range in budget from between $1,000,000 and $200,000,000 with between 100 and 900 cost codes depending on their size and scope.

The project that I am proposing is to predict the budget category (Cost Code) that is associated with purchase orders created during a construction project. 
This project requires evaluating both text description and accompanying numerical data of a purchase order item to predict its category. This classification problem is similar to classifying bank transactions, this paper reviews some techniques for addressing this problem: [Automatic Classification of Bank
Transactions](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2456871/17699_FULLTEXT.pdf?sequence=1&isAllowed=y)

*Source: Olav Eirik Ek Folkestad, Erlend Emil Nøtsund Vollset. "Automatic Classification of Bank Transactions." June 2017. Norwegian University of Science and Technology Department of Computer Science.*

This paper, however, did not include any numerical transaction data in the prediction, nor did it explore using an ensemble of multiple models as I will.


### Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

Currently, Project Coordinators on the construction site must review and assign a cost code to each item on a purchase order so it can be accounted for in the correct location of the construction project's budget. The Project Manager then reviews and confirms or modifies the correct cost code is selected for an item before committing it to the budget.

This is a time consuming and therefore expensive process for both the Project Coordinator and Project Manager that must be completed manually at least once per month for every purchase order item before the supplier of the purchase order can be paid and the costs of the project can be entered to the budget. A completed project will have between 1,000 and 10,000 purchase order items that will have needed to be manually coded.

This process cannot feasibly be accomplished by a one-to-one mapping of products to cost codes because products are continuously added or changed, different projects may use different vendors, vendors are continuously being added, one product may be associated with different cost codes depending on how it is used, and many other factors.

I propose that one way to reduce the amount of time and the expense of choosing the correct cost code for an cost is to implement a predictive model that will use the data that is present on a purchase order (The vendor name, product description, cost, etc.) to predict what the associated cost code should be for each item on a purchase order.

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

This is a supervised classification problem where we are trying to predict what cost code a purchase order item belongs to.
I believe that the best evaluation for this metric is an F1-Score. I based this decision on my findings in the data exploration workbook. 

The conclusion of my findings is that the dataset is very unbalanced. The most used cost code appears 4,157 times, whereas the mean usage of a cost code is 97 times with a median of 5.5. This means that there are a small number of codes that are used a large number of times, but most codes are used relatively rarely.  

This imbalance makes accuracy poor measure of performance because, as discussed in the benchmark model section, the model can achieve an accuracy of almost %20 by always predicting the most common cost code despite being an obviously poor model. F1-score is a better metric than accuracy.  
The formula for f1-score is:  
![Source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9](https://cdn-images-1.medium.com/max/800/1*T6kVUKxG_Z4V5Fm1UXhEIw.png)  
*Source: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9*

By using the f1-score we get a balance between precision and recall that better reflects the performance of the model when compared to accuracy.   

Fbeta-score is another metric that could be useful, however, this metric is used to weight either precision or recall higher than the other. This would be used if one metric was more important than the other. Eg. Recall is more important if the cost of a false negative is higher than the cost of a false positive. In the case of this model, since we are suggesting cost codes to an end-user, the cost of a false positive is the same as a false negative.  

As a result, F1-score is the metric I have used to evaluate the model's performance but with recall and precision individually for reference.

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

The [dataset](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/raw_data/PO_Dataset.csv) that I will use for this project is a CSV list of purchase order items that I have exported from the SQL server database of my company's ERP system and obtained permission to use. This file contains over 39,000 examples of purchase order items, their accompanying information and their corresponding cost codes. These purchase order items have been previously entered into the company's ERP system and had their cost codes selected manually over the course of over 5 years and several construction projects. This labelled data will be split up and used as the training, validation, and test sets for the model. 

As previously mentioned the raw export contains over 39,000 records, however through some data exploration, documented in [this workbook](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/PO_Dataset_Exploration.ipynb), I found that there were some records that would not be helpful for achieving an accurate model. For example, there were some items with negative cost amounts which would correspond with credits that we had received from a supplier for items. These records are not relevant to the purchase of new items.

The [clean dataset](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/clean_data/Clean_PO_Dataset.csv) that I created as a result of the data exploration and cleaning resulted in a file containing approximately 28,000 records. The dataset has both numerical data fields and text.

As noted in the data exploration workbook, this is an unbalanced dataset with a few cost codes being used much more than others. This creates a problem where just randomly splitting the dataset into training, test, and validation sets could introduce a significant amount of bias. I will mitigate this by using the stratifying feature of sklearn's train_test_split and specifying the Cost Code column. This will maintain the ratio of codes relative to each other in each of the datasets.

Because splitting the data will reduce the number of samples I have for training, I can use SMOTE (Synthetic Minority Oversampling TEchnique) to generate more data points based on the existing information, giving my model more data to train with.

![Smote](https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/smote.png)

*Source: https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets


There are 9 features in this dataset, plus the variable that I want to predict. I will be using 7 of the features:

![Dataset description screenshot](https://github.com/Daniel-M-Kelly/Udacity-MLND-Project/blob/master/Dataset%20Description.png)

- **Company #** This is used internally to identify which internal company the project is associated with. It is not relevant to the prediction and will not be used.  

- **Purchase Order** This is the purchase order number, the format is *project_number-###* where the project number is a unique number for each project, then the next 3 digits are the purchase order number. Each project starts at purchase order 001 and increments for each purchase order until the project is complete. This field could be useful if I strip off the project number and just look at the purchase order number. Low purchase order numbers would happen at the beginning of the project and higher numbers would be near the end of the project. This could give information on what stage the project is in when the purchase order is made, and subsequently, have some correlation with which cost code should be selected. 

- **Item** This feature defines which item this record was on the purchase order, as purchase orders may contain multiple products. It is not useful for this prediction.

- **Vendor** This is the name of the vendor who sold the item on the purchase order. I will use one-hot encoding for this nominal data.

- **Description** This is a text field that contains the description of the item purchased. This should be the most important feature in the data. I will have to do further research, but I will either convert this field to a vector using [word2vec](https://skymind.ai/wiki/word2vec) or I will create a separate model to predict the cost code based only on this field, then feed that prediction into another model that includes the other data.

- **Unit of Measure** This feature tells us how the units of each item are measured. Most commonly LS - lump sum or EA - each. I will one-hot encode this feature.

- **Units** How many of these items were ordered. I will use this feature without any modification

- **Unit Cost** How much each unit ordered costs. I will use this feature without any modification

- **Cost** This is the total cost of the line item (Units * Unit Cost). I will use this feature without any modification

- **Cost Code** This is the variable that my model will predict and will be used for training.



### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


Currently, the processes for selecting cost codes for a purchase order items is entirely manual. We do not have statistics for how accurate the initial cost coding is, or how often Project Managers change cost codes when they are reviewing them.
Additionally, we also have no documented data on relationships between the information on a purchase order and the cost codes. 
Therefore, without any additional data on what the correlation is between a cost code and the information in the purchase order, I believe that the most relevant benchmark model would be to use a model that always predicts the most commonly used cost code. After doing data exploration and clean up, the most commonly used cost code was used 4,157 times out of a total of 28,446 records so a model that always predicted this cost code would have an accuracy of almost 15%. 


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
