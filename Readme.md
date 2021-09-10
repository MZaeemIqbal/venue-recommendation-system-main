#Introduction:
This module is about Venue Recommendation System , which gives the recommendations to user . Multiple Machine learning models used but Random Forest performs much better.
Almost 500 records used for developing the model , The major issue was dataset in this module , If we get more dataset then its accuracy
will definitely gonna be increased.

#Directory Structure Overview

There are multiple files in the directory , all the details of the files are shown below:

1) labelencoder.pkl: This file contains the label encoding of training dataset

2) Requirements.txt: All the requirements should be satisfied before running the project.

3) runner.py: This file contains the actual code of recommendation system

4) training_dataframe.csv: This is the CSV file which is going to be used for training the model

5) Original_DataFrame.csv: This is the actual CSV file (provided by Client)

6) record_appended_df.csv: This is the Dataframe which is used to cover the corner cases.

7) Random_Forest_Complete_Dataset.pkl: This is the weights file of model.


#Dataset Overview

Almost 500 records are present in Original_DataFrame.csv file. We have used only single column from all of them because 
that column (Categories) was highly correlated in predicting the venues . 


#How to run the project 

You have to pass the categories of particular users in the form of list containing single or multiple dictionaries.
##Valid Commands

Input:
~~~
python runner.py --cat "[{'S':'Vibes'}]"
~~~

If you specify the provided categories which are already present in the file , it will notify you and although , it'll give you results

Input :
~~~
python runner.py --cat "[{'S':'Workouts'}]"
~~~
Output :
~~~
Record already exists. Recommended Venue is Test Place Burnham2 , 0.14
~~~
If you specify the provided categories which are not present in the dataset , then you have to first insert that particular catgeories using below command and then run it.

Command: 
~~~
python runner.py --new_record "[{'S':'Working'}]" 'Paris'
~~~

##Invalid Commands

If you specify any invalid command it will not give you result as shown below :

Input :
~~~
python runner.py --cat "[{'S':}]"
~~~

Result:
~~~
[] , 0.0
~~~


#Commands Overview

~~~
python runner.py --cat "[{'S':'Workouts'}]"
~~~

Above command is used if provided categories are already present in the file and Please provide the
categories in string form . '--cat' is used because record is already present.

~~~
python runner.py --new_record "[{'S':'Restaurants'},{'S':'Rooftop'}]" 'Upper 14th'
~~~

Above command is used if provided categories are not present in the file and Please provide the 
categories in string form .'--new_record' is used for inserting the new record



#Machine Learning Model
Multiple machine learning models used for developing the recommendation system , but Random Forest gives much better results.

-> Training Accuracy : 96%

-> Testing Accuracy :  78%

#Requirements 

Run the following command to download  all the dependencies required for the project

~~~
pip install -r Requirements.txt
~~~

If you want to install any single particular package name  then run the following command

~~~
pip install <package-name>
~~~

