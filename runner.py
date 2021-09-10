import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,f1_score
import joblib
import warnings
import os
import glob
import copy
import argparse
import traceback
import pickle
import regex
import re
warnings.filterwarnings("ignore")

def preprocessing(df):  
    df['Events']= df['itemTitle (S)'].where(df['itemType (S)'] == 'EVENT')
    df['Venue'] = df['itemTitle (S)'].where(df['itemType (S)'] == 'ENTITY')
    df=df.drop(['Events','itemType (S)','itemTitle (S)','createdAt (S)','updatedAt (S)','__typename (S)','type (S)','city (S)','itemId (S)','owner (S)','userId (S)','id (S)'],axis=1)
    df.replace('[]', np.nan, inplace=True)
    df.dropna(inplace=True)
    # df.drop_duplicates(inplace=True)
    df = df.sample(frac = 1).reset_index(drop=True)
    return df

def Convert_Categories_Into_List(df):
    new_df = copy.deepcopy(df)
    new_dict = {}
    k=0
    for index, row in new_df.iterrows():
        if type(row['categories (L)'])==str:
            for i in eval(row['categories (L)']):
                new_dict[i['S']]=''
        else:
            for i in row['categories (L)']:
                new_dict[i['S']]=''
    new_dict = {k:i for i,(k,v) in enumerate(new_dict.items())}

    for index, row in new_df.iterrows():
        for cat in eval(row['categories (L)']):
            row["categories (L)"] = row["categories (L)"].replace(cat['S'],str(cat['S']),1)
    for k,v in new_df.iterrows():
        v['categories (L)'] = [i['S'] for i in eval(v['categories (L)'])]
    return new_df,new_dict

def Encoding_DataFrame(df,new_dict):
    labelencoder = LabelEncoder()
    columns_list = sorted(list(new_dict))
    columns_list.append('Venue')
    new_df = pd.DataFrame(columns=columns_list)
    for k,v in df.iterrows():
        data_list =[]
        for c in columns_list:
            if c =='Venue':
                break
            if c in v['categories (L)']:
                data_list.append(1)
            else:
                data_list.append(0)
        data_list.append(v['Venue'])
        new_df.loc[k] = data_list
    new_df['Venue']=labelencoder.fit_transform(new_df['Venue'])
    return new_df,labelencoder

def Train_Test_Split(encoded_df):
    data = encoded_df.loc[:, encoded_df.columns != 'Venue']
    labels = encoded_df.iloc[:,-1:]
    train_data,test_data,train_labels,test_labels=train_test_split(data,labels,random_state=42,test_size=0.20)
    return train_data,test_data,train_labels,test_labels

def Complete_Dataset(new_df):
    data = new_df.loc[:, new_df.columns != 'Venue']
    labels = new_df.iloc[:,-1:]
    return data,labels

def Check_Record(dff,find_list,venue):
    dff,new_dict=Convert_Categories_Into_List(dff)
    new_list=[[" ".join(j.split()) for j in i] for i in dff['categories (L)']]
    dff['Venue'] = dff['Venue'].str.strip()
    find_list=[list(dicx.values())[0] for dicx in eval(find_list)]
    find_list = [" ".join(w.capitalize() for w in i.split()) for i in find_list]
    dff=list(dff.itertuples(index=False,name=None))	
    venue=" ".join(venue.split())
    new_find_list=[" ".join(i.split()) for i in find_list]
    check_input=(new_find_list,venue)
    if check_input in dff:
        return True
    else:
        return False

def RandomForest(train_data,train_labels,test_data,test_labels):
    global model
    model = RandomForestClassifier(n_estimators=100,max_depth=10)
    model.fit(train_data,train_labels)
    predicted_labels = model.predict(test_data)

    global RFF_train_accuracy
    global RFF_test_accuracy

    RFF_train_accuracy=model.score(train_data,train_labels)
    RFF_test_accuracy=model.score(test_data,test_labels)

    print("Model Score of Random Forest on training data",RFF_train_accuracy)
    print("Model Score of Random Forest on testing data",RFF_test_accuracy)
    
    joblib.dump(model,'Random_Forest_Model_Complete_Dataset.pkl')

    precision = precision_score(test_labels,predicted_labels,average='micro')
    recall = recall_score(test_labels,predicted_labels,average='macro')
    f_score = f1_score(test_labels,predicted_labels,average='weighted')

    print('Model Precision of Random Forest is',precision)
    print('Model Recall of Random Forest is',recall)
    print('Model F1 Score of Random Forest is',f_score)

def RandomForest_Complete_Dataset_Training(data,labels):
    global model
    model = RandomForestClassifier(n_estimators=100,max_depth=10)
    model.fit(data,labels)

    global RFF_accuracy
    RFF_accuracy=model.score(data,labels)
    complete_dataset_model_path=joblib.dump(model,'Random_Forest_Model_Complete_Dataset.pkl')
    return complete_dataset_model_path,RFF_accuracy

def encoding_category(category_data,df):
    columns_list = list(df.columns)
    encoded_cat = [0 for i in columns_list]
    for i in category_data:
        encoded_cat[columns_list.index(i)] = 1
    encoded_cat.pop(-1)
    return encoded_cat

def Retrain_Model(recieving_invalid_data_for_retraining,Venue,orig_df):
    
    print('Retraining ...')
    new_row = {'categories (L)':recieving_invalid_data_for_retraining,'Venue':Venue}
    if  os.path.isfile('record_appended_df.csv'):
        preprocessing_df=pd.read_csv('record_appended_df.csv')
    else:
        preprocessing_df = preprocessing(orig_df)
    record_appended_df = preprocessing_df.append(new_row,ignore_index=True) 
    record_appended_df.to_csv('record_appended_df.csv',index=False)
    record_appended_df,new_dict=Convert_Categories_Into_List(record_appended_df)
    training_dataframe,labelencoder=Encoding_DataFrame(record_appended_df,new_dict)
    invalid_and_valid_data,invalid_and_valid_data_labels=Complete_Dataset(training_dataframe)
    complete_dataset_model_path ,accuracy= RandomForest_Complete_Dataset_Training(invalid_and_valid_data,invalid_and_valid_data_labels)
    training_dataframe.to_csv('training_dataframe.csv',index=False)
    
    with open('labelencoder.pickle','wb') as f:
        pickle.dump(labelencoder, f)
    return complete_dataset_model_path,training_dataframe

def predict(orig_df,trained_df,data=None,new_categories=None,venue=None):
    try:
        category_data=[]
        if data:
            list_of_data=[list(dicx.values())[0] for dicx in eval(data)]
            for i in list_of_data:
                category_data.append(" ".join(w.capitalize() for w in i.split()))
            categories = list(trained_df.columns)
            Invalid_Data = [x for x in category_data if x not in categories]
            pass_invalid_data_for_retraining=[{list(i.keys())[0]:i[list(i.keys())[0]].capitalize()} for i in eval(data)]
            pass_invalid_data_for_retraining=str(pass_invalid_data_for_retraining)
            if len(Invalid_Data)==0:
                result=encoding_category(category_data,trained_df)
                cwd = os.getcwd()
                model_path=cwd+'/Random_Forest_Model_Complete_Dataset.pkl'
                load_model=joblib.load(model_path)
                predictions=load_model.predict([result])
                confidence_score= load_model.predict_proba([result])
                with open('labelencoder.pickle','rb') as f:
                    labelencoder = pickle.load(f)
                print("Recommended Venue for this type of category is",' '.join(labelencoder.inverse_transform(predictions)))
                print("Confidence Score : ", np.max(confidence_score))
                return (' '.join(labelencoder.inverse_transform(predictions))),np.max(confidence_score)
            else:
                print('The given record is not present in the dataset. Please use new record argument')
                return [],0.0
        else:
            list_of_data=[list(dicx.values())[0] for dicx in eval(new_categories)]
            for i in list_of_data:
                category_data.append(" ".join(w.capitalize() for w in i.split()))
            categories = list(trained_df.columns)
            # print(categories)
            Invalid_Data = [x for x in category_data if x not in categories]
            pass_invalid_data_for_retraining=[{list(i.keys())[0]:' '.join(w.capitalize() for w in i[list(i.keys())[0]].split())} for i in eval(new_categories)]
            pass_invalid_data_for_retraining=str(pass_invalid_data_for_retraining)

        if new_categories and venue:

            complete_dataset_model_path,training_dataframe = Retrain_Model(pass_invalid_data_for_retraining,venue,orig_df)
            result = encoding_category(Invalid_Data,training_dataframe)
            cwd = os.getcwd()
            model_path = cwd+'/Random_Forest_Model_Complete_Dataset.pkl'
            load_model = joblib.load(model_path)
            predictions = load_model.predict([result])
            confidence_score = load_model.predict_proba([result])
            with open('labelencoder.pickle','rb') as f:
                labelencoder = pickle.load(f)
            print("Recommended Venue for this type of category is",' '.join(labelencoder.inverse_transform(predictions)))
            print("Confidence Score : ", np.max(confidence_score))
            return (' '.join(labelencoder.inverse_transform(predictions))),np.max(confidence_score)
    except Exception as e:
        return [],0.0

def main(args):
    try:
        if args.cat:
            vals = ','.join([list(i.values())[0] for i in eval(args.cat)])
            if len(re.findall(r'[^A-Za-z0-9,]', vals))!=0:
                return [],0.0    
        elif args.new_record:
            vals = ','.join([list(i.values())[0] for i in eval(args.new_record[0][0])])
            if len(re.findall(r'[^A-Za-z0-9, ]', vals))!=0:
                return [],0.0
    except Exception as e:
        # print(e)
        return [],0.0

    cwd=os.getcwd()
    orig_df=pd.read_csv(cwd+'/Original_Dataframe.csv')
    dataframe=cwd+'/training_dataframe.csv'
    check_record_df=pd.read_csv(cwd+'/record_appended_df.csv')
    trained_df=pd.read_csv(dataframe)    
    
    if args.new_record:
        new_categories=args.new_record[0][0]
        new_venues=args.new_record[0][1]
        if Check_Record(check_record_df,new_categories,new_venues):
            print('Record already exists')
            result=predict(orig_df,trained_df,new_categories)
        else:           
            if len([True for i in eval(new_categories) if list(i.values())[0]==''])!=0:
                print('Invalid Data , Please provide appropriate data')
                return [],0.0            
            result=predict(orig_df,trained_df,new_categories=new_categories,venue=new_venues)
    else:    
        result=predict(orig_df,trained_df,args.cat)
    return result

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--cat', type=str, help='Enter the categories')
    parser.add_argument('--new_record',action='append',nargs=2,metavar=('category','venue'), type=str, help='Enter the new categories')
    args = parser.parse_args()
    main(args)
