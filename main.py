#!/usr/bin/env python
# coding: utf-8

# ## Importing required Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv(r"https://raw.githubusercontent.com/gitanirudh/Fake_job_posting/main/fake_job_postings.csv")


# ## Understanding Dataset

# In[3]:


print("Dataset Shape:", df.shape)
print("Null Values:")
print(df.isnull().sum())


# ## Replacing Null values

# In[4]:


col = ['location', 'company_profile', 'department','description', 'requirements', 'benefits','salary_range','employment_type','required_experience','required_education','industry','function']
df[col] = df[col].fillna('NAN')
df.head()


# ## Dropping Un-required Data columns 
# Since the salary range column did not provide any viable information, we have decided to drop it.

# In[5]:


df = df.drop(['salary_range'], axis = 1)


# # Statistical Ideation

# In[6]:


#stats
print(df['fraudulent'].value_counts())
df.describe()


# In[7]:


#information about dataset Dtype, columns, null
df.info()


# # Standardizing Data

# In[8]:


#removing punctuations
def clean_col(dframe, columns):
    for column in columns:
        if column in dframe.columns:
            dframe[column] = (
                dframe[column]
                .str.replace(r'[^\w\s]', '', regex=True)
                .str.replace(' ', '', regex=False)
                .str.strip()
                .str.lower()
            )
    return dframe


columns = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type']

cleaned_df = clean_col(df, columns)
print(cleaned_df.head(5))


# # Imputing Missing Values

# In[9]:


def imputetion(dframe):
    # Replace placeholder strings with actual NaN values
    dframe = dframe.replace(['NAN', 'nan', 'NaN', 'None'], np.nan)

    # Fill missing values
    for column in dframe.columns:
        if dframe[column].dtype in ['float64', 'int64']:  # Numeric columns
            dframe[column] = dframe[column].fillna(dframe[column].mean())
        else:  # Categorical or non-numeric columns
            mode = dframe[column].mode().iloc[0] if not dframe[column].mode().empty else ''
            dframe[column] = dframe[column].fillna(mode)

    return dframe

# Apply the function to the DataFrame
imputed_df = imputetion(cleaned_df)

# Check for remaining null values to verify
print(imputed_df.isnull().sum())


# In[10]:


#EDA step2- categorical data and OHE

def convert_columns_to_categorical(dataframe, columns):

  for col in columns:
    if col in dataframe.columns:
      dataframe[col] = dataframe[col].astype('category')
    return dataframe

columns_to_convert = ['title', 'location', 'department', 'employment_type', 'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 'function']
df_converted = convert_columns_to_categorical(imputed_df, columns_to_convert)


print(df_converted.dtypes)
df_converted.head()


# In[11]:


#one hot encoding
def ohe(dataframe, columns):

    dataframe_encoded = pd.get_dummies(dataframe, columns=columns, drop_first=True)
    return dataframe_encoded

col_encode_ohe = ['title', 'location', 'department', 'employment_type', 'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 'function']
df_encoded_ohe = ohe(df_converted, col_encode_ohe)
df_encoded_ohe.head()
df_encoded_ohe['fraudulent'].unique()


# In[12]:


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns


# ## Handling outliers

# In[13]:


def handle_outliers(df, columns, target_column):
    for col in columns:
        if col == target_column:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(
            df[col] < lower_bound, lower_bound,
            np.where(df[col] > upper_bound, upper_bound, df[col])
        )
    return df

df_no_outliers = handle_outliers(df_encoded_ohe, numeric_cols, target_column='fraudulent')


# ## Splitting Dataset

# In[14]:


X = df_no_outliers.drop(columns=['fraudulent'])
y = df_no_outliers['fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training target:", y_train.shape)
print("Test target:", y_test.shape)


# ## Decision Tree

# In[15]:


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# tree_model = DecisionTreeClassifier()
# tree_model.fit(X_train, y_train)
# y_pred_tree = tree_model.predict(X_test)

# tree_accuracy = accuracy_score(y_test, y_pred_tree)
# print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")


# In[16]:


# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred_tree)

# # Plot confusion matrix heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraudulent", "Fraudulent"], yticklabels=["Non-Fraudulent", "Fraudulent"])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


# In[17]:


# from sklearn.metrics import roc_curve, auc

# # Assuming you have predicted probabilities
# y_pred_proba = tree_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random guess line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()


# ## Gradient Boosting

# In[19]:


# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score

# # Gradient Boosting
# gb_model = GradientBoostingClassifier()
# gb_model.fit(X_train, y_train)
# y_pred_gb = gb_model.predict(X_test)

# # Evaluate
# gb_accuracy = accuracy_score(y_test, y_pred_gb)
# print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")


# In[20]:


# cm_boost = confusion_matrix(y_test, y_pred_gb)

# # Plot Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_boost, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraudulent", "Fraudulent"], yticklabels=["Non-Fraudulent", "Fraudulent"])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix - Gradient Boosting')
# plt.show()


# In[21]:


# y_pred_boost = gb_model.predict_proba(X_test)[:, 1]

# fpr, tpr, thresholds = roc_curve(y_test, y_pred_boost)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'{roc_auc:.2f})')
# plt.xlabel('False Positive')
# plt.ylabel('True Positive')
# plt.title('ROC Curve - Gradient Boosting')
# plt.legend(loc='lower right')
# plt.grid()
# plt.show()


# ## Random Forest

# In[22]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)

forest_accuracy = accuracy_score(y_test, y_pred_forest)
print(f"Random Forest accuracy: {forest_accuracy:.4f}")


# In[23]:


# cm_random = confusion_matrix(y_test, y_pred_forest)

# # Plot Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_random, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraudulent", "Fraudulent"], yticklabels=["Non-Fraudulent", "Fraudulent"])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix - Random Forest')
# plt.show()


# In[24]:


# y_pred_random = forest_model.predict_proba(X_test)[:, 1]

# # Compute ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_random)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'{roc_auc:.2f}')
# plt.xlabel('False Positive')
# plt.ylabel('True Positive')
# plt.title('ROC Curve - Random Forest')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()


# ## Logistic Regression

# In[25]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# log_model = LogisticRegression(max_iter=1000, random_state=42)
# log_model.fit(X_train, y_train)

# y_pred = log_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# In[26]:


# cm_logit = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_logit, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Fraudulent", "Fraudulent"], yticklabels=["Non-Fraudulent", "Fraudulent"])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()


# ## Support Vector Machines

# In[ ]:


# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score

# model_smv = LinearSVC(max_iter=10000, random_state=42)
# model_smv.fit(X_train, y_train)

# y_pred = model_smv.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)


# In[ ]:


# cm_svm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 4))
# sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pos', 'Neg'], yticklabels=['Pos', 'Neg'])
# plt.title('Support Vector machine - confusion matrix')
# plt.ylabel('Predicted')
# plt.xlabel('Actual')
# plt.show()

# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Assuming forest_model is your RandomForestClassifier
# joblib.dump(forest_model, "forest_model.pkl")  # Save the model
# # Load the saved model
# loaded_model = joblib.load("forest_model.pkl")
