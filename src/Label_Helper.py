#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", help = "Input_Path", default = "../data/filtered.csv")
parser.add_argument("-o", "--output_path", help = "Output_Path", default = "../data/with_labels.csv")
args = parser.parse_args()


# In[49]:
df = pd.read_csv(args.input_path)
df = df.drop(columns=["Unnamed: 0"])
print(df.head())


# In[38]:


df_with_labels = pd.DataFrame({"Time_Visited":pd.Series(),"Title":pd.Series(),"URL":pd.Series(),"LABEL":pd.Series()})
df_with_labels = pd.DataFrame()


# In[46]:


for index, row in df.iterrows():
    print("Time Visited: ",row['TIME_VISITED'], " Title: ", row['TITLE'], " URL", row['URL'])
    inp = input(str("p: Productive ") + str("n: Non-Productive") + str(" s: Save "))
    if inp == "s" or inp == "S":
        df_with_labels.to_csv(args.output_path)
        break
    print(row['TIME_VISITED'],row['TITLE'],row['URL'])
    record = pd.DataFrame({"Time_Visited" : [row['TIME_VISITED']],"Title":[row['TITLE']], "URL": [row['URL']], "LABEL":[inp]})
    df_with_labels = df_with_labels.append(record, ignore_index=True)


# In[48]:


print(df_with_labels.head())


# In[ ]:




