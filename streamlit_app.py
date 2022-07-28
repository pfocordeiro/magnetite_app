#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from umap import UMAP
import umap
import hdbscan
import streamlit as st

df_pre = pd.read_csv('MAG_INPUT.csv')

st.title("Magnetite Composition Clustering")
c29, c30, c31 = st.columns([1, 6, 1])

with c30:

    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [MAG_test.csv](https://raw.githubusercontent.com/pfocordeiro/magnetite_app/main/MAG_test.csv)
                """
        )

        st.stop()


# In[2]:


df_dropped = df_pre.drop(['Location',
 'Type',
 'Reference',
 'Sample', 'Lithology', 'Model', 'Fe'],
                     axis=1)

df_scaled = QuantileTransformer(output_distribution='uniform',
                                n_quantiles=2000).fit_transform(df_dropped)


# In[3]:


df_umap = UMAP(random_state=1, n_components=2, n_neighbors=20,
               min_dist=0.1,
               a=10, b=0.8,
               metric='manhattan').fit_transform(df_scaled)

df_pre['umap_1'] = df_umap[:, 0]
df_pre['umap_2'] = df_umap[:, 1]
df_pre = df_pre.drop(columns=['Unnamed: 0'])


# In[4]:


fig_2d = px.scatter(
    df_pre,
    x='umap_1',
    y='umap_2',
    color=df_pre.Model,
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig_2d.show()
# fig_2d.write_html("1_interactive_model.html")


# In[5]:


predicted_labels = hdbscan.HDBSCAN(min_cluster_size=40,
                                   min_samples=55,
                                   ).fit_predict(df_umap)
df_pre['hdbscan'] = list(predicted_labels)


# In[6]:


df = df_pre[df_pre.hdbscan != -1]

fig_2d = px.scatter(
    df,
    x='umap_1',
    y='umap_2',
    color=df.hdbscan.astype('str'),
    hover_data=["Location", "Type", 'Sample', 'Lithology', 'Reference', 'hdbscan'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig_2d.show()
# fig_2d.write_html("1_interactive_hdbscan.html")


# In[7]:


df_umap = UMAP(random_state=1, n_components=2, n_neighbors=100,
               min_dist=0.8,
               a=12, b=11,
               metric='manhattan').fit_transform(df_scaled)

df_pre['umap_3'] = df_umap[:, 0]
df_pre['umap_4'] = df_umap[:, 1]


# In[8]:


mapper = umap.UMAP(random_state=1, n_components=2, n_neighbors=20,
               min_dist=0.1,
               a=10, b=2,
               metric='manhattan').fit(df_scaled, y=df_pre.hdbscan)


# In[17]:


df_test = pd.read_csv(uploaded_file#'MAG_Test.csv')
df_dropped = df_test.drop(['Location', 'Type', 'Reference', 'Sample', 'Lithology', 'Model', 'Fe'], axis=1)

df_dropped.fillna(0, inplace=True)

df_dropped.loc[df_dropped['Al'] < 10, "Al"] = 10
df_dropped.loc[df_dropped['Si'] < 2000, 'Si'] = 2000
df_dropped.loc[df_dropped['Ti'] < 11, 'Ti'] = 11
df_dropped.loc[df_dropped['V'] < 2, 'V'] = 2
df_dropped.loc[df_dropped['Cr'] < 6, 'Cr'] = 6
df_dropped.loc[df_dropped['Mn'] < 3, 'Mn'] = 3
df_dropped.loc[df_dropped['Co'] < 1, 'Co'] = 1
df_dropped.loc[df_dropped['Ni'] < 2, 'Ni'] = 2
df_dropped.loc[df_dropped['Zn'] < 6, 'Zn'] = 6
df_dropped.loc[df_dropped['Ga'] < 1, 'Ga'] = 1

df_scaled_test = QuantileTransformer(output_distribution='uniform', n_quantiles=50).fit_transform(df_dropped)


# In[18]:


train_embedding = mapper.transform(df_scaled)
test_embedding = mapper.transform(df_scaled_test)

df_pre['umap_trainX'] = train_embedding[:, 0]
df_pre['umap_trainY'] = train_embedding[:, 1]

df_test['umap_testX'] = test_embedding[:, 0]
df_test['umap_testY'] = test_embedding[:, 1]


# In[19]:


df_pre_train = df_pre[df_pre.hdbscan != -1]

fig_2d = px.scatter(
    df_pre_train,
    x=df_pre_train['umap_trainX'],
    y=df_pre_train['umap_trainY'],
    color=df_pre_train.Model.astype('str'),
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference', 'hdbscan'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig_2d.show()
# fig_2d.write_html("1_interactive_global_model.html")


# In[20]:


fig_2d = px.scatter(
    df_test,
    x=df_test['umap_testX'],
    y=df_test['umap_testY'],
    color=df_test.Model,
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig_2d.show()
# fig_2d.write_html("1_interactive_global_model.html")


# In[21]:


fig, ax = plt.subplots(2, figsize=(8, 14))

sns.scatterplot(ax=ax[0], data=df_pre_train, x='umap_trainX', y='umap_trainY', hue='Model',
               palette='Paired',
               s=50, alpha=0.05)
ax[0].set(ylim=(-6, 14))
ax[0].set(xlim=(-2, 15))

sns.scatterplot(ax=ax[1], data=df_test, x='umap_testX', y='umap_testY', hue='Location',
               palette='Set2',
               s=50)
ax[1].set(ylim=(-6, 14))
ax[1].set(xlim=(-2, 15))

plt.tight_layout()


plt.savefig("3_embedding_unlabeled_data.svg", dpi=300, bbox_inches='tight')


# In[ ]:




