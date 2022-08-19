#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
from umap import UMAP
import umap
import hdbscan
import streamlit as st
pd.set_option("display.max_rows", None)


df_train = pd.read_csv('https://raw.githubusercontent.com/pfocordeiro/magnetite_app/main/MAG_INPUT.csv')
df_test = pd.read_csv('MAG_test.csv')

st.title("Magnetite Composition Clustering")
c29, c30, c31 = st.columns([1, 6, 1])

st.markdown("""
This app allows the user to plot their own magnetite chemistry data over the clustering results performed in Cordeiro et al X. Please notice that the UMAP-HDBSCAN clustering from Cordeiro et al was designed around LA-ICPMS data only and will likely not perform well with EPMA data, considering the differences in Ni, Zn and Ga detection limits.

Before uploading your data, please refer to the following guidelines:
1) Download the MAG_test.csv file below and place your data in that format.
2) Your test data must contain all eleven elements used in the original training (Al, Si, Ti, Vi, Cr, Mn, Co, Ni, Zn, Ga), even if your results are below detection limit. If any of these elements were not analysed, say Zn, the algorithm will consider Zn below detection limit and might deliver a biased embedded result.
3) Delete all below detection nomenclature (b.d., B.D.L., etc) from your csv file. The program understands that cells left blank mean below detection results and treats them as such in the calculation of the UMAP embedded space.
4) Only the 11 mentioned elements are used by the algorithms. Therefore, you can populate the Model, Type, Reference and Lithology columns as you wish. 
""")

with c30:

    uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        df_test = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(df_test)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [MAG_test.csv](https://raw.githubusercontent.com/pfocordeiro/magnetite_app/main/MAG_test.csv)
                """
        )

        st.stop()


# In[2]:


df_dropped = df_train.drop(
    ['Location', 'Type', 'Reference', 'Sample', 'Lithology', 'Model'], axis=1)

df_scaled = PowerTransformer(method="box-cox").fit_transform(df_dropped)

#UMAP model favoring local structures and clustered using HDBSCAN
df_umap1 = UMAP(random_state=1,
                n_components=2,
                n_neighbors=24,
                a=1.2,
                b=0.74,
                min_dist=0,
                metric='manhattan').fit_transform(df_scaled)

#UMAP model favoring global structures
df_umap2 = UMAP(random_state=1,
                n_components=2,
                n_neighbors=600,
                a=15,
                b=12,
                min_dist=0.8,
                metric='manhattan').fit_transform(df_scaled)

df_train['umap_1'] = df_umap1[:, 0]
df_train['umap_2'] = df_umap1[:, 1]

df_train['umap_3'] = df_umap2[:, 0]
df_train['umap_4'] = df_umap2[:, 1]


# In[3]:


predicted_labels = hdbscan.HDBSCAN(min_cluster_size=40,
                                   min_samples=55,
                                   ).fit_predict(df_umap1)
df_train['hdbscan'] = list(predicted_labels)


# In[4]:


df_train.Model.replace({'IOA':1, "Porphyry":2, "IOCG":3, "BIF":4, "Skarn":5, 'Fe-Ti-V':6}, inplace=True)

mapper = umap.UMAP(random_state=1, n_components=2,
                   n_neighbors=200,
               a=5, b=3,
               metric='manhattan').fit(df_scaled, y=(df_train.Model))

df_train.Model.replace({1:'IOA', 2:"Porphyry", 3:"IOCG", 4:"BIF", 5:"Skarn", 6:'Fe-Ti-V'}, inplace=True)


# In[5]:


df_calc_test = pd.concat([df_train, df_test])
df_test_dropped = df_calc_test.drop(['Location', 'Type', 'Reference', 'Sample', 'Lithology', 'Model', 'umap_1', 'umap_2','umap_3','umap_4','hdbscan'], axis=1)

df_test_dropped.fillna(0, inplace=True)

df_test_dropped.loc[df_test_dropped['Al'] < 10, "Al"] = 10
df_test_dropped.loc[df_test_dropped['Si'] < 2000, 'Si'] = 2000
df_test_dropped.loc[df_test_dropped['Ti'] < 11, 'Ti'] = 11
df_test_dropped.loc[df_test_dropped['V'] < 2, 'V'] = 2
df_test_dropped.loc[df_test_dropped['Cr'] < 6, 'Cr'] = 6
df_test_dropped.loc[df_test_dropped['Mn'] < 3, 'Mn'] = 3
df_test_dropped.loc[df_test_dropped['Co'] < 1, 'Co'] = 1
df_test_dropped.loc[df_test_dropped['Ni'] < 2, 'Ni'] = 2
df_test_dropped.loc[df_test_dropped['Zn'] < 6, 'Zn'] = 6
df_test_dropped.loc[df_test_dropped['Ga'] < 1, 'Ga'] = 1


# In[6]:


df_scaled_test = PowerTransformer(method="box-cox").fit_transform(df_test_dropped)

df_scaled_test = df_scaled_test[4262:]

train_embedding = mapper.transform(df_scaled)
test_embedding = mapper.transform(df_scaled_test)

df_train['umap_trainX'] = train_embedding[:, 0]
df_train['umap_trainY'] = train_embedding[:, 1]

df_test['umap_testX'] = test_embedding[:, 0]
df_test['umap_testY'] = test_embedding[:, 1]


# In[7]:


st.markdown("""
This plot represents the literature magnetite data after supervised embedding using the HDBSCAN clustering as labels. Therefore, you'll see 22 clusters with relative good separation between them. The idea of this treatment is to provide an opportunity for metric learning, where we use this new metric as a measure of distance between new unlabelled points in your dataset. 
""")

df_train_hdbscan = df_train[df_train.hdbscan != -1]

fig_2d = px.scatter(
    df_train_hdbscan,
    x='umap_trainX',
    y='umap_trainY',
    color=df_train_hdbscan.Model.astype('str'),
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference', 'hdbscan'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_xaxes(range=[-11, 20])
fig_2d.update_yaxes(range=[-10, 17])

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
), yaxis_title=None, xaxis_title=None)

fig_2d.show()
st.plotly_chart(fig_2d)


# In[8]:


st.markdown("""
This plot represents the embedding of your data in the metric created by literature data (the previous plot) and they are using the same UMAP coordinates, the X and Y axes. As in other UMAP embeddings discussed in Cordeiro et al X, proximity in the space is a good proxy for relatedness. By comparing both graphs, you can see how your datapoints compare to the HDBSCAN clusters from Cordeiro et al X.
""")

fig_2d = px.scatter(
    df_test,
    x=df_test['umap_testX'],
    y=df_test['umap_testY'],
    color='Location',
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference', 'Si'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_xaxes(range=[-11, 20])
fig_2d.update_yaxes(range=[-10, 17])

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
), yaxis_title=None, xaxis_title=None)

fig_2d.show()
st.plotly_chart(fig_2d)


# In[9]:


mapper = umap.UMAP(random_state=1, n_components=2,
                   n_neighbors=20,
               a=2, b=1.3,
               metric='manhattan').fit(df_scaled, y=(df_train.hdbscan))

train_embedding = mapper.transform(df_scaled)
test_embedding = mapper.transform(df_scaled_test)

df_train['umap_trainX_hdbscan'] = train_embedding[:, 0]
df_train['umap_trainY_hdbscan'] = train_embedding[:, 1]

df_test['umap_testX_hdbscan'] = test_embedding[:, 0]
df_test['umap_testY_hdbscan'] = test_embedding[:, 1]


# In[10]:


st.markdown("""
This plot represents the literature magnetite data after supervised embedding using the HDBSCAN clustering as labels. Therefore, you'll see 22 clusters with relative good separation between them. The idea of this treatment is to provide an opportunity for metric learning, where we use this new metric as a measure of distance between new unlabelled points in your dataset. 
""")

df_train = df_train[df_train.hdbscan != -1]

fig_2d = px.scatter(
    df_train,
    x=df_train['umap_trainX_hdbscan'],
    y=df_train['umap_trainY_hdbscan'],
    color='Model',
#     df_train.hdbscan.astype('str'),
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference', 'hdbscan'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_xaxes(range=[-17,16])
fig_2d.update_yaxes(range=[-12, 21])


fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
), yaxis_title=None, xaxis_title=None)

fig_2d.show()
st.plotly_chart(fig_2d)


# In[11]:


st.markdown("""
This plot represents the embedding of your data in the metric created by literature data (the previous plot) and they are using the same UMAP coordinates, the X and Y axes. As in other UMAP embeddings discussed in Cordeiro et al X, proximity in the space is a good proxy for relatedness. By comparing both graphs, you can see how your datapoints compare to the HDBSCAN clusters from Cordeiro et al X.
""")

fig_2d = px.scatter(
    df_test,
    x='umap_testX_hdbscan',
    y='umap_testY_hdbscan',
    color='Location',
    hover_data=['Model',"Location", "Type", 'Sample', 'Lithology', 'Reference', 'Si'],
    color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=500, template='simple_white')

fig_2d.update_xaxes(range=[-17,16])
fig_2d.update_yaxes(range=[-12, 21])

fig_2d.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1    
), yaxis_title=None, xaxis_title=None
                    )

fig_2d.show()
st.plotly_chart(fig_2d)


# In[15]:


st.markdown("""
Here you can save both figures at same scale in various formats that should be editable in vector editing software (such as Corel Draw and Adobe Illustrator softwares). Overlapping these graphs in vector software should allow you to compare your results against that of literature.
""")

fig, ax = plt.subplots(2,3, figsize=(22, 14))

sns.scatterplot(ax=ax[0,0], data=df_train, x='umap_trainX_hdbscan', y='umap_trainY_hdbscan', hue='hdbscan',
               palette='Paired',
               s=50)
ax[0,0].set_title('HDBSCAN-calculated UMAP embedding')
ax[0,0].set(xlim=(-17,17), ylim=(-12, 21), xlabel=None, ylabel=None)
ax[0,0].legend(loc=2, bbox_to_anchor=(0.99, 1))

sns.scatterplot(ax=ax[0,1], data=df_train, x='umap_trainX_hdbscan', y='umap_trainY_hdbscan', hue='Model',
               palette='Paired',
               s=50)
ax[0,1].set_title('HDBSCAN-calculated UMAP embedding')
ax[0,1].set(xlim=(-17,17), ylim=(-12, 21), xlabel=None, ylabel=None)

sns.scatterplot(ax=ax[0,2], data=df_test, x='umap_testX_hdbscan', y='umap_testY_hdbscan', hue='Location',
               palette='Set2',
               s=50)
ax[0,2].set_title('Test Data over HDBSCAN-calculated UMAP embedding')
ax[0,2].set(xlim=(-17,17), ylim=(-12, 21), xlabel=None, ylabel=None)

sns.scatterplot(ax=ax[1,0], data=df_train, x='umap_trainX', y='umap_trainY', hue='Model',
               palette='Paired',
               s=50)
ax[1,0].set_title('Deposit Model-calculated UMAP embedding')
ax[1,0].set(xlim=(-11,20), ylim=(-10, 17), xlabel=None, ylabel=None)

sns.scatterplot(ax=ax[1,1], data=df_test, x='umap_testX', y='umap_testY', hue='Location',
               palette='Set2',
               s=50)
ax[1,1].set_title('Test Data over Deposit Model-calculated UMAP embedding')
ax[1,1].set(xlim=(-11,20), ylim=(-10, 17), xlabel=None, ylabel=None)

fig.delaxes(ax[1,2])

st.pyplot(fig)

fn = 'embedding_unlabeled_data.svg'
plt.savefig(fn, dpi=300, bbox_inches='tight')
with open(fn, "rb") as img:
    btn = st.download_button(
        label="Download svg image",
        data=img,
        file_name=fn,
        mime="image/svg")

fg = 'embedding_unlabeled_data.eps'
plt.savefig(fg, dpi=300, bbox_inches='tight')
with open(fg, "rb") as img:
    btn = st.download_button(
        label="Download eps image",
        data=img,
        file_name=fg,
        mime="image/eps"
    )

st.markdown("""
Notes on interpretation:
1) Test Data over HDBSCAN-calculated UMAP embedding: Your datapoints will embed within or in in spatial proximity to one of the 22 clusters from the original study. So, if your datapoints embedded within cluster 3, it means that the algorithm classified these points as most similar to those from cluster 3, in relation to all other clusters.

2) Test Data over Deposit Model-calculated UMAP embedding: Your datapoints will embed within one or more than one deposit model. This embedding is a good way to classify your datapoints according to their deposit model. However, notice that porphyry deposits, for example, have a magnetite profile that is far different from the main "porphyry magnetite" and that is very similar to BIF magnetite compositions. 

3) Beyond the scope of the original paper, I've tested how magnetite from carbonatite and Ni-sulfide deposits would plot within this metric learning. In the trained space, most of their datapoints embedded either over the porphyry-dominant clusters or the Ti-V-dominant cluster, suggesting a strong magmatic affinity for all three deposit models, as expected.

""")


# In[ ]:




