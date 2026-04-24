#Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

#Initialize Page
st.set_page_config(page_title = 'Unsupervised Machine Learning App', layout = 'wide')

#Format Header
st.markdown('# Welcome to Clever Cluster 🔍')
st.markdown('In this app, you can use pre-loaded data or your own to explore unsupervised machine learning'
            'throught Principal Components Analysis, K-Means Clustering, and Hierarchical Clustering!')

#Initialize session variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'X_pca' not in st.session_state:
    st.session_state.X_pca = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'wcss' not in st.session_state:
    st.session_state.wcss = None
if 'kmeans' not in st.session_state:
    st.session_state.kmeans = None
if 'agg' not in st.session_state:
    st.session_state.agg = None

#Create tabs for app
tab1, tab2, tab3, tab4 = st.tabs(['Data', 'Select Features', 'Model', 'Results'])

#Construct first tab
with tab1:
    #File selection
    file = st.file_uploader('Upload a file of your own!', type = 'csv')
    data_selection = st.selectbox('Or Choose Built-In Data!', ['Beer Quality', 'Books'])

    #Backup data prep
    if data_selection == 'Beer Quality':
        backup_file = 'MLUnsupervisedApp/sample_data/beer.csv'
    else:
        backup_file = 'MLUnsupervisedApp/sample_data/books.csv'

    #Reading in data
    if st.session_state.data is None:
        if file:
            st.session_state.data = pd.read_csv(file)
        else:
            st.session_state.data = pd.read_csv(backup_file)
    #Display data
    st.dataframe(st.session_state.data, hide_index = True)

    a1, a2 = st.columns([1,1])
    with a1:
        #Convert categorical features to dummy variables
        select_dummy_cols = pd.DataFrame({'Column Name': st.session_state.data.columns, 
                                          'Data Type': [str(dtype) for dtype in st.session_state.data.dtypes],
                                          'Convert Column': [False] * len(st.session_state.data.columns)})
        
        edited_select_dummy_cols = st.data_editor(select_dummy_cols,
                                                  column_config = {'Convert Column': st.column_config.CheckboxColumn('Convert?', default=False),
                                                                   'Column Name': st.column_config.TextColumn('Column Name', disabled=True),
                                                                   'Data Type': st.column_config.TextColumn('Data Type', disabled=True)},
                                                  hide_index=True,
                                                  use_container_width=True)
        
        selected_dummy_cols = edited_select_dummy_cols[edited_select_dummy_cols['Convert Column']]['Column Name'].to_list()

        cat_cols = st.session_state.data[selected_dummy_cols].select_dtypes(include = ['object', 'category']).columns.tolist()
        num_cols = st.session_state.data[selected_dummy_cols].select_dtypes(include = ['number']).columns.tolist()

        if num_cols:
            st.warning(f'You selected numeric columns: {num_cols}. These will not be encoded!')
        else:
            st.success('All columns are categorical! You can confirm selections!')

        if st.button('Confirm Column Selection'):
            st.session_state.data = pd.get_dummies(st.session_state.data, columns = cat_cols, drop_first = True, dtype = int)

        if st.button('Reset Data'):
            if file:
                st.session_state.data = pd.read_csv(file)
            else:
                st.session_state.data = pd.read_csv(backup_file)

    with a2:
        #Histogram
        hist_col = st.selectbox('Column to look at:', st.session_state.data.columns)
        hist = px.histogram(st.session_state.data[hist_col])
        st.plotly_chart(hist) 

with tab2:
    b1, b2 = st.columns([1,1])
    with b1:
        #Select features for the model
        feature_cols = st.multiselect('Select feature columns', st.session_state.data.columns)
        st.session_state.X = st.session_state.data[feature_cols]

    with b2:
        st.write('Data Options')
        #Scaling option
        scale = st.checkbox('Scale data')

        if scale:
            scaler = StandardScaler()
            st.session_state.X = pd.DataFrame(scaler.fit_transform(st.session_state.X), columns = st.session_state.X.columns)


        #Handling missing data option
        missing = st.selectbox('How to handle missing data', ['Drop Observation', 'Mean Impute', 'Median Impute'])

        if missing == 'Drop Observation':
            st.session_state.X = st.session_state.X.dropna()
        elif missing == 'Mean Impute':
            for column in st.session_state.X.columns:
                st.session_state.X[column] = st.session_state.X[column].fillna(st.session_state.X[column].mean())
        elif missing == 'Median Impute':
            for column in st.session_state.X.columns:
                st.session_state.X[column] = st.session_state.X[column].fillna(st.session_state.X[column].median())
        
    st.dataframe(st.session_state.X, hide_index = True)

    st.subheader('Choose model to build:')
    st.session_state.model = st.selectbox('Select model', 
                                          ['Principal Components Analysis', 'K-Means Clustering', 'Hierarchical Clustering'])

def pca_model(data):
    components = st.slider('Choose number of components', min_value = 1, max_value = len(data.columns))

    if st.button('Run model!'):
        st.session_state.pca = PCA(n_components = components)
        st.session_state.X_pca = st.session_state.pca.fit_transform(data)
        st.write(st.session_state.pca.explained_variance_ratio_)

def pca_vis():
    st.metric(label = 'Explained Variance', value = sum(st.session_state.pca.explained_variance_ratio_))

    loadings_df = pd.DataFrame(st.session_state.pca.components_, 
                               columns = st.session_state.X.columns, 
                               index = [f'PC{i+1}' for i in range(st.session_state.pca.n_components_)])
    
    fig = px.imshow(loadings_df, 
                    labels = dict(x = "Variables", y = "Components", color = "Loading"), 
                    color_continuous_scale = 'blues')
    
    st.dataframe(loadings_df)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    c1, c2 = st.columns([1,1])

    with c1:
        st.plotly_chart(fig)

    with c2:
        pca_full = PCA(n_components = len(st.session_state.X.columns)).fit(st.session_state.X)

        explained = pca_full.explained_variance_ratio_ * 100
        components = np.arange(1, len(explained) + 1)
        cumulative = np.cumsum(explained)
   
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(x=components, y=explained, name="Individual Variance", marker_color='steelblue', text=[f"{v:.1f}%" for v in explained],
                        textposition='auto'), secondary_y=False)
    
        fig.add_trace(go.Scatter(x = components, y = cumulative, name = 'Cumulative Variance', mode = 'lines+markers', 
                             line = dict(color = 'crimson', width = 2), marker = dict(size = 8)),
                             secondary_y = True)
    
        fig.update_layout(title = dict(text = 'PCA: Variance Explained', xanchor = 'center', x = 0.5), 
                      hovermode = 'x unified', 
                      legend = dict(orientation = 'v', yanchor = 'middle', y = 0.5, xanchor = 'right', x = 0.8, bordercolor = 'black', borderwidth = 2))
    
        fig.update_yaxes(title_text = 'Individual Variance Explained (%)', secondary_y = False, color = 'steelblue')
        fig.update_yaxes(title_text = 'Cumulative Variance Explained (%)', secondary_y = True, color = 'crimson', range = [0,105])
        fig.update_xaxes(title_text = 'Principal Component', tickmode = 'array', tickvals = components, ticktext = [f"PC{i}" for i in components])

        st.plotly_chart(fig)

def k_cluster_model(data):
    label = st.multiselect('Do you have a target variable in mind?', st.session_state.data.columns, max_selections = 1)
    if label:
        st.session_state.y = st.session_state.data[label]
    else: st.session_state.y = None

    nclusters = st.slider('How many clusters would you liked to make?', min_value = 2, max_value = 15)

    if st.button('Run model!'):
        kmeans = KMeans(n_clusters = nclusters)
        kmeans.fit(st.session_state.X)
        st.session_state.kmeans = kmeans
        #st.rerun()

def k_cluster_vis():
    if 'kmeans' not in st.session_state or st.session_state.kmeans is None:
        st.warning('Model not fit yet!')
    elif st.session_state.y is not None:
        d1, d2, d3 = st.columns([1,1,1])
        with d1:
            st.metric(label = 'Silhouette Score', value = silhouette_score(st.session_state.X, st.session_state.kmeans.labels_))
        with d2:
            st.metric(label = 'Accuracy', value = accuracy_score(st.session_state.y, st.session_state.kmeans.predict(st.session_state.X)))
        with d3:
            st.metric(label = 'Within-Cluster Sum of Squares', value = st.session_state.kmeans.inertia_)

        pca_2d = PCA(n_components=2)
        pca_data = pca_2d.fit_transform(st.session_state.X)

        plot_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        plot_df['Cluster'] = st.session_state.kmeans.labels_.astype(str)
        plot_df['Actual'] = st.session_state.y.reset_index(drop=True).astype(str)

        fig = px.scatter(plot_df, x='PC1', y='PC2',color='Cluster',symbol='Actual',title='PCA: K-Means Clusters vs Real Labels',hover_data=['Cluster', 'Actual'])

        st.plotly_chart(fig)

    elif st.session_state.y is None:
        e1, e2 = st.columns([1,1])
        with e1:
            st.metric(label = 'Silhouette Score', value = silhouette_score(st.session_state.X, st.session_state.kmeans.labels_))
        with e2:
            st.metric(label = 'Within-Cluster Sum of Squares', value = st.session_state.kmeans.inertia_)
        
        pca_2d = PCA(n_components = 2)
        pca_data = pca_2d.fit_transform(st.session_state.X)

        plot_df = pd.DataFrame(pca_data, columns = ['PC1', 'PC2'])
        plot_df['Cluster'] = st.session_state.kmeans.labels_.astype(str)

        fig = px.scatter(plot_df, x = 'PC1', y = 'PC2', color = 'Cluster', title = 'PCA: K-Means Clusters', hover_data = 'Cluster')

        st.plotly_chart(fig)

    ks = range(2,16)

    wcss = []
    silhouette_scores = []

    for k in ks:
        km = KMeans(n_clusters = k, random_state = 1)
        km.fit(st.session_state.X)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(st.session_state.X, labels))

    fig1 = px.line(x = ks, y = wcss, markers = True)

    st.plotly_chart(fig1)

    fig2 = px.line(x = ks, y = silhouette_scores, markers = True)

    st.plotly_chart(fig2)

def hierarch_model(data):
    link_method = st.selectbox('Select linkage method', ['ward', 'complete', 'average', 'single'])

    Z = linkage(data, method = link_method)

    plt.figure(figsize=(20, 7))

    dendrogram(Z)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance")
    plt.xticks([])

    st.pyplot(plt)

    nclusters = st.slider('Using the dendrogram, select a number of clusters', min_value = 2, max_value = 15)

    if st.button('Run model!'):
        st.session_state.agg = AgglomerativeClustering(n_clusters = nclusters, linkage = link_method)
        #cluster_labels = agg.fit_predict(st.session_state.X)
        #st.rerun()
    else:
        st.warning('You have not run the model!')
    
def hierarch_vis():
    cluster_labels = st.session_state.agg.fit_predict(st.session_state.X)

    st.metric(label = 'Silhouette Score', value = silhouette_score(st.session_state.X, cluster_labels))

    results = pd.DataFrame(st.session_state.X.copy())
    results['Cluster'] = cluster_labels

    st.dataframe(results, hide_index = True)

    #st.dataframe(results['Cluster'].value_counts)

    pca_2d = PCA(n_components=2)
    pca_data = pca_2d.fit_transform(st.session_state.X)

    plot_df = pd.DataFrame(pca_data, columns = ['PC1', 'PC2'])
    plot_df['Cluster'] = st.session_state.agg.labels_.astype(str)

    fig = px.scatter(plot_df, x = 'PC1', y = 'PC2', color = 'Cluster', title = 'PCA: Hierarchical Clustering', hover_data = 'Cluster')

    st.plotly_chart(fig)

    ns = range(2,16)
    silhouette_scores = []

    for n in ns:
        labels  = AgglomerativeClustering(n_clusters = n, linkage = 'ward').fit_predict(st.session_state.X)
        score = silhouette_score(st.session_state.X, labels)
        silhouette_scores.append(score)

    fig = px.line(x = ns, y = silhouette_scores, markers = True)

    st.plotly_chart(fig)

with tab3:
    if st.session_state.model == 'Principal Components Analysis' and st.session_state.pca is not None:
        pca_model(st.session_state.X)
    elif st.session_state.model == 'K-Means Clustering' and st.session_state.kmeans is not None:
        k_cluster_model(st.session_state.X)
    elif st.session_state.model == 'Hierarchical Clustering' and st.session_state.agg is not None:
        hierarch_model(st.session_state.X)
    else:
        st.info('Complete previous steps first!')

with tab4:
    if st.session_state.model == 'Principal Components Analysis' and st.session_state.pca is not None:
        pca_vis()
    elif st.session_state.model == 'K-Means Clustering' and st.session_state.kmeans is not None:
        k_cluster_vis()
    elif st.session_state.model == 'Hierarchical Clustering' and st.session_state.agg is not None:
        hierarch_vis()
    else:
        st.info('Complete previous steps first!')