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
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

#Initialize Page
st.set_page_config(page_title = 'Unsupervised Machine Learning App', layout = 'wide')

#Format Header
st.markdown('# Welcome to Clever Cluster 🔍')
st.markdown('In this app, you can use pre-loaded data or your own to explore unsupervised machine learning'
            ' through Principal Components Analysis, K-Means Clustering, and Hierarchical Clustering!')

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
tab1, tab2, tab3, tab4 = st.tabs(['🔢 Data', '📋 Select Features', '🚀 Run Model', '📊 Results'])

#Construct first tab
with tab1:
    #File selection
    file = st.file_uploader('Upload a file of your own!', type = 'csv')
    data_selection = st.selectbox('Or Choose Built-In Data! If you would like to test clusters against true values, these sample sets are highly recommended!', 
                                  ['Wine', 'Penguins'])

    #Intro datasets
    st.markdown('**Wine Dataset**: Contains many numerical features as well as a category column; great for all 3 model types! '
                '"Class" is a great category to test clusters with!')
    st.markdown('**Penguins Dataset**: Contained Island and Species groupings, as well as some features; great for clustering! '
                '"Species" and "Island" are great categories to test clusters with!')

    #Backup data prep
    if data_selection == 'Wine':
        backup_file = 'MLUnsupervisedApp/sample_data/wine.csv'
    else:
        backup_file = 'MLUnsupervisedApp/sample_data/penguins.csv'

    #Reading in data
    if file:
        st.session_state.data = pd.read_csv(file)
    else:
        st.session_state.data = pd.read_csv(backup_file)

    #Display data
    st.markdown('### First, preview the data you have selected!')
    st.dataframe(st.session_state.data, hide_index = True)

    a1, a2 = st.columns([1,1])
    with a1:
        #Convert categorical features to dummy variables
        st.markdown('### Convert Columns')
        st.write('Do you have any columns you would like to convert to dummy variables?' \
                 ' This is often helpful for coding binary variables before analysis!')

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
        st.markdown('### Histogram')
        st.write('Are there any columns you would like to see the distribution of?')
        hist_col = st.selectbox('Column to look at:', st.session_state.data.columns)
        hist = px.histogram(st.session_state.data[hist_col], title = f'Histogram of {hist_col}', color_discrete_sequence = ['skyblue'])
        st.plotly_chart(hist) 

#Construct second tab
with tab2:
    b1, b2 = st.columns([1,1])
    with b1:
        #Select features for the model
        numeric_data = st.session_state.data.select_dtypes(include = ['number'])
        st.markdown('### Select Data Variables and Options')
        feature_cols = st.multiselect('Select at least TWO feature columns', 
                                      numeric_data.columns, 
                                      help = 'Feature columns are the variables that will be used! These MUST be numeric variables.')
        if feature_cols:
            if len(feature_cols) < 2:
                st.warning('Select at least TWO columns')
                st.stop()
            else:
                st.session_state.X = st.session_state.data[feature_cols]
        else:
            st.warning('At least one column must be selected!')
            st.stop()

    with b2:
        st.write('Data Options')
        #Scaling option
        scale = st.checkbox('Scale data', value = True, help = 'Scaling data is important for these models! Only in rare situations should this be unchecked!')

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
    
    #Preview selected data
    st.markdown('### Preview Selected Data')
    st.dataframe(st.session_state.X, hide_index = True)

    #Model selection
    st.markdown('### Choose model to build')
    st.markdown('How to choose the best model:')
    st.markdown('''
                Do you want **dimensionality reduction** or **clustering**? \n\n
                - Dimensionality Reduction - Reducing the number of variables into latent key components \n\n
                - Clustering - Collecting observations in similar groups \n\n
                If you want dimensionality reduction, choose Principal Components Analysis! \n\n
                If you want clustering, choose K-Means or Hierarchical! They can be tested against real data too! \n\n
                ''')
    st.session_state.model = st.selectbox('Select model', 
                                          ['Principal Components Analysis', 'K-Means Clustering', 'Hierarchical Clustering'])

#Construct PCA Model
def pca_model(data):
    #Intro PCA
    st.markdown('### Welcome to Principal Components Analysis!')
    st.markdown('Principal components analysis condenses your feature variables into fewer components that still contain most of the information!'
                ' All you have to do is choose the number of components you would like!')
    st.markdown('For now, select a number that is less than the number of features you have!' 
                ' In the next tab, we will look at better ways to choose the number of components!')
    
    #Choose number of components
    components = st.slider('Choose number of components', min_value = 1, max_value = len(data.columns))

    #Initialize model
    if st.button('Run model!'):
        st.session_state.pca = PCA(n_components = components)
        st.session_state.X_pca = st.session_state.pca.fit_transform(data)
        st.success('Congratulations! You have run the PCA model!')

#Construct PCA visualization
def pca_vis():
    #Display explained variance
    st.markdown('### Explained Variance Ratio')
    st.write("First, let's look at what proportion of the total variance in the features you selected is " \
             "explained by the principal components you have created")
    
    m1, m2, m3 = st.columns([1,1,1])
    with m2:
        st.metric(label = 'Total Explained Variance by Principal Components', 
                  value = round(sum(st.session_state.pca.explained_variance_ratio_), 2), 
                  border = True)

    #Component loadings
    st.markdown('### Feature Component Loadings')
    st.write('This dataframe shows us how much each of the selected factors is weighted into the principal component(s)!')
    loadings_df = pd.DataFrame(st.session_state.pca.components_, 
                               columns = st.session_state.X.columns, 
                               index = [f'PC{i+1}' for i in range(st.session_state.pca.n_components_)])
    
    #Initialize component loading heatmap
    fig = px.imshow(loadings_df, 
                    labels = dict(x = "Variables", y = "Components", color = "Loading"), 
                    color_continuous_scale = 'RdBu')
    
    #Display component loading
    st.dataframe(loadings_df)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    c1, c2 = st.columns([1,1])

    with c1:
        #Display heatmap
        st.markdown('### Component Loading Heatmap')
        st.plotly_chart(fig)

    with c2:
        #Variance explained chart
        st.markdown('### Variance Explained by Components')
        pca_full = PCA(n_components = len(st.session_state.X.columns)).fit(st.session_state.X)

        explained = pca_full.explained_variance_ratio_ * 100
        components = np.arange(1, len(explained) + 1)
        cumulative = np.cumsum(explained)
   
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Bar(x=components, 
                             y=explained, 
                             name="Individual Variance", 
                             marker_color='steelblue', 
                             text=[f"{v:.1f}%" for v in explained],
                             textposition='auto'), 
                             secondary_y=False)
    
        fig.add_trace(go.Scatter(x = components, y = cumulative, name = 'Cumulative Variance', mode = 'lines+markers', 
                             line = dict(color = 'crimson', width = 2), marker = dict(size = 8)),
                             secondary_y = True)
    
        fig.update_layout(hovermode = 'x unified', 
                          legend = dict(orientation = 'v', 
                                        yanchor = 'middle', 
                                        y = 0.3, 
                                        xanchor = 'right', 
                                        x = 0.8, 
                                        bordercolor = 'black', 
                                        borderwidth = 2))
    
        fig.update_yaxes(title_text = 'Individual Variance Explained (%)', secondary_y = False, color = 'steelblue')
        fig.update_yaxes(title_text = 'Cumulative Variance Explained (%)', secondary_y = True, color = 'crimson', range = [0,105])
        fig.update_xaxes(title_text = 'Principal Component', tickmode = 'array', tickvals = components, ticktext = [f"PC{i}" for i in components])

        st.plotly_chart(fig)
    
    #Explain component selection
    st.markdown('### Choosing Number of Components')
    st.write('In general, you should select a number of principal components that explains at least 90% of the variance!' \
             ' The number of components can vary widely depending on the dataset and number of initial features!')

#Construct K-Means model
def k_cluster_model(data):
    #Intro k means
    st.markdown('### Welcome to K-Means Clustering!')
    st.write('K-Means Clustering splits your data into k clusters using iterative grouping and mean calculations, ultimately working to ' \
             'find groups with converging centroids.')
    st.write('All you have to do is select the number of clusters you would like, and K-Means will do the rest!')
    st.write('While this model does NOT use target variables to cluster, if you would like, you can select a target label to use later ' \
             'to see how accurate your clusters are!')

    #Give option for having a target variable to compare with
    label = st.multiselect('Do you have a target variable in mind?', 
                           st.session_state.data.columns, 
                           max_selections = 1)
    
    if label:
        target_col = label[0]
        y_full = st.session_state.data[target_col]
        st.session_state.y = y_full.loc[st.session_state.X.index]
        num_groups = int(st.session_state.y.astype(str).nunique())
        st.info(f'Your target has {num_groups} groups! Maybe try that number of clusters!')
    else: st.session_state.y = None

    #Select number of clusters
    nclusters = st.slider('How many clusters would you liked to make?', 
                          min_value = 2, 
                          max_value = 15)

    #Run model
    if st.button('Run model!'):
        kmeans = KMeans(n_clusters = nclusters)
        kmeans.fit(st.session_state.X)
        st.session_state.kmeans = kmeans
        st.success('Congratulations! You have run the K-Means model!')

#Construct K-Means visualization
def k_cluster_vis():
    #Check if model is fit and target variable is selected
    if 'kmeans' not in st.session_state or st.session_state.kmeans is None:
        st.warning('Model not fit yet!')

    #If a target variable is selected
    elif st.session_state.y is not None:
        #Intro evaluations
        st.markdown('### K-Means Model Evaluation')
        st.markdown('''
                    There are a few ways we can evaluate our K-Means model: \n\n
                    - Silhouette Score: This score indicates how similar an item is to its own cluster compared to other clusters (higher
                    numbers are better) \n\n
                    - Accuracy: This is a measure of the proportion of items that are clustered properly given the user's target variable \n\n
                    - Within-Cluster Sum of Squares: This indicates how similar items in a cluster are to each other
                    ''')
        
        d1, d2, d3 = st.columns([1,1,1])
        
        #Map labels to clusters
        preds = st.session_state.kmeans.predict(st.session_state.X)
        target_raw = st.session_state.y.values.ravel()

        unique_targets = np.unique(target_raw)
        target_mapped = np.array([np.where(unique_targets == t)[0][0] for t in target_raw])

        cm = confusion_matrix(target_mapped, preds)

        row_ind, col_ind = linear_sum_assignment(-cm)

        mapping = {col: row for row, col in zip(row_ind, col_ind)}

        aligned_preds = np.array([mapping.get(p, -1) for p in preds])

        final_labels = [unique_targets[i] if i != -1 else "Unassigned" for i in aligned_preds]

        #Display metrics
        with d1:
            st.metric(label = 'Silhouette Score', 
                      value = round(silhouette_score(st.session_state.X, st.session_state.kmeans.labels_), 2),
                      border = True)
        with d2:
            st.metric(label = 'Accuracy', 
                      value = round(accuracy_score(target_mapped, aligned_preds), 2),
                      border = True)
        with d3:
            st.metric(label = 'Within-Cluster Sum of Squares', 
                      value = round(st.session_state.kmeans.inertia_, 2),
                      border = True)
        
        #PCA intro
        st.markdown('### Using PCA to visualize clusters')
        st.write('Using PCA, we can condense feature variables into two components, which, while not perfect, allows us to visualize' \
                 ' the K-Means clusters, as well as how they compare to actual label values.')

        #Use PCA for visualizing clusters
        pca_2d = PCA(n_components=2)
        pca_data = pca_2d.fit_transform(st.session_state.X)

        plot_df = pd.DataFrame(pca_data, 
                               columns = ['PC1', 'PC2'])
        
        plot_df['Cluster_Aligned'] = final_labels
        plot_df['Cluster_Aligned'] = plot_df['Cluster_Aligned'].astype(str)
        plot_df['Actual'] = st.session_state.y.reset_index(drop=True).astype(str)

        fig = px.scatter(plot_df, 
                         x = 'PC1', 
                         y = 'PC2',
                         color = 'Cluster_Aligned',
                         symbol = 'Actual',
                         title = 'PCA: K-Means Clusters vs Real Labels',
                         hover_data = ['Cluster_Aligned', 'Actual'])

        st.plotly_chart(fig)

    #If no target variable is selected
    elif st.session_state.y is None:
        st.markdown('### K-Means Model Evaluation')
        st.markdown('''
                    There are a couple ways we can evaluate our K-Means model: \n\n
                    - Silhouette Score: This score indicates how similar an item is to its own cluster compared to other clusters (higher
                    numbers are better) \n\n
                    - Within-Cluster Sum of Squares: This indicates how similar items in a cluster are to each other
                    ''')
        
        e1, e2 = st.columns([1,1])
        #Display Metrics
        with e1:
            st.metric(label = 'Silhouette Score', 
                      value = round(silhouette_score(st.session_state.X, st.session_state.kmeans.labels_),2),
                      border = True)
        with e2:
            st.metric(label = 'Within-Cluster Sum of Squares', 
                      value = round(st.session_state.kmeans.inertia_, 2),
                      border = True)
        
        #PCA for visualization
        st.markdown('### Using PCA to visualize clusters')
        st.write('Using PCA, we can condense feature variables into two components, which, while not perfect, allows us to visualize' \
                 ' the K-Means clusters.')
        
        pca_2d = PCA(n_components = 2)
        pca_data = pca_2d.fit_transform(st.session_state.X)

        plot_df = pd.DataFrame(pca_data, columns = ['PC1', 'PC2'])
        plot_df['Cluster'] = st.session_state.kmeans.labels_.astype(str)

        fig = px.scatter(plot_df, 
                         x = 'PC1', 
                         y = 'PC2', 
                         color = 'Cluster', 
                         title = 'PCA: K-Means Clusters', 
                         hover_data = 'Cluster')

        st.plotly_chart(fig)

    #Elbow plot and silhouette plot
    ks = range(2,16)

    wcss = []
    silhouette_scores = []

    for k in ks:
        km = KMeans(n_clusters = k, random_state = 1)
        km.fit(st.session_state.X)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(st.session_state.X, labels))

    st.markdown('### Choosing Optimal Number of Clusters')
    st.markdown('''
                There are two easy visual ways to choose the optimal number of clusters (k)! \n\n
                1. **Elbow Plot** - This plots WCSS vs. number of clusters, and the optimal k is at the 'elbow', 
                or where the rate of decrease sharply changes \n\n
                2. **Silhouette Plot** - This plots silhouette score vs. number of clusters, and the optimal k is the highest point \n\n
                ''')

    f1, f2 = st.columns([1,1])
    with f1:
        fig1 = px.line(x = ks, 
                       y = wcss, 
                       markers = True)
        fig1.update_layout(title = 'Elbow Method for Optimal K',
                           xaxis_title = 'Number of clusters (k)',
                           yaxis_title = 'Within-Cluster Sum of Squares (WCSS)')

        st.plotly_chart(fig1)

    with f2:
        fig2 = px.line(x = ks, 
                       y = silhouette_scores, 
                       markers = True, 
                       color_discrete_sequence = ['red'])
        fig2.update_layout(title = 'Silhouette Score for Optimal K',
                           xaxis_title = 'Number of clusters (k)',
                           yaxis_title = 'Silhouette Score')

        st.plotly_chart(fig2)

#Construct Hierarchical Clustering model
def hierarch_model(data):
    #Intro hierarchical clustering
    st.markdown('### Welcome to Hierarchical Clustering!')
    st.write('Hierarchical Clustering begins by clustering individual observations, selecting clusters that minimize variance and distance,' \
    'and then combines multiple clusters in a new cluster. This process continues, and what we end up with is called a dendrogram, which ' \
    'can be seen below. Using this dendrogram, we can select a number of clusters we would like to make that seems adequate for our data!')

    #Select linkage method
    st.markdown('''
                Now, select a linkage method: \n\n
                - **Ward**: Mimimizes variance within clusters \n\n
                - **Complete**: Calculates maxiumum distance between points in separate clusters \n\n
                - **Average**: Calculates average distance between points in separate clusters \n\n
                - **Single**: Calculates minimum distance between points in separate clusters
                ''')
    
    link_method = st.selectbox('Select linkage method', ['ward', 'complete', 'average', 'single'])

    #Build dendrogram
    Z = linkage(data, method = link_method)

    plt.figure(figsize=(20, 7))

    dendrogram(Z)

    plt.title("Hierarchical Clustering Dendrogram")
    plt.ylabel("Distance")
    plt.xticks([])

    st.pyplot(plt)

    #Give option for validation label
    label = st.multiselect('Do you have a target variable in mind?', 
                           st.session_state.data.columns, 
                           max_selections = 1)
    
    if label:
        target_col = label[0]
        y_full = st.session_state.data[target_col]
        st.session_state.y = y_full.loc[st.session_state.X.index]
        num_groups = int(st.session_state.y.astype(str).nunique())
        st.info(f'Your target has {num_groups} groups! Maybe try that number of clusters!')
    else: st.session_state.y = None

    #Select number of clusters
    nclusters = st.slider('Using the dendrogram, select a number of clusters', min_value = 2, max_value = 15)

    #Initialize model
    if st.button('Run model!'):
        st.session_state.agg = AgglomerativeClustering(n_clusters = nclusters, linkage = link_method)
        st.success('Congratulations! You have run the hierarchical model!')
    else:
        st.warning('You have not run the model!')
    
#Construct hierarchical clustering visualization
def hierarch_vis():
    #Check if model has been initialized
    if 'agg' not in st.session_state or st.session_state.agg is None:
        st.warning('Model not fit yet!')

    #If label selected
    elif st.session_state.y is not None:
        #Intro model evaluation
        st.markdown('### Hierarchical Model Evaluation')
        st.markdown('In order to evaluate our hierarchical model, we can use the **silhouette score**, ' \
                    'which indicates how similar an item is to its own cluster compared to other clusters - higher ' \
                    'numbers are better. We can also use **accuracy**, which measures how closely our predictions fit the real values!')
        
        d1, d2 = st.columns([1,1])
        
        #Map clusters to labels
        preds = st.session_state.agg.fit_predict(st.session_state.X)
        target_raw = st.session_state.y.values.ravel()

        unique_targets = np.unique(target_raw)
        target_mapped = np.array([np.where(unique_targets == t)[0][0] for t in target_raw])

        cm = confusion_matrix(target_mapped, preds)

        row_ind, col_ind = linear_sum_assignment(-cm)

        mapping = {col: row for row, col in zip(row_ind, col_ind)}

        aligned_preds = np.array([mapping.get(p, -1) for p in preds])

        final_labels = [unique_targets[i] if i != -1 else "Unassigned" for i in aligned_preds]

        #Display metrics
        with d1:
            st.metric(label = 'Silhouette Score', 
                      value = round(silhouette_score(st.session_state.X, st.session_state.agg.labels_), 2),
                      border = True)
        with d2:
            st.metric(label = 'Accuracy', 
                      value = round(accuracy_score(target_mapped, aligned_preds), 2),
                      border = True)

        e1, e2 = st.columns([1,1])
        with e1:
            st.markdown('### Using PCA to visualize clusters')
            st.write('Using PCA, we can condense feature variables into two components, which, while not perfect, allows us to visualize' \
                     ' the hierarchical clusters, as well as how they compare to actual label values.')
            
            #Use PCA for visualizing clusters
            pca_2d = PCA(n_components=2)
            pca_data = pca_2d.fit_transform(st.session_state.X)

            plot_df = pd.DataFrame(pca_data, 
                                   columns = ['PC1', 'PC2'])
        
            plot_df['Cluster_Aligned'] = final_labels
            plot_df['Cluster_Aligned'] = plot_df['Cluster_Aligned'].astype(str)
            plot_df['Actual'] = st.session_state.y.reset_index(drop=True).astype(str)

            fig = px.scatter(plot_df, 
                             x = 'PC1', 
                             y = 'PC2',
                             color = 'Cluster_Aligned',
                             symbol = 'Actual',
                             title = 'PCA: Hierarchical Clusters vs Real Labels',
                             hover_data = ['Cluster_Aligned', 'Actual'])

            st.plotly_chart(fig)
        with e2:
            #Silhouette Score n optimization
            st.markdown('### Choosing Optimal Number of Clusters')
            st.markdown('''
                        We can easily find the optimal number of clusters! \n\n
                        **Silhouette Plot** - This plots silhouette score vs. number of clusters, and the optimal n is the highest point \n\n
                        ''')
            ns = range(2,16)
            silhouette_scores = []

            for n in ns:
                labels  = AgglomerativeClustering(n_clusters = n, linkage = 'ward').fit_predict(st.session_state.X)
                score = silhouette_score(st.session_state.X, labels)
                silhouette_scores.append(score)

            fig = px.line(x = ns, 
                          y = silhouette_scores, 
                          markers = True,
                          color_discrete_sequence = ['red'])

            fig.update_layout(title = 'Silhouette Score for Optimal n',
                              xaxis_title = 'Number of clusters (n)',
                              yaxis_title = 'Silhouette Score')

            st.plotly_chart(fig)

    #If no label selected
    elif st.session_state.y is None:
        
        d1, d2 = st.columns([2,1])

        with d1:
            #Intro model evaluation
            st.markdown('### Hierarchical Model Evaluation')
            st.markdown('In order to evaluate our hierarchical model, we will use the **silhouette score**, ' \
                        'which indicates how similar an item is to its own cluster compared to other clusters - higher ' \
                        'numbers are better')
       
        #Create cluster labels
        cluster_labels = st.session_state.agg.fit_predict(st.session_state.X)

        #Display silhouette score
        with d2:
            st.metric(label = 'Silhouette Score', 
                      value = round(silhouette_score(st.session_state.X, cluster_labels),2), 
                      border = True)

        e1, e2 = st.columns([1,2])

        #Create and display dataframe with cluster labels
        results = pd.DataFrame(st.session_state.X.copy())
        results['Cluster'] = cluster_labels

        with e2:
            st.dataframe(results, hide_index = True, height = int(round(35*(results['Cluster'].nunique()+1), 0)))

        #Display cluster value counts
        with e1:
            cluster_counts = results['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            cluster_counts = cluster_counts.sort_values(by = 'Cluster', ascending = True)
            st.dataframe(cluster_counts, hide_index = True)

        f1, f2 = st.columns([1,1])
        with f1:
            st.markdown('### Using PCA to visualize clusters')
            st.write('Using PCA, we can condense feature variables into two components, which, while not perfect, allows us to visualize' \
                     ' the hierarchical clusters.')
            #Create 2D PCA for visualization
            pca_2d = PCA(n_components=2)
            pca_data = pca_2d.fit_transform(st.session_state.X)

            plot_df = pd.DataFrame(pca_data, columns = ['PC1', 'PC2'])
            plot_df['Cluster'] = st.session_state.agg.labels_
            plot_df = plot_df.sort_values(by = 'Cluster', ascending = True)
            plot_df['Cluster'] = plot_df['Cluster'].astype('str')

            fig = px.scatter(plot_df, x = 'PC1', y = 'PC2', color = 'Cluster', title = 'PCA: Hierarchical Clustering', hover_data = 'Cluster')

            st.plotly_chart(fig)

        with f2:
            #Silhouette Score n optimization
            st.markdown('### Choosing Optimal Number of Clusters')
            st.markdown('''
                        We can easily find the optimal number of clusters! \n\n
                        **Silhouette Plot** - This plots silhouette score vs. number of clusters, and the optimal n is the highest point \n\n
                        ''')
            ns = range(2,16)
            silhouette_scores = []

            for n in ns:
                labels  = AgglomerativeClustering(n_clusters = n, linkage = 'ward').fit_predict(st.session_state.X)
                score = silhouette_score(st.session_state.X, labels)
                silhouette_scores.append(score)

            fig = px.line(x = ns, 
                          y = silhouette_scores, 
                          markers = True,
                          color_discrete_sequence = ['red'])

            fig.update_layout(title = 'Silhouette Score for Optimal n',
                              xaxis_title = 'Number of clusters (n)',
                              yaxis_title = 'Silhouette Score')

            st.plotly_chart(fig)

#Construct model tab
with tab3:
    if st.session_state.model == 'Principal Components Analysis': 
        pca_model(st.session_state.X)
    elif st.session_state.model == 'K-Means Clustering': 
        k_cluster_model(st.session_state.X)
    elif st.session_state.model == 'Hierarchical Clustering': 
        hierarch_model(st.session_state.X)
    else:
        st.info('Complete previous steps first!')

#Construct results tab
with tab4:
    if st.session_state.model == 'Principal Components Analysis' and st.session_state.pca is not None:
        pca_vis()
    elif st.session_state.model == 'K-Means Clustering' and st.session_state.kmeans is not None:
        k_cluster_vis()
    elif st.session_state.model == 'Hierarchical Clustering' and st.session_state.agg is not None:
        hierarch_vis()
    else:
        st.info('Complete previous steps first!')