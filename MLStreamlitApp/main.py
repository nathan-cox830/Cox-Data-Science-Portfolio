#Import necessary tools
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

#Intialize app
st.set_page_config(page_title = 'Machine Learning App', layout = 'wide')

#Welcome text and instructions
st.markdown('# Welcome to the Machine Learning Wizard App! 🧙')
st.markdown('In this app, you can upload **any** clean dataset, **select** a model, and **watch** it analyze your data!')
st.markdown('To use this app, simply find interesting data, pick the machine learning technique you would like to use,' \
'and explore the data as well as the results of the model you chose!')
st.markdown('How to choose the best model:')
st.markdown('1. If your **target** variable is *numeric*, try linear regression')
st.markdown('2. If your **target** variable is *binary*, try logistic regression')
st.markdown('3. Otherwise, try either decision tree or KNN classifiers')
st.markdown('Feel free to play around with options and try things that may not follow these instructions as well to see how they work!')

#File Uploader and Model Chooser
with st.sidebar:
    st.header('Options')
    file = st.file_uploader('Upload File', type = 'csv')
    backup_data = st.selectbox('Or Choose Built-In Data!', ['Diabetes Data', 'Grade Data'])
    model = st.selectbox('Choose Model Type:', ['Linear Regression', 
                                                'Logistic Regression', 
                                                'Decision Tree Classifier', 
                                                'K-Nearest-Neighbors Classifier'])

#Load Data
if backup_data == 'Diabetes Data':
    backup_file = 'sample_data/diabetes.csv'
else:
    backup_file = 'sample_data/grades.csv'

if file is None:
    data = pd.read_csv(backup_file)
else:
    data = pd.read_csv(file)

#Linear Regression Model Rendering
def LinearRegressionModel(data):
    try:
            #Describe Data
            st.sidebar.write('Number of Rows:', data.shape[0])
            st.sidebar.write('Number of Rows:', data.shape[1])

            #Select desired columns
            target_col = st.sidebar.selectbox('Select the target (output) variable:', 
                                              data.columns, 
                                              index = len(data.columns)-1)
            feature_cols = st.sidebar.multiselect('Select the feature (input) variables:', data.columns)

            #Preview data
            a1, a2 = st.columns((.67, .33))
            with a1:
                st.subheader('Preview')
                st.dataframe(data.head(len(data.columns)), hide_index = True)
            with a2:
                st.subheader('Data Types')
                data_types = pd.DataFrame(data.dtypes, columns=["Data Type"]).reset_index()
                data_types.columns = ["Column", "Data Type"]
                st.dataframe(data_types, hide_index = True)

            #Look at distributions of columns
            b1, b2 = st.columns((.67,.33))
            with b1:
                st.subheader('Summary Statistics')
                st.write('How are each of our variables distributed?')
                st.write("Let's look at some statistics and histograms to get more information!")
                st.write('Then, use this information to select your target and feature variables for your model!')
                st.dataframe(data.describe().head())
            with b2:
                st.subheader('Histogram')
                hist_col = st.selectbox('Histogram variable:', data.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data = data, x = hist_col, kde = False, ax = ax)
                ax.set_xlabel('')
                st.pyplot(fig)

            #See missing values and select model-specific options
            with st.sidebar:
                st.subheader('Missing Values')
                missing_values = pd.DataFrame(data.isnull().sum(), columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.dataframe(missing_values, hide_index = True)
                st.subheader('Model Options')
                drop_na = st.checkbox('Drop missing values')
                scale = st.checkbox('Scale numeric data', 
                                    help = 'Normalize numeric feature variables')
            
            #Drop missing values
            if drop_na:
                data = data.dropna()

            #Model Section
            st.subheader('Linear Regression Model')

            #Select chosen columns
            X = data[feature_cols]
            y = data[target_col]
            
            #Create dummy variables if necessary
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

            #Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

            #Scale data if selected
            if scale:
                scaler1 = StandardScaler()
                X_train = pd.DataFrame(scaler1.fit_transform(X_train), columns = X_train.columns)
                scaler2 = StandardScaler()
                X_test = pd.DataFrame(scaler2.fit_transform(X_test), columns = X_test.columns)

            #Initialize model
            model = LinearRegression()

            #Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #Show model coefficients
            if st.checkbox('Show Coefficients:'):
                st.dataframe({'Feature': feature_cols, 'Coeficient': model.coef_, })

            #Calculate model metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            #Display model metrics
            c1, c2, c3 = st.columns(3)
            container1 = c1.container(border = True)
            container1.metric('R²', f'{r2:.2f}')
            container2 = c2.container(border = True)
            container2.metric('MSE:', f"{mse:.2f}")
            container3 = c3.container(border = True)
            container3.metric('RMSE:', f"{rmse:.2f}")

            #Explain model metrics
            container1.write('The percent of the variance in y that is explained by x')
            container2.write('The average of the squared errors between predicted and true values')
            container3.write('The square root of the Mean Squared Error')

            #Plot actual vs. predicted values
            d1, d2 = st.columns(2)
            with d1:
                fig, ax = plt.subplots()
                ax.plot(X_test, y_test, '.', color="green", label="Actual Values", alpha = .25, markersize = 20)
                ax.plot(X_test, y_pred, '.', color="red", label="Predicted Values", alpha = .25, markersize = 20)
                ax.set_title("Predicted vs. Actual Values")
                ax.set_xlabel('Input')
                ax.set_ylabel('Output')
                ax.legend()
                st.pyplot(fig)

            #Plot residual distribution
            with d2:
                fig, ax = plt.subplots()
                residuals = y_test - y_pred
                sns.regplot(x = y_pred, y = residuals)
                ax.set_title("Residual Distribution")
                ax.set_xlabel('Fitted Value')
                ax.set_ylabel('Residual')
                st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))

#Logistic Regression Model Rendering
def LogisticRegressionModel(data):
    try:
            #Describe Data
            st.sidebar.write('Number of Rows:', data.shape[0])
            st.sidebar.write('Number of Rows:', data.shape[1])

            #Select columns
            target_col = st.sidebar.selectbox('Select the target (output) variable:', 
                                              data.columns, 
                                              index = len(data.columns)-1)
            feature_cols = st.sidebar.multiselect('Select the feature (input) variables:', data.columns)

            #Preview Data
            a1, a2 = st.columns((.67, .33))
            with a1:
                st.subheader('Preview')
                st.dataframe(data.head(len(data.columns)), hide_index = True)
            with a2:
                st.subheader('Data Types')
                data_types = pd.DataFrame(data.dtypes, columns=["Data Type"]).reset_index()
                data_types.columns = ["Column", "Data Type"]
                st.dataframe(data_types, hide_index = True)

            #Look at distribution of columns
            b1, b2 = st.columns((.67,.33))
            with b1:
                st.subheader('Summary Statistics')
                st.write('How are each of our variables distributed?')
                st.write("Let's look at some statistics and histograms to get more information!")
                st.write('Then, use this information to select your target and feature variables for your model!')
                st.dataframe(data.describe().head())
            with b2:
                st.subheader('Histogram')
                hist_col = st.selectbox('Histogram variable:', data.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data = data, x = hist_col, kde = False, ax = ax)
                ax.set_xlabel('')
                st.pyplot(fig)

            #See missing values and select model-specific options
            with st.sidebar:
                st.subheader('Missing Values')
                missing_values = pd.DataFrame(data.isnull().sum(), columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.dataframe(missing_values, hide_index = True)
                st.subheader('Model Options')
                drop_na = st.checkbox('Drop missing values')
                scale = st.checkbox('Scale numeric data', 
                                    help = 'Normalize numeric feature variables')
            
            #Drop missing values
            if drop_na:
                data = data.dropna()
            
            #Model selection
            st.subheader('Logistic Regression Model')

            #Select chosen columns
            X = data[feature_cols]
            y = data[target_col]

            #Create dummy variables if necessary
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
            
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            #Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

            #Scale data if selected
            if scale:
                scaler1 = StandardScaler()
                X_train = pd.DataFrame(scaler1.fit_transform(X_train), columns = X_train.columns)
                scaler2 = StandardScaler()
                X_test = pd.DataFrame(scaler2.fit_transform(X_test), columns = X_test.columns)

            #Initialize data
            model = LogisticRegression()

            #Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #Show model coefficients
            if st.checkbox('Show Coefficients:'):
                st.dataframe({'Feature': feature_cols, 'Coeficient': model.coef_[0], })
            
            #Show predicted probabilities
            if st.checkbox('Show Predicted Probabilities:'):
                probabilities = model.predict_proba(X_test)
                st.dataframe(probabilities)
            
            #Display accuracy metrics
            c1, c2, c3, c4 = st.columns((.25, .25, .25, .25))
            with c1:
                container1 = c1.container(border = True)
                accuracy = accuracy_score(y_test, y_pred)
                container1.metric('Accuracy:', f'{accuracy:.2f}')
                container1.write('The proportion of predictions that are correct')

            with c2:
                container2 = c2.container(border = True)
                f1 = f1_score(y_test, y_pred)
                container2.metric('F1-Score:', f'{f1:.2f}')
                container2.write('An overall combination of precision and recall')
            
            with c3:
                container3 = c3.container(border = True)
                precision = precision_score(y_test, y_pred)
                container3.metric('Precision:', f'{precision:.2f}')
                container3.write('Proportion of predicited positives that are true positives')
            
            with c4:
                container4 = c4.container(border = True)
                recall = recall_score(y_test, y_pred)
                container4.metric('Recall:', f'{recall:.2f}')
                container4.write('Proportion of true positives that are predicted positive')

            #Confusion matrix
            d1, d2 = st.columns((.5, .5))
            with d1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            #ROC curve
            with d2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f'LR ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))

#Decision Tree Classidier Rendering
def DecisionTreeModel(data):
    try:
            #Describe data
            st.sidebar.write('Number of Rows:', data.shape[0])
            st.sidebar.write('Number of Rows:', data.shape[1])

            #Select columns
            target_col = st.sidebar.selectbox('Select the target (output) variable:', 
                                              data.columns, 
                                              index = len(data.columns)-1)
            feature_cols = st.sidebar.multiselect('Select the feature (input) variables:', data.columns)

            #Preview data
            a1, a2 = st.columns((.67, .33))
            with a1:
                st.subheader('Preview')
                st.dataframe(data.head(len(data.columns)), hide_index = True)
            with a2:
                st.subheader('Data Types')
                data_types = pd.DataFrame(data.dtypes, columns=["Data Type"]).reset_index()
                data_types.columns = ["Column", "Data Type"]
                st.dataframe(data_types, hide_index = True)

            #Look at distribution of columns
            b1, b2 = st.columns((.67,.33))
            with b1:
                st.subheader('Summary Statistics')
                st.write('How are each of our variables distributed?')
                st.write("Let's look at some statistics and histograms to get more information!")
                st.write('Then, use this information to select your target and feature variables for your model!')
                st.dataframe(data.describe().head())
            with b2:
                st.subheader('Histogram')
                hist_col = st.selectbox('Histogram variable:', data.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data = data, x = hist_col, kde = False, ax = ax)
                ax.set_xlabel('')
                st.pyplot(fig)

            #See missing values and model-specific options
            with st.sidebar:
                st.subheader('Missing Values')
                missing_values = pd.DataFrame(data.isnull().sum(), 
                                              columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.dataframe(missing_values, hide_index = True)
                st.subheader('Model Options')
                drop_na = st.checkbox('Drop missing values')
                scale = st.checkbox('Scale numeric data', 
                                    help = 'Normalize numeric feature variables')
                depth = st.slider('Decision Tree Max Depth:', 
                                  min_value = 1, 
                                  max_value = 25, 
                                  step = 1, 
                                  value = 3, 
                                  help = 'Maximum number of splits within the decision tree')
                leaf = st.slider('Minimum Leaf Samples:', 
                                 min_value = 1, 
                                 max_value = 25, 
                                 step = 1, 
                                 value = 1, 
                                 help = 'Minimum number of observations in final groupings')
                criterion = st.selectbox('Split Quality Metric:', 
                                         ['gini', 'entropy', 'log_loss'])
            
            #Drop missing values
            if drop_na:
                data = data.dropna()
            
            #Chosen model
            st.subheader('Decision Tree Classification Model')

            #Select chosen data
            X = data[feature_cols]
            y = data[target_col]

            #Create dummy variables if necessary
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            #Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

            #Scale data if selected
            if scale:
                scaler1 = StandardScaler()
                X_train = pd.DataFrame(scaler1.fit_transform(X_train), columns = X_train.columns)
                scaler2 = StandardScaler()
                X_test = pd.DataFrame(scaler2.fit_transform(X_test), columns = X_test.columns)

            #Initialize model
            model = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = leaf, criterion = criterion)

            #Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #Display decision tree
            if st.checkbox('Show Decision Tree:'):
                dot_data = tree.export_graphviz(model, feature_names = X_train.columns,
                                class_names = ['Not_Survived', 'Survived'],
                                filled = True)
                st.graphviz_chart(dot_data)

            #Display predicted probabilities
            if st.checkbox('Show Predicted Probabilities:'):
                probabilities = model.predict_proba(X_test)
                st.dataframe(probabilities)
            
            #Display model metrics
            c1, c2, c3, c4 = st.columns((.25, .25, .25, .25))
            with c1:
                container1 = c1.container(border = True)
                accuracy = accuracy_score(y_test, y_pred)
                container1.metric('Accuracy:', f'{accuracy:.2f}')
                container1.write('The proportion of predictions that are correct')


            with c2:
                f1 = f1_score(y_test, y_pred)
                container2 = c2.container(border = True)
                container2.metric('F1-Score:', f'{f1:.2f}')
                container2.write('An overall combination of precision and recall')
            
            with c3:
                container3 = c3.container(border = True)
                precision = precision_score(y_test, y_pred)
                container3.metric('Precision:', f'{precision:.2f}')
                container3.write('Proportion of predicited positives that are true positives')
            
            with c4:
                container4 = c4.container(border = True)
                recall = recall_score(y_test, y_pred)
                container4.metric('Recall:', f'{recall:.2f}')
                container4.write('Proportion of true positives that are predicted positive')

            #Confusion matrix
            d1, d2 = st.columns((.5, .5))
            with d1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            #ROC Curve
            with d2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f'DT ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))

#KNN Classifier Rendering
def KNN_Model(data):
    try:
            #Describe data
            st.sidebar.write('Number of Rows:', data.shape[0])
            st.sidebar.write('Number of Rows:', data.shape[1])

            #Select columns
            target_col = st.sidebar.selectbox('Select the target (output) variable:', 
                                              data.columns, 
                                              index = len(data.columns)-1)
            feature_cols = st.sidebar.multiselect('Select the feature (input) variables:', data.columns)

            #Preview data
            a1, a2 = st.columns((.67, .33))
            with a1:
                st.subheader('Preview')
                st.dataframe(data.head(len(data.columns)), hide_index = True)
            with a2:
                st.subheader('Data Types')
                data_types = pd.DataFrame(data.dtypes, columns=["Data Type"]).reset_index()
                data_types.columns = ["Column", "Data Type"]
                st.dataframe(data_types, hide_index = True)

            #Look at distribution of columns
            b1, b2 = st.columns((.67,.33))
            with b1:
                st.subheader('Summary Statistics')
                st.write('How are each of our variables distributed?')
                st.write("Let's look at some statistics and histograms to get more information!")
                st.write('Then, use this information to select your target and feature variables for your model!')
                st.dataframe(data.describe().head())
            with b2:
                st.subheader('Histogram')
                hist_col = st.selectbox('Histogram variable:', data.columns)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data = data, x = hist_col, kde = False, ax = ax)
                ax.set_xlabel('')
                st.pyplot(fig)

            #See missing values and model-specific options
            with st.sidebar:
                st.subheader('Missing Values')
                missing_values = pd.DataFrame(data.isnull().sum(), columns=["Missing Values"]).reset_index()
                missing_values.columns = ["Column", "Missing Values"]
                st.dataframe(missing_values, hide_index = True)
                st.subheader('Model Options')
                drop_na = st.checkbox('Drop missing values')
                scale = st.checkbox('Scale numeric data', 
                                    help = 'Normalize numeric feature variables')
                neighbors = st.slider('Number of Neighbors:', 
                                      min_value = 1, 
                                      max_value = 25, 
                                      step = 1, 
                                      value = 5, 
                                      help = 'Number of nearby observations to group with')
                weight = st.selectbox('Weighting Method:', 
                                      ['uniform', 'distance'], 
                                      help = 'Method to weight importance of each neighbor')
            
            #Drop missing values
            if drop_na:
                data = data.dropna()
            
            #Selected model
            st.subheader('K Nearest Neighbors Classification Model')

            #Select chosen columns
            X = data[feature_cols]
            y = data[target_col]

            #Create dummy variables if necessary
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            #Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

            #Scale data if selected
            if scale:
                scaler1 = StandardScaler()
                X_train = pd.DataFrame(scaler1.fit_transform(X_train), columns = X_train.columns)
                scaler2 = StandardScaler()
                X_test = pd.DataFrame(scaler2.fit_transform(X_test), columns = X_test.columns)

            #Initialize model
            model = KNeighborsClassifier(n_neighbors = neighbors, weights = weight)

            #Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            #Plot showing accuracy vs. k
            if st.checkbox('Explore Accuracy as K Changes'):
                accuracies = []
                k_values = range(1,25,1)

                for k in k_values:
                    knn_temp = KNeighborsClassifier(n_neighbors = k)
                    knn_temp.fit(X_train, y_train)
                    y_temp_pred = knn_temp.predict(X_test)
                    accuracies.append(accuracy_score(y_test, y_temp_pred))

                fig, ax = plt.subplots()
                ax.plot(k_values, accuracies, marker='o')
                ax.set_title('Accuracy vs. Number of Neighbors (k)')
                ax.set_xlabel('Number of Neighbors (k)')
                ax.set_ylabel('Accuracy')
                ax.set_xticks(k_values)
                st.pyplot(fig)

            #Display predicted probabilities
            if st.checkbox('Show Predicted Probabilities:'):
                probabilities = model.predict_proba(X_test)
                st.dataframe(probabilities)
            
            #Display model metrics
            c1, c2, c3, c4 = st.columns((.25, .25, .25, .25))
            with c1:
                container1 = c1.container(border = True)
                accuracy = accuracy_score(y_test, y_pred)
                container1.metric('Accuracy:', f'{accuracy:.2f}')
                container1.write('The proportion of predictions that are correct')

            with c2:
                f1 = f1_score(y_test, y_pred)
                container2 = c2.container(border = True)
                container2.metric('F1-Score:', f'{f1:.2f}')
                container2.write('An overall combination of precision and recall')
            
            with c3:
                container3 = c3.container(border = True)
                precision = precision_score(y_test, y_pred)
                container3.metric('Precision:', f'{precision:.2f}')
                container3.write('Proportion of predicited positives that are true positives')
            
            with c4:
                container4 = c4.container(border = True)
                recall = recall_score(y_test, y_pred)
                container4.metric('Recall:', f'{recall:.2f}')
                container4.write('Proportion of true positives that are predicted positive')

            #Confusion matrix
            d1, d2 = st.columns((.5, .5))
            with d1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

            #ROC Curve
            with d2:
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                roc_auc = roc_auc_score(y_test, y_probs)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, lw=2, label=f'KNN ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess') # Plotting 50% line
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred: {} : Try another column for continue".format(str(e)))

#Display the correctly rendered model
if model == 'Linear Regression':
    LinearRegressionModel(data)
elif model == 'Logistic Regression':
    LogisticRegressionModel(data)
elif model == 'Decision Tree Classifier':
    DecisionTreeModel(data)
elif model == 'K-Nearest-Neighbors Classifier':
    KNN_Model(data)