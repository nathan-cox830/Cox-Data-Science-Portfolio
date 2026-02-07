#load necessary libraries

import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import numpy as np

#set up initial app

st.set_page_config(page_title='Exploring Music Popularity by Genre and Feature',  layout='wide')

#load and rename data

data = pd.read_csv("data/spotify_dataset.csv")

features = ['popularity', 'loudness', 'acousticness', 'valence', 'tempo']

feature_defs = {'loudness':'The overall loudness of a track in decibels (dB)', 
                'acousticness':'A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic', 
                'valence':'A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track',
                'tempo':'The overall estimated tempo of a track in beats per minute (BPM)'}

# create sidebar with selections

with st.sidebar:
    st.header('Options')
    genre = st.selectbox('Choose Genre', 
                     data['track_genre'].unique(), 
                     help = 'Filter report to only show one genre')
    feature = st.selectbox('Choose feature', ['loudness', 'acousticness', 'valence', 'tempo'])

    if st.button('Define Feature'):
        st.write(feature_defs[feature])

#create title and description

a1, a2 = st.columns((.07,1))

a1.image('images/spotify.png', width = 120)
a2.title('Exploring Music Popularity by Genre and Feature')
st.write('Each year, oven 700 million people use Spotify to listen to music. This exploratory data analysis looks a sample of songs from Spotify, and allows you to explore how song popularity changes with different genres and features. Use the panel on the left to explore and learn more!')

#create filtered data

filtered_data = data[data['track_genre'] == genre]
songs = len(data)
avg_pop = filtered_data['popularity'].mean()
genres = data['track_genre'].nunique()

#build initial insights boxes

m1, m2, m3, m4, m5 = st.columns((1,1,1.1,1,1))

m2.metric(label = 'Total Number of Songs (All Genres)', 
          value = songs, 
          border = True)

m3.metric(label = 'Average Popularity for Selected Genre', 
          value = avg_pop, 
          delta = str(round(avg_pop - data['popularity'].mean(),2)) + ' Compared to overall average', 
          border = True)

m4.metric(label = 'Total Number of Genres', 
          value = genres, 
          border = True)

#create histogram and dataframe of filtered data

st.write('Learn more about the popularity of your selected genre, and find some popular songs!')
g1, g2 = st.columns((1,1))
hist1 = px.histogram(filtered_data['popularity'], 
                     nbins = 10, 
                     range_x = [0,100], 
                     labels = {'value':'Popularity'})
hist1.update_layout(showlegend = False, yaxis_title = 'Number of Songs')
hist1.update_traces(marker_color = '#1DB954')

g1.subheader('Popularity Distribution of Selected Genre')
g1.plotly_chart(hist1)
g2.subheader('Songs of Selected Genre by Popularity')
g2.dataframe(filtered_data[['track_name', 'artists', 'popularity']].sort_values(by = 'popularity', ascending = False), 
             hide_index = True, 
             column_config = {'track_name':'Track Name', 'artists':'Artists', 'popularity':'Popularity'})

#build correlation matrix chart

st.write('The matrix below shows how the popularity of your genre correlates with different factors. Use it to help choose a factor to explore!')

corr = filtered_data[features].corr()
corr_chart = px.imshow(corr, 
                       text_auto = '.2f', 
                       color_continuous_scale = 'greens', 
                       aspect = 'auto')

st.subheader('Correlation Matrix Between Popularity and Features for Selected Genre')
st.plotly_chart(corr_chart)

#create visuals for selected feature and popularity
st.write("Now that you've selected a feature, let's see how it is related to the popularity of songs in your genre!")
h1, h2 = st.columns((1,1))

        #create bins of popularity
filtered_data['pop_group'] = np.where(filtered_data['popularity'] > 70, 'High', 
                                      np.where(filtered_data['popularity'] > 30, 'Medium', 'Low'))

scatter1 = px.scatter(filtered_data, x = feature, 
                y = 'popularity',
                trendline = 'ols',
                labels = {feature:feature.capitalize(), 'popularity':'Popularity'})

scatter1.update_traces(marker_color = '#1DB954')

h1.subheader('Scatterplot Between Popularity and Selected Feature')
h1.plotly_chart(scatter1)

violin1 = px.violin(filtered_data, 
                    y = feature, 
                    color = 'pop_group',
                    labels = {feature:feature.capitalize(), 'pop_group':''},
                    color_discrete_sequence = ['forestgreen','mediumseagreen','springgreen'],
                    category_orders = {'pop_group':['Low', 'Medium', 'High']})

violin1.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.15,
    xanchor="center",
    x=0.5))

h2.subheader('Violin Plot of Selected Feature by Popularity Group')

h2.plotly_chart(violin1)