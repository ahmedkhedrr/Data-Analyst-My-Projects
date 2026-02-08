#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Once you complete this project, remove these **Tip** sections from your report before submission. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset - [IMDb Movie Dataset]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
#  My project is based on IMdb dataset which includes the following columns: { id , imdb_id , popularity , budget , revenue , original_title , cast , homepage , director , tagline , keywords , overview , runtime , genres , production_companies , release_date , vote_count , vote_average , release_date , budget_adj , revenue_adj} which describes a lot of information about all time recorded films on imdb website and it was a little bit challenging to clean this datasets as there was duplicates and null values.
# 
# 
# ### Question(s) for Analysis
# Q1/What is the highest selling movie in 2015? 
# 
# Q2/What are the highest rated films of all time? 
# 
# Q3/Which Genre attracts the people the most? 
# 
# Q4/Which time of the year is most preferable to release a film?
# 
# Q5/Who is the director with the most films?
# 
# Q6/The Variation of Rating for Each Movie Genre
# 

# In[5]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
# While optional for current notebooks, if you are having trouble with visualizations,
#   remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


df=pd.read_csv('tmdb-movies.csv')


df.info()
df.head()


# <a id='wrangling'></a>

# ## Data Wrangling
# 
# > I found out that there are lots of missing data but I chose to drop only the unnecessary columns that had NaN values and made the needed modifications on the data types to fit my analysis and to achieve the aimed results of this analysis.
# Also, There was only one duplicate needed to be droped so this one was easy.
# 
# 
# 

# In[6]:


# Load your data and print out a few lines. What is the size of your dataframe? 
#   Perform operations to inspect data types and look for instances of missing
#   or possibly errant data. There are at least 4 - 6 methods you can call on your
#   dataframe to obtain this information.

#TO SEE IF THERE ANY MISSING VALUES OR DATA TYPES TO BE CHANGED
df.info()


# In[7]:


#TO SEE THE INTIAL SIZE OF THE DATA
df.shape


# In[8]:


#TO SEE IF THERE ANY DUPICATES
df.duplicated().sum()


# In[9]:


#TO SEE IF THERE ANY NULL VALUES
df.isnull().sum()


# 
# ### Data Cleaning
# 

# In[10]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.


# TO REMOVE THE ONE DUPLICATE FOUND IN THE DATASET

# In[11]:


df.drop_duplicates(inplace=True)
df.duplicated().sum()


# I DROPPED THE DUPLICATES AS THEY SKEW FREQUENCY COUNTS AND STATISTICAL SUMMARIES
#  
# DATASET REDUCDED FROM 10,866 to 10,865 ROWS

# 
# 
# TO REMOVE THE NULL VALUES THAT WON'T AFFECT THE DATASET

# In[12]:


#THE NUMBER OF DROPPED ROWS FROM 'director' is 44 ROWS
df.dropna(subset=['director'], inplace=True)
#THE NUMBER OF DROPPED ROWS FROM 'imdp_id' is 10 ROWS
df.dropna(subset=['imdb_id'], inplace=True)
#THE NUMBER OF DROPPED ROWS FROM 'genres' is 23 ROWS
df.dropna(subset=['genres'], inplace=True)
#THE NUMBER OF DROPPED ROWS FROM 'production_companies' is 1030 ROWS
df.dropna(subset=['production_companies'], inplace=True)
#THE NUMBER OF DROPPED ROWS FROM 'cast' is 76 ROWS
df.dropna(subset=['cast'], inplace=True)


#These columns are necessary for my analysis as Q3 is dependent on the genres column , Q5 is dependent on the director column, imdp_id is the identifier which is essential to be nonull value 
# and I cleaned the cast column hoping to make insights like 'who is the most appearing actor in 2010s?' but I couldn't split it as it will be divided into multiple columns that can't be managed.


# I DROPPED THE NULL VALUES IN THE FOLLOWING COLUMNS:(director,imdp_id,genres,production_companies,cast).
# I SELECTED TO ONLY DROP THE NULL VALUES IN THESE COLUMNS AS THIS COLUMNS ARE CRUCIAL AND THE NULL VALUE IN ANY OF THEM CAN AFFECT THE RECORD WHILE ON THE OTHER HAND WITH A RECORD LIKE: homepage, THERE ARE LOTS OF RECORDS MISSING HOMEPAGE BUT IT'S NOT ESSENTIAL TO THE ANALYSIS.

# In[13]:


df.isnull().sum()
#TO CHECK IF THE ROWS ARE ITERATING ACCURATELY
df.iloc[250:260,]


# In[14]:


df.describe()


# CHECK THE DATASET SIZE AFTER CLEANING:

# In[15]:


df.shape


# NOW THE DATA SET CARRIES 9770 CLEANED ROWS RATHER THAN 10866 
# 
# PERCENTAGE RETAINED:90%

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 
# 
# 
# 
# ### Research Question 1 (What is the highest selling movie in 2015?)

# In[16]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.


# SPECIFY THE YEAR YOU ARE TARGETING

# In[17]:


df2015=df[df['release_year']==2015]
df2015.shape
df.describe()
max=df2015['revenue'].max()
df2015[df2015['revenue']== max]


# In[ ]:


From this , We identified the highest selling movie in 2015 which is Star Wars: The Force Awakens w


# ### Research Question 2  (What are the highest rated films of all time?)

# CREATE A FUNCTION FOR LABELING THE PLOT

# In[18]:


def label_plot(title, x_label, y_label):
    

    plot.title(title)          # Set title
    plot.xlabel(x_label)      # Set X label
    plot.ylabel(y_label)      # Set Y label          
    plot.show() 


# In[19]:


# Continue to explore the data to address your additional research
#   questions. Add more headers as needed if you have more questions to
#   investigate.
highest_rated=df.nlargest(20, 'vote_average')
print(f"Mean rating of top 20 films: {highest_rated['vote_average'].mean():.2f}")
print(f"Median rating: {highest_rated['vote_average'].median():.2f}")
print(f"Rating range: {highest_rated['vote_average'].min():.2f} to {highest_rated['vote_average'].max():.2f}")
highest_rated.plot(x='original_title', y='vote_average' , kind='bar')
label_plot('Top Rated Films of All Time','Film Titles','Rating Scale')


# 
# The bar chart above shows the top 20 highest-rated films of all time. Key findings:
# - The highest-rated film is [Pink Floyd: Pulse] with a rating of [8.7]
# - Most top-rated films score between [8.1] and [8.7]
# 
# 
# This suggests that older movies has higher ratings recorded than the new movies(>2001).

# ### Research Question 3  (Which Genre attracts the people the most?)

# SPLIT THE GENRES INTO 5 COLUMNS SO WE CAN HAVE A CLEAR ANALYSIS 

# In[20]:


genres_split = df['genres'].str.split('|', expand=True)


# NAME THE COLUMNS

# In[21]:


genres_split.columns = [f'genre_{i+1}' for i in range(genres_split.shape[1])]

#CONCAT
df1 = pd.concat([df, genres_split], axis=1)

df1.head()

df1['genre_1'].value_counts()
genre_stats = df1.groupby('genre_1').agg({
    'revenue': 'mean',
    'popularity': 'mean',
    'vote_average': 'mean'
})
#PRINT GENRE STATISTICS FOR EACH GENRE
print("Genre Statistics:")
print(genre_stats.sort_values('revenue', ascending=False))


# In[22]:


df1['genre_1'].value_counts()


# In[23]:


#to identify the highest rated drama movie
drama=df1[df1['genre_1']=='Drama']
max_movie_rat=drama[drama['vote_average']==8.4]
max_movie_rat


# COMPARISON BETWEEN THE AMOUNT OF FILMS FOR EACH GENRE THROGH HISTOGRAM

# In[24]:


df1['genre_1'].hist(figsize=(25,8));
label_plot('Most Attracting Genres','Genres','No. of Films')


# The histogram above shows the no. of films in each genre. Key findings:
# - The genre with the largest amount of films is [Drama] with a no. of [2223] films
# - The highest rated Drama movie is [The Shawshank Redemption] released in 1994 & [Guten Tag] released in 2013.
# 

# HIGHEST EARNING GENRES

# In[25]:


genre_stats.plot(y='revenue',kind='bar')
label_plot('Highest Revenue Genres','Genres','Revenue(*1e8)')


# This bar plot shows the average revenue gained by each genre.
# 
# The Adventure genre is th one with the highest revenue average.
# 
# It's average revenue is equal to 1.328017e+08

# HIGHEST RATING GENRES

# In[26]:


genre_stats.plot(y='vote_average',kind='bar')
label_plot('Highest Rating Genres','Genres','Rating Scale')


# This bar plot shows the average rating of each genre.
# 
# The Documentary genre is th one with the highest average rating.
# 
# It's average revenue is equal to 6.944

# ### Research Question 4  (Which time of the year is most preferable to release a film?)

# NOW, WE NEED TO DIVIDE THE release_date INTO THREE COLUMNS:

# In[27]:


release_dates=df1['release_date'].str.split('/',expand=True)

df1 = pd.concat([df1, release_dates], axis=1)


df1.head()
df1.rename(columns={0: 'quarter', 1: 'day', 2: 'year'}, inplace=True)


# DEFINING A METHOD TO DIVIDE THE MONTHS OF THE YEAR ACCROSS THE 4 QUARTERS.

# In[28]:


def dividing_to_quarters(column,quarter,month1,month2,month3):
    df1[column] = np.where(df1[column].isin([month1, month2, month3]), quarter , df1[column])


# DEFINING THE MONTHS OF THE YEAR INTO THE 4 QUARTERS

# In[29]:


df1['quarter']=df1['quarter'].replace({'1':'Q1','2':'Q1','3':'Q1'})

dividing_to_quarters('quarter','Q2','4','5','6')
dividing_to_quarters('quarter','Q3','7','8','9')
dividing_to_quarters('quarter','Q4','10','11','12')

df1.head()
# I decided to divided into quarters to be easier to analyze what time of the year the production companies target to release their movies within
# By that, I concluded that it's better to release on the 3rd quarter where the traffic is the highest.


# In[30]:


#identify the number of movies released in each quarter
df1['quarter'].value_counts()


# In[36]:


quarters_descending=df1.sort_values(by='quarter',ascending=False)
quarters_descending['quarter'].hist(figsize=(10,10));
label_plot('Distribution of Films all over the Year','Year Quarters','Films Released')


# The bar plot above shows the distribution of the films across the 4 quarter of the year. Key findings:
# 
#  The quarter where the largest amount of films is released is the 3rd Quarter.
#  
#  There are 2759 movies released in Q3
#  
# From this Stat, We can indicate that most production companies releases on the 3rd quarter as people are more welcoming to attend movies in the 3rd quarter.
# 

# #### Research Question 5  (Who is the director with the most films?)

# CREATING A NEW DATAFRAME WITH THE TOP MOST PRODUCING DIRECTORS

# In[32]:


#identify the no. of movies produced by each director
df1['director'].value_counts()


# In[39]:


top_directors=df1['director'].value_counts().head(10)
top_directors.plot(kind='bar',figsize=(10,10))
label_plot('Top 10 Producing Director','Directors','No of films produced by each one')


# This bar plot shows the most producing directors of all time.
# 
# The Highest Producing director is Woody Allen.
# 
# He has produced 42 movies of the IMDb list of movies.

# ### Research Question 6  (The Variation of Rating for Each Movie Genre )

# In[63]:


df1.boxplot(column='vote_average', by='genre_1', figsize=(20, 6));
label_plot('The variation of ratings for each genre','Genres','Rating Scale')


# In[38]:


std_genres = df1.groupby('genre_1').agg({
    'vote_average': ['mean', 'std']
})
std_genres


# This boxplot shows the variation of the ratings for each genre.
# 
# The Highest Standard of variation is in the Science Fiction Genre (it means there are lots of different opinions and controversy on this genre)
# 
# He has a standard deviation of 1.034888
# 
# Comes After it TV Movie Genre with 0.950195

# <a id='conclusions'></a>
# ## Conclusions
# 
# This analysis explored several aspects of film performance, popularity, and industry patterns in order to answer five guiding questions. Based on the available dataset, the highest-selling movie in 2015 was identified clearly, showing a strong dominance in box-office revenues compared to its competitors that year. In addition, the exploration of film ratings revealed a list of the highest-rated films of all time, highlighting long-standing classics as well as more recent critically acclaimed titles.
# 
# When examining audience preferences, the dataset showed that certain genres consistently attract more viewers, suggesting clear patterns in popularity. However, while these findings indicate strong correlations between specific genres and higher engagement, they should not be interpreted as causal relationships. Similarly, the investigation into optimal film-release timing showed noticeable seasonal trends, with specific months yielding higher revenues and more frequent releases. These trends may reflect marketing strategies, holiday schedules, or audience availability, although the analysis did not include statistical testing to confirm significance.
# 
# Finally, the analysis identified the director with the highest number of films in the dataset, giving insight into which filmmakers have the most extensive output within the provided sample.
# 
# Despite these insights, the analysis has several limitations. Most importantly, the dataset may not represent the entire film industry globally, meaning that results may vary if more complete data were included. Moreover, because no statistical tests were conducted, the conclusions are descriptive rather than inferential. Additional factors—such as budgets, advertising, cultural trends, or international releases—were not examined and could influence the results. Future research could integrate larger datasets, include statistical modeling, or investigate audience demographics to provide a more comprehensive understanding of film performance.
# 
# The only limitation I met was dealing with the cast column in which i didn't know what to do with this column as it can't be separated into multiple columns and also i had a question regarding it whic
# 
# Overall, the findings offer valuable descriptive insight into movie ratings, sales, genres, release patterns, and director outputs, while also highlighting opportunities for deeper exploration.
# 
# ## Submitting your Project 
# 
# 

# In[ ]:


# Running this cell will execute a bash command to convert this notebook to an .html file
get_ipython().system('python -m nbconvert --to html Investigate_a_Dataset.ipynb')


# In[ ]:




