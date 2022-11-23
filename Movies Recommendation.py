# Importing the Libraries

import joblib
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector as connector
import os
from sklearn.feature_extraction.text import TfidfVectorizer
 
# Loading the datset

data = pd.read_csv(r"C:\Users\Shashi\Model Building\Unsupervised Learning or Descriptive Model\Recommendation Engine\Entertainment.csv")

# Connecting with mysql

con = connector.connect(host = 'localhost',
                        port = '3306',
                        user = 'root',
                        password = 'Shashi@16',
                        database = 'mysql',
                        auth_plugin = 'mysql-connector-python')

# Performing operation from python

cur = con.cursor()
con.commit()

# Selct * fun gives the all the information
cur.execute('SELECT * FROM enter.entertainment')
df = cur.fetchall()

# Checking the missing values
data.isna().sum()


# Create a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")

# Transform a count matrix to a normalized tf-idf representation
tfidf_matrix = tfidf.fit(data.Titles)


# Save the Pipeline for tfidf matrix

joblib.dump(tfidf_matrix, 'matrix')

os.getcwd()

mat = joblib.load("matrix")

tfidf_matrix = mat.transform(data.Titles) 


tfidf_matrix.shape # (51, 90)

# cosine(x, y)= (x.y⊺) / (||x||.||y||)
# Computing the cosine similarity on Tfidf matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

joblib.dump(cosine_sim_matrix, 'cosine_matrix')


# Create a mapping of anime name to index number
data_index = pd.Series(data.index, index = data.Titles).drop_duplicates()


# Example
data_id = data_index["GoldenEye (1995)"]

data_id



# Custom Function to Find the Top Games to be Recommended

def get_recommendations(Titles, topN):    
     topN = 10
    # Getting the movie index using its title 
     data_id = data_index[2]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
     cosine_scores = list(enumerate(cosine_sim_matrix[data_id]))
    
    # Sorting the cosine_similarity scores based on scores 
     cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar games 
     cosine_scores_N = cosine_scores[0: topN + 1]
    
    # Getting the movie index 
     data_idx  =  [i[0] for i in cosine_scores_N]
     data_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar Games 
     df_similar_show = pd.DataFrame(columns = ["Titles", "Score"])
     df_similar_show['Titles'] = data.loc[data_idx, 'Titles']
     df_similar_show["Score"] = data_scores
     df_similar_show.reset_index(inplace = True)  
     df_similar_show.drop(["index"], axis=1, inplace=True)
     return(df_similar_show.iloc[1:, ])


# Printing the End Result
rec = get_recommendations("Heat (1995)", topN = 9)
print(rec)



