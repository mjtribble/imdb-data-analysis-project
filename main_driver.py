"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import correlation
import classification
import regression
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import pymysql
import regression


def query():
    config = pymysql.connect("localhost", "root", "*light*Bright", "IMBD")

    cursor = config.cursor()

    # EXECUTE AND PRINT QUERY 1
    # Do the number of parts an actor works on increase, decrease or stay the same with age?
    # query1 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")

    # EXECUTE AND PRINT QUERY 2
    # # As a movie's budget increases do the sales also continuously increase
    # # query a movie budget and sales.
    query2 = ("SELECT Total_gross , Budget "
              "FROM MOVIE, TITLE "
              "WHERE TM_const=T_const AND NOT Total_gross=0 AND NOT Budget=0 "
              "ORDER BY Budget "
              )

    cursor.execute(query2)
    raw_data_2 = []
    for response in cursor:
        raw_data_2.append(response)

    df_2 = pd.DataFrame(raw_data_2, columns=("Gross", "Budget"))

    print(df_2)

    regression.LRegression(df_2)

    # EXECUTE AND PRINT QUERY 3
    # Can we predict a genre based on the actor and director of a film
    # query3 = ('SELECT Genre, ACTOR_N_const, DIRECTOR_N_const '
    #           'FROM ACTOR_HAS_ROLE_IN_TITLE AS a, TITLE_GENRE, DIRECTOR_DIRECTS_A_TITLE AS d '
    #           'WHERE a.TITLE_T_const = TG_const and a.TITLE_T_const = d.TITLE_T_const and d.TITLE_T_const= TG_const ')
    #
    # cursor.execute(query3)

    # This dictionary has genre as a key and a list of actors and directors associated with that genre
    # query3_dict = {}
    #
    # for (Genre, ACTOR_N_const, DIRECTOR_N_const) in cursor:
    #     # print("{}, {}, {}".format(Genre, ACTOR_N_const, DIRECTOR_N_const))
    #     if Genre in query3_dict:
    #         query3_dict[Genre].append(ACTOR_N_const)
    #         query3_dict[Genre].append(DIRECTOR_N_const)
    #     else:
    #         query3_dict.update({Genre: [ACTOR_N_const, DIRECTOR_N_const]})
    #
    # print(query3_dict)

    # EXECUTE AND PRINT QUERY 4
    # What is the probability that a particular genre is more popular in the U.S. vs. other countries?
    # query4 = ("SELECT Country, Genre "
    #           "FROM TITLE_GENRE, MOVIE_COUNTRIES "
    #           "WHERE TC_const = TG_const "
    #           "ORDER BY Country, Genre "
    #           )
    #
    # cursor.execute(query4)

    # not sure how we want this data, right now it is creating a list of genres for a particular country.
    # query4_dict = {}

    # for (Country, Genre) in cursor:
    #     # print("{}, {}, {}".format(Genre, ACTOR_N_const, DIRECTOR_N_const))
    #     if Country in query4_dict:
    #         query4_dict[Country].append(Genre)
    #     else:
    #         query4_dict.update({Country: [Genre]})

    # print(query4_dict)

    # for response in cursor:
    #     print(response)


    # EXECUTE AND PRINT QUERY 5
    # # Can we predict a director based on actors, genre, budget, gross, and country of a film?
    # query5 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")

    # for response in cursor:
    #     print(response)
    #
    # config.close()


if __name__ == '__main__':
    print('Running Queries')
    query()

    # print('Importing data')
    # data = pd.read_csv("INSERT FILE PATH HERE")
    #
    # data.head()


def visualize(d, x_vars, y_vars):
    sns.pairplot(d, x_vars, y_vars, size=7, aspect=0.7, kind='reg')
