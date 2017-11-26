"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""

import correlation
import classification
import regression
import pandas as pd
import seaborn as sns
import pymysql


def query():
    config = pymysql.connect("localhost", "root", "*light*Bright", "IMBD")

    cursor = config.cursor()

    # # Do the number of parts an actor works on increase, decrease or stay the same with age?
    # query1 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")
    #

    # # What is the probability that a particular genre is more popular in the U.S. vs. other countries?
    # query4 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")
    #
    # # Can we predict a director based on actors, genre, budget, gross, and country of a film?
    # query5 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")

    # EXECUTE AND PRINT QUERY 2
    # # As a movie's budget increases do the sales also continuously increase
    # # query a movie budget and sales.
    query2 = ("SELECT Primary_title, Total_gross , Budget "
              "FROM MOVIE, TITLE "
              "WHERE TM_const=T_const AND NOT Total_gross=0 AND NOT Budget=0 "
              "ORDER BY Budget ")
    # cursor.execute(query2)
    # query2_dict = {}
    # for (Primary_title, Total_gross, Budget) in cursor:
    #     print("{}, {}, {}".format(Primary_title, Total_gross, Budget))
    #     query2_dict.update({Primary_title: [Total_gross, Budget]})

    # EXECUTE AND PRINT QUERY 3
    # Can we predict a genre based on the actor and director of a film
    query3 = ('SELECT Genre, ACTOR_N_const, DIRECTOR_N_const '
              'FROM ACTOR_HAS_ROLE_IN_TITLE AS a, TITLE_GENRE, DIRECTOR_DIRECTS_A_TITLE AS d '
              'WHERE a.TITLE_T_const = TG_const and a.TITLE_T_const = d.TITLE_T_const and d.TITLE_T_const= TG_const ')

    cursor.execute(query3)
    query3_dict = {}
    # for (Primary_title, ACTOR_N_const, DIRECTOR_N_const, Genre) in cursor:
    #     print("{}, {}, {}".format(Primary_title, Total_gross, Budget))
    #     query3_dict.update({Primary_title: [Total_gross, Budget]})

    for response in cursor:
        print(response)

    config.close()


if __name__ == '__main__':
    print('Running Queries')
    query()

    # print('Importing data')
    # data = pd.read_csv("INSERT FILE PATH HERE")
    #
    # data.head()


def visualize(d, x_vars, y_vars):
    sns.pairplot(d, x_vars, y_vars, size=7, aspect=0.7, kind='reg')
