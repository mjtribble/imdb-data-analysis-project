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
    #    'user': 'root',
    #    'password': 'Montana12',
    #    'host': 'localhost',
    #    'database': 'mydb',
    #    'raise_on_warnings': True,
    # )

    # cnx = pymysql.connect(**config)

    cursor = config.cursor()

    # # Do the number of parts an actor works on increase, decrease or stay the same with age?
    # query1 = ("SELECT * "
    #           "FROM EMPLOYEE "
    #           "WHERE employee.Super_ssn IS NOT NULL;")
    #
    # # As a movie's budget increases do the sales also continuously increase
    # # query a movie budget and sales.
    query2 = ("SELECT Primary_title, Total_gross , Budget "
              "FROM MOVIE, TITLE "
              "WHERE TM_const=T_const AND NOT Total_gross=0 AND NOT Budget=0 ")
    #
    # # Can we predict a genre based on the actor and director of a film
    # query3 = ("SELECT * "
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

    cursor.execute(query2)

    for (Primary_title, Total_gross, Budget) in cursor:
        print("{}, {}, {}".format(Primary_title, Total_gross, Budget))

    # for response in cursor:
    #     print(response)

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
