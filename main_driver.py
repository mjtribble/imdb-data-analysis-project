"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""
import pandas as pd
import pymysql
import regression


def query():
    config = pymysql.connect("localhost", "root", "*light*Bright", "IMBD")

    cursor = config.cursor()

    # EXECUTE AND PRINT QUERY 1
    # Do the number of parts an actor works on increase, decrease or stay the same with age?
    query1 = ("SELECT  Name, Primary_title, Release_date, Birth_year "
              "FROM TEMP_DirectorActor, TITLE, PERSON "
              "WHERE Movie_title = Primary_title "
              "AND Actor1_name = Name "
              "AND NOT Actor1_name = '' "
              "AND NOT Movie_title = '' "
              "AND NOT Release_date = '' "
              "AND NOT Release_date = '\\N' "
              "AND NOT Birth_year= '' "
              "AND NOT Birth_year= '\\N' "
              "LIMIT 3000 ")

    # Run the query
    cursor.execute(query1)

    # This will hold the query results
    query1_dict = {}

    # Creates a dictionary where the key is the person's name, and the value is a set of tuples
    # Name: (Age, Title), (Age, Title)...
    for (Actor1_name, Movie_title, Release_date, Birth_year) in cursor:

        # clean the title name
        Movie_title = ''.join(Movie_title.split())

        # try to calculate the age and set values
        try:
            age = int(Release_date) - int(Birth_year)
            if Actor1_name in query1_dict:
                query1_dict[Actor1_name].append(age)
            else:
                query1_dict.update({Actor1_name: [age]})

        # This will catch an exception if either year values cannot be converted to an int.
        except ValueError:
            continue

    # print(query1_dict)
    first_item = next(iter(query1_dict))
    df_1 = pd.DataFrame({"Name": first_item, "Age": query1_dict[first_item]})
    # need to create a data frame with people and age ranges.

    for key in query1_dict:
        df_1a = pd.DataFrame({"Name": key, "Age": query1_dict[key]})
        df_1 = pd.concat([df_1, df_1a])

    print(df_1)


    # ********** EXECUTE AND PRINT QUERY 2************************
    # # As a movie's budget increases do the sales also continuously increase
    # # query a movie budget and sales.
    # query2 = ("SELECT Total_gross , Budget "
    #           "FROM MOVIE, TITLE "
    #           "WHERE TM_const=T_const AND NOT Total_gross=0 AND NOT Budget=0 "
    #           "ORDER BY Budget "
    #           )
    #
    # cursor.execute(query2)
    # raw_data_2 = []
    # for response in cursor:
    #     raw_data_2.append(response)
    #
    # df_2 = pd.DataFrame(raw_data_2, columns=("Gross", "Budget"))
    #
    # print(df_2)
    #
    # regression.LRegression(df_2)

    # ************* EXECUTE AND PRINT QUERY 3 **************************
    # Can we predict a genre based on the actor and director of a film?
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
    #
    # # not sure how we want this data, right now it is creating a list of genres for a particular country.
    # query4_dict = {}
    #
    # for (Country, Genre) in cursor:
    #     # print("{}, {}, {}".format(Genre, ACTOR_N_const, DIRECTOR_N_const))
    #     if Country in query4_dict:
    #         query4_dict[Country].append(Genre)
    #     else:
    #         query4_dict.update({Country: [Genre]})
    #
    # print(query4_dict)
    #
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
