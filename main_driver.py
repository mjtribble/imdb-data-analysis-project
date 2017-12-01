"""
Created on November 14, 2017
@author: Melody Tribble & Xuying Wang

"""
import pandas as pd
import pymysql
from PyQt5.QtCore import pyqtSlot

import regression
import correlation
import classification
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton

config = pymysql.connect("localhost", "root", "*light*Bright", "IMBD")
cursor = config.cursor()


class Query:
    def __init__(self):
        pass

    # This will execute query 1 and call the Pearson's Correlation function
    # Do the number of parts an actor works on increase, decrease or stay the same with age?
    @pyqtSlot()
    def query_1(self):
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

        # Creates a dictionary where the key a person's age, and the value is count of roles for that age
        for (Actor1_name, Movie_title, Release_date, Birth_year) in cursor:
            # tries to calculate the age based on query results and count the number of roles per age.
            try:
                age = int(Release_date) - int(Birth_year)
                if age in query1_dict:
                    query1_dict[age] += 1
                else:
                    query1_dict.update({age: 1})

                    # This will catch an exception if either year values cannot be converted to an int.
            except ValueError:
                continue

        # print(query1_dict)

        # Analyse data with Pearson's Correlation Coefficient
        correlation.Pearson(query1_dict)

    # This will execute query 2 and call the Linear Regression function
    # As a movie's budget increases do the sales also continuously increase
    @pyqtSlot()
    def query_2(self):
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

    # This will execute query 3 and call the KNN function
    # Can we predict a genre based on the actor and director of a film?
    @pyqtSlot()
    def query_3(self):

        # pulls the genre and actor/director's unique name ids for each title
        query3 = ('SELECT Genre, ACTOR_N_const, DIRECTOR_N_const '
                  'FROM ACTOR_HAS_ROLE_IN_TITLE AS a, TITLE_GENRE, DIRECTOR_DIRECTS_A_TITLE AS d '
                  'WHERE a.TITLE_T_const = TG_const '
                  'AND a.TITLE_T_const = d.TITLE_T_const '
                  'AND d.TITLE_T_const= TG_const '
                  )

        # execute query
        cursor.execute(query3)

        # This dictionary holds each genre as a key
        # with list integers that correspond to an actor or director as the value
        query3_dict = {}

        for (Genre, ACTOR_N_const, DIRECTOR_N_const) in cursor:

            # remove the first 4 characters of the name id so that only numbers remain
            actor_name = ACTOR_N_const[4:]
            director_name = DIRECTOR_N_const[4:]

            # if the genre already exists in the dictionary, add the two id's to the list
            if Genre in query3_dict:
                query3_dict[Genre].append(int(actor_name))
                query3_dict[Genre].append(int(director_name))
            # if the genre doesn't already exist in the dictionary start a new entry
            else:
                query3_dict.update({Genre: [int(actor_name), int(director_name)]})

        # print dictionary
        # for key in query3_dict: print(key, query3_dict[key])

        # sent the data to be processed by the K-NearestNeighbor function
        classification.KNN(query3_dict)

    # This will execute query 4 and call the Spearman Rank function
    # Is there a correlation between a movie's country and budget
    @pyqtSlot()
    def query_4(self):
        query4 = ("SELECT Country, Budget "
                  "FROM MOVIE, MOVIE_COUNTRIES, TITLE "
                  "WHERE TM_const = T_const "
                  "AND TC_const = T_const "
                  "ORDER BY Budget "
                  )

        cursor.execute(query4)

        # not sure how we want this data, right now it is creating a list of genres for a particular country.
        country_l = []
        budget_l = []
        # for response in cursor:
        #     print(response)

        options = {'USA': 0,
                   'UK': 1,
                   'Mexico': 2,
                   'France': 3,
                   'South Korea': 4,
                   'Canada': 5,
                   'Germany': 6,
                   'Australia': 7,
                   'Hong Kong': 8,
                   'China': 9,
                   'Spain': 10,
                   'Japan': 11,
                   'New Zealand': 12
                   }
        options_l = []
        for key in options:
            options_l.append((key, options[key]))

        df = pd.DataFrame(options_l, columns=['Country', 'Code'])
        print(df)
        for (Country, Budget) in cursor:
            if Budget == 0 or Country == 'New Line':
                continue
            country_int = options[Country]
            country_l.append(country_int)
            budget_l.append(Budget)

        data = country_l, budget_l
        correlation.SpearmanRank(data)

    # Can we predict a director based on actors, genre, budget, gross, and country of a film?
    @pyqtSlot()
    def query_5(self):
        query5 = ("SELECT * "
                  "FROM EMPLOYEE "
                  "WHERE employee.Super_ssn IS NOT NULL;")

        for response in cursor:
            print(response)

        config.close()


class App(QWidget):
    q = Query()

    def __init__(self):
        super().__init__()
        self.title = 'Choose a Question'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.button_1 = QPushButton('Question 1', self)
        self.button_2 = QPushButton('Question 2', self)
        self.button_3 = QPushButton('Question 3', self)
        self.button_4 = QPushButton('Question 4', self)
        self.button_5 = QPushButton('Question 5', self)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # button for question 1
        self.button_1.setToolTip('Question 1 button')
        self.button_1.move(0, 0)

        # self.button_1.clicked.connect(q.query_1())

        # button for question
        self.button_2.setToolTip('Question 2 button')
        self.button_2.move(0, 70)
        # self.button_2.clicked.connect(q.query_2())

        # button for question 3
        self.button_3.setToolTip('Question 3 button')
        self.button_3.move(0, 140)
        # self.button_3.clicked.connect(q.query_3())

        # button for question 4
        self.button_4.setToolTip('Question 4 button')
        self.button_4.move(0, 210)
        # self.button_4.clicked.connect(q.query_4())

        # button for question 5
        self.button_5.setToolTip('Question 5 button')
        self.button_5.move(0, 280)
        # self.button_5.clicked.connect(q.query_5())

        self.show()


if __name__ == '__main__':
    print('Starting Application')
    q = Query()
    # q.query_1()
    q.query_4()
    # application = QApplication(sys.argv)
    # ex = App()
    # sys.exit(application.exec_())
