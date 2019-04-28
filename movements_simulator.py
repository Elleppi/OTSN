import numpy as np
from random import randint
import configparser

DEG_RAND = 'deg_rand'
MAX_SPEED = 'max_speed'
MAX_AREA = 'max_area'

#   Features of the Mobile Object on the coordinate
class Coordinate:
    def __init__(self):
        self.__pos = 0 # position (0 to max_area)
        self.__act = 1 # action (1=decrease, 2=stay, 3=increase)
        self.__speed = 1 # speed (1 up to max_speed)

        self.__deg_rand = 1
        self.__max_area = 0
        self.__max_speed = 0

        self.__validate_config()

    def __validate_config(self):
        self.__max_area = int(self.__check_config_key(MAX_AREA))
        self.__deg_rand = float(self.__check_config_key(DEG_RAND))
        self.__max_speed = int(self.__check_config_key(MAX_SPEED))

    def __check_config_key(self, key, default=None):
        config = configparser.ConfigParser()
        config.read('movements_predictions.ini')

        if key in config['DEFAULT']:
            return config['DEFAULT'][key]
        if default:
            return default
        raise ValueError("{key} should be in the config".format(key=key))

    #   Evaluate the next position in order to not exceed the edges of the area
    def __evaluate_next_pos(self):
        if(self.__act == 1): #decrease position without exceed the edge of the area
            if(self.__pos - self.__speed >= 0):
                self.__pos = self.__pos - self.__speed
            else:
                self.__pos = abs(self.__pos - self.__speed)
                self.__act = 3
                
        if(self.__act == 3): #increase position without exceed the edge of the area
            if(self.__pos + self.__speed <= self.__max_area):
                self.__pos = self.__pos + self.__speed
            else:
                self.__pos = self.__max_area - ((self.__pos + self.__speed) % self.__max_area)
                self.__act = 1


    #   Adjust the new position (i.e. pick a random integer) according to the degree of randomness
    def __deg_rand_influence(self, max_speed, old_value):
        '''
        max: upper bound random integer
        deg_rand: degree of randomness (0 <= deg_rand <= 1)
        old_value: previous value (may be chosen according to the deg_rand)

        return: a random integer based on the degree of randomness
        '''

        new_value = randint(1, max_speed)

        return np.random.choice([new_value, old_value], p=[self.__deg_rand, 1-self.__deg_rand])


    #   Compute the next position of the coordinate based on the previous one and the degree of randomness
    def __next_cell(self):
        '''
        return the updated positioning details
        '''

        self.__act = self.__deg_rand_influence(3, self.__act) # 1=decrease, 2=stay, 3=increase
        
        if(self.__act != 2):
            self.__speed = self.__deg_rand_influence(self.__max_speed, self.__speed)
            self.__evaluate_next_pos()

        return self.__pos

    def simulate(self, n_simulations):
        self.__pos = randint(0, self.__max_area) # position (0 to max_area)
        self.__act = randint(1, 4) # action (1=decrease, 2=stay, 3=increase)
        self.__speed = randint(1, self.__max_speed) # speed (1 up to max_speed)
        
        l = []

        l.append(self.__pos)

        for i in range (1, n_simulations):
            l.append(self.__next_cell())

        return l