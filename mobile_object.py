from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import configparser
from threading import Thread
import threading
from queue import Queue
import numpy as np
import time

import movements_simulator as ms
import movements_predictor_test as mpt
import movements_predictor as mp

N_STEPS = 'n_steps'
MAX_AREA = 'max_area'

class MobileObject():
    def __init__(self):
        self.__n_steps = 0
        self.__max_area = 0
        self.__validate_config()
        self.__coordinate = ms.Coordinate()
        self.__lock = threading.Lock()

    def __validate_config(self):
        self.__n_steps = int(self.__check_config_key(N_STEPS))
        if self.__n_steps < 25:
            self.__n_steps = 25
        self.__max_area = int(self.__check_config_key(MAX_AREA))

    def __check_config_key(self, key, default=None):
        config = configparser.ConfigParser()
        config.read('movements_predictions.ini')

        if key in config['DEFAULT']:
            return config['DEFAULT'][key]
        if default:
            return default
        raise ValueError("{key} should be in the config".format(key=key))
        
    #   Plot on a 3D map the simulation and the sensors
    def plot_movements(self, X, Y, tri_dim=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, self.__max_area)
        ax.set_ylim(0, self.__max_area)

        if tri_dim:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim3d(0, self.__max_area)
            ax.set_ylim3d(0, self.__max_area)
            ax.set_zlim3d(0, self.__max_area)
        
        ax.plot(X[:1], Y[:1], 'o')
        ax.plot(X, Y, '--')
        ax.plot(X[-1:], Y[-1:], '^')
        
        plt.show()

    def plot_prediction(self, X_axis, Y_axis, X_predictions, Y_predictions, tri_dim=False):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, self.__max_area)
        ax.set_ylim(0, self.__max_area)

        ax.plot(X_axis[:1], Y_axis[:1], 'o')
        ax.plot(X_axis, Y_axis)
        #ax.plot(X_predictions[:1], Y_predictions[:1], 'o')
        ax.plot(X_predictions, Y_predictions, '--')
        ax.plot(X_predictions[-1:], Y_predictions[-1:], '^')

        plt.show()

    #   Plot comparisons between predictions and test_sets
    def plot_prediction_comparison(self, predicted_X, predicted_Y):
        #   X-axis graph
        plt.subplot(221)
        plt.plot(np.arange(1, len(predicted_X)+1), predicted_X, '.-')
        plt.title('X prediction accuracy')
        plt.ylabel('X-axis')
        plt.xlabel('step')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
        plt.xticks(np.arange(1, len(predicted_X)+1, 1))

        #   Y-axis graph
        plt.subplot(222)
        plt.plot(np.arange(1, len(predicted_Y)+1), predicted_Y, '.-')
        plt.title('Y prediction accuracy')
        plt.xlabel('step')
        plt.ylabel('Y-axis')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 10))
        plt.xticks(np.arange(1, len(predicted_X)+1, 1))

        #   Table
        plt.subplot(212)
        plt.axis('tight')
        plt.axis('off')
        plt.table(cellLoc='center', cellText=np.column_stack((predicted_X, predicted_Y)), colLabels=['X', 'Y'], loc='center')

        plt.show()

    def __get_predictor(self, test):
        if test:
            return mpt.Predictor()
        
        return mp.Predictor()

    #   Simulate MO's movements based on the degree of randomness and the sensor field
    def generate_simulations(self):
        '''
        deg_rand = degree of MO's movement randomness
        n_simulations = number of movement to simulate

        return: X and Y list of coordinates, missing positions (due to SNs bad deployment)
        '''
    
        return self.__coordinate.simulate(self.__n_steps)

    def __create_thread(self, coordinate, verbosity, name, test):
        predictor = self.__get_predictor(test)
        q = Queue()
        th = Thread(target=self.__start_thread, args=(coordinate, verbosity, name, q, predictor, test))

        th.daemon = True

        return th, q

    def __thread_get_value(self, q, test=False):
        predictions = q.get()

        if not test:
            return predictions

        comparison = q.get()

        return predictions, comparison

    def __start_thread(self, coordinate, verbosity, name, q, predictor, test=False):
        q.put(predictor.find_best_model(coordinate, name, self.__lock, verbosity))
        
        if test:
            q.put(predictor.normalize_percentage(self.__lock, name))

    def predict(self, X_axis, Y_axis, test=False):
        X_th, X_q = self.__create_thread(X_axis, 2, 'X', test)
        Y_th, Y_q = self.__create_thread(Y_axis, 2, 'Y', test)

        X_th.start()
        Y_th.start()

        if test:
            X_predictions, X_comparison = self.__thread_get_value(X_q, test)
            Y_predictions, Y_comparison = self.__thread_get_value(Y_q, test)

            min_len = min(len(X_comparison), len(Y_comparison))

            self.plot_prediction_comparison(X_comparison[:min_len], Y_comparison[:min_len])
            self.plot_movements(X_axis, Y_axis)

        else:
            X_predictions = self.__thread_get_value(X_q, test)
            Y_predictions = self.__thread_get_value(Y_q, test)

            min_len = min(len(X_predictions), len(Y_predictions))

            self.plot_prediction(X_axis, Y_axis, X_predictions[:min_len], Y_predictions[:min_len])
            print("X: ", X_axis[-5:], X_predictions)
            print("Y: ", Y_axis[-5:], Y_predictions)

        return X_predictions[:min_len], Y_predictions[:min_len]

def check(X_axis, Y_axis):
    if len(set(X_axis)) == 1 or len(set(Y_axis)) == 1:
        return False

    return True

def main():
    mo = MobileObject()

    X_axis = mo.generate_simulations()
    Y_axis = mo.generate_simulations()
    
    # Used to check whether during the simulation the MO remained always still at either the coordinates
    while not check(X_axis, Y_axis):
        X_axis = mo.generate_simulations()
        Y_axis = mo.generate_simulations()

    # If 'test' = True, return only the result of the tests (without new predictions)
    # If 'test' = False, predict new locations based on the MO's movements simulation
    X_predictions, Y_predictions = mo.predict(X_axis, Y_axis, test=True)
    
if __name__ == '__main__':
    main()