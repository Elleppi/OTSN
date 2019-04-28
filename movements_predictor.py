import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import configparser

N_TEST = 'n_test'
MAX_AREA = 'max_area'
TOT_PREDS = 'tot_predictions'

class Predictor:
    def __init__(self):
        self.__max_accuracy = 0
        self.__best_w_avg = 0
        self.__best_w_ml = 0
        self.__best_poly_deg = 1

        self.__train_set = []
        self.__test_set = []
        self.__dataset = []

        self.__n_test = 0
        self.__tot_predictions = 0
        self.__max_area = 0

        self.__regressor = None

        self.__validate_config()

    def __validate_config(self):
        self.__max_area = int(self.__check_config_key(MAX_AREA))
        self.__tot_predictions = int(self.__check_config_key(TOT_PREDS))

    def __check_config_key(self, key, default=None):
        config = configparser.ConfigParser()
        config.read('movements_predictions.ini')

        if key in config['DEFAULT']:
            return config['DEFAULT'][key]
        if default:
            return default
        raise ValueError("{key} should be in the config".format(key=key))

    #   Apply a sliding window for w_ml
    def __sliding_window(self, w_ml, w_history):
        '''
        w_ml = size of the window
        w_history = coordinate list

        return: Features (X) and Labels (Y)
        '''
        X = []
        Y = []
        
        for i in range(len(w_history)-w_ml):
            X.append(w_history[i:i+w_ml])
            Y.append(w_history[i+w_ml])
        
        return np.array(X), np.array(Y)

    #   Apply a sliding window for w_avg
    def __sliding_window_average(self, w_avg, train_set):
        '''
        w_avg = size of the window
        train_set = dataset to apply the sliding window

        return: dataset filtered
        '''

        if w_avg == 0:
            return train_set

        history_w = []
        
        for i in range(len(train_set)-w_avg):
            w_a = np.average(train_set[i:i+w_avg], weights=np.arange(1, w_avg+1))
            history_w.append(int(np.round(w_a)))
        
        return history_w

    #   Normalize range within the edges' map
    def __normalize_range(self, val):
        '''
        val: value to normalize

        return: value normalized
        '''
        if(val < 0):
            return 0
        elif(val > self.__max_area):
            return self.__max_area
        else:
            return val

    #   Compute the accuracy based on the proposed accuracy method
    def __accuracy(self, predicted):
        '''
        predicted: list of predicted values

        return: accuracy percentage
        '''
        accuracy = []
        difference = 0
        
        for i in range(len(predicted)):
            difference += abs(self.__test_set[i] - predicted[i])
            
            step_accuracy = int(np.round(100 - difference))
            
            if(step_accuracy > 0):
                accuracy.append(step_accuracy)
            else:
                accuracy.append(0)
            
        if(np.count_nonzero(accuracy) > 0):
            return np.array(accuracy).mean()
        else:
            return 0

    #   Compute predictions in a multi_step fashion (based on previous predictions)
    def __multi_step_prediction(self, w_history, w_ml, regressor):
        '''
        w_history: dataset where to take the last knowing positions
        w_ml: size of the window for ML
        regressor: ML model to use for the predictions

        return: accuracy of the predictions
        '''
        predictions = []
        temp_pred = w_history[-w_ml:]

        for i in range(len(self.__test_set)):
            pred =  np.nan_to_num(regressor.predict([temp_pred])).astype(int)
            predictions.append(self.__normalize_range(int(np.round(pred))))
            temp_pred = temp_pred[1:]+list(np.round(pred).astype(int))
        
        return self.__accuracy(predictions)

    #   Start the simulation process with the proposed model
    def __simulation(self, w_ml, w_history, mdl):
        '''
        w_ml: size of the ML window
        w_history: dataset used to apply the ML sliding window
        mdl: ML model used for the predictions
        q: queue used for the thread to put the results

        return: accuracy and predictions list inside the queue
        '''

        X_train, Y_train = self.__sliding_window(w_ml, w_history)
        
        regressor = mdl
        regressor.fit(X_train, Y_train)

        return self.__multi_step_prediction(w_history, w_ml, regressor), regressor

    def __predict(self):        
        predictions = [self.__dataset[-1]]
        temp_pred = self.__dataset[-(self.__best_w_ml):]

        n_predictions = int(round(self.__tot_predictions * self.__max_accuracy / 100))

        for i in range(n_predictions):            
            pred = self.__regressor.predict([temp_pred])            
            predictions.append(self.__normalize_range(int(np.round(pred))))
            temp_pred = temp_pred[1:]+list(np.round(pred).astype(int))

        return predictions

    def find_best_model(self, coordinate_history, name, lock, verbosity=1):
        '''
        coordinate_history: dataset rekated to X-axis
        Y_history: dataset rekated to Y-axis
        verbosity: how many info to print during the task
        plot_prediction: whether to plot the predictions or not

        return: list of predictions related to X and Y coordinates, average of accuracy between X and Y
        '''
        self.__dataset = coordinate_history
        self.__n_test = int(len(coordinate_history)/4)
        self.__train_set = coordinate_history[:-self.__n_test]
        self.__test_set = coordinate_history[-self.__n_test:]
                    
        for w_avg in range(0, 4):
            w_history = self.__sliding_window_average(w_avg, self.__train_set)                    
            
            for w_ml in range(1, len(w_history)-2):
                for pol_deg in range(1, 4):
                    pipeline = Pipeline([("polynomial_features", PolynomialFeatures(degree=pol_deg, include_bias=False)), ("linear_regression", LinearRegression())])
                             
                    improved = False
                        
                    if(self.__max_accuracy < 100):
                        accuracy, regressor = self.__simulation(w_ml, w_history, pipeline)
                    
                    if(accuracy > self.__max_accuracy):
                        self.__max_accuracy = accuracy
                        self.__best_w_ml = w_ml
                        self.__best_w_avg = w_avg
                        self.__best_poly_deg = pol_deg
                        self.__regressor = regressor
                        improved = True                    
                    
                    if(verbosity == 2 and improved == True):
                        lock.acquire()
                        print("COORDINATE %s: best_accuracy=%d%%, best_w_avg=%d, best_w_ml=%d, best_poly_deg=%d" % (name, self.__max_accuracy, self.__best_w_avg, self.__best_w_ml, self.__best_poly_deg))
                        print("--------------------------------")
                        lock.release()
                        
                    if(verbosity == 3):
                        lock.acquire()
                        print("COORDINATE %s:  (best_accuracy=%d%%, best_w_avg=%d, best_w_ml=%d, best_poly_deg=%d) - current_accuracy = %d%%" % 
                              (name, self.__max_accuracy, self.__best_w_avg, self.__best_w_ml, accuracy, self.__best_poly_deg))
                        
                        if(improved == True):
                            print("IMPROVED")
                            print("--------------------------------")
                            lock.release()
                
        lock.acquire()
        print("COORDINATE %s: final_accuracy=%d%%, final_w_avg=%d, final_w_ml=%d, final_poly_deg=%d" % (name, self.__max_accuracy, self.__best_w_avg, self.__best_w_ml, self.__best_poly_deg))
        print("--------------------------------")
        lock.release()

        return self.__predict()
