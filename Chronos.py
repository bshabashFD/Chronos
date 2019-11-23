import numpy as np
import pandas as pd

class Chronos():

    ################################################################################
    def __init__(self, 
                 G = 100, 
                 MU=50, 
                 LAMBDA=50, 
                 p_m=0.1, 
                 evaluation_function=None):
        self.G = 100
        self.MU = 50
        self.LAMBDA = 50
        self.p_m=0.1
        self.extra_regressors = None

        self.population_fitnesses = np.empty((MU,))
        self.offspring_fitnesses = np.empty((LAMBDA,))


        def RMSE(y_true, y_pred):
            return np.nanmean(np.abs(y_true - y_pred))

        if (evaluation_function is None):
            self.evaluation_function = RMSE

    
    ################################################################################
    def init_population(self):
        
        if (self.extra_regressors is None):
            self.population = np.random.normal(size=(self.MU, self.series_lag))
            return self.series_lag
        else:
            raise Exception('Chronos::fit. Can\'t deal with additional regressors yes.')

        
    ################################################################################
    
    ################################################################################
    def fit(self, y_df, lag=5):
        self.series_lag = lag

        max_lag = self.init_population()



        predictions = np.full((self.MU, y_df.shape[0]), np.nan)
        for i in range(max_lag, y_df.shape[0]):
            predictions[:, i] = np.dot(self.population, 
                                       y_df['y'].iloc[i-max_lag:i].values)
        
        for i in range(self.MU):
            self.population_fitnesses[i] = self.evaluation_function(y_df['y'], 
                                                                    predictions[i])
        print(self.population_fitnesses)
        
        print(np.argmin(self.population_fitnesses))
    

    ################################################################################
    def predict(y_hat_df):
        pass