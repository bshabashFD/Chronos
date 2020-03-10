import numpy as np
import pandas as pd
import math

def CHRONOS_RMSE(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred)**2))

def CHRONOS_MSE(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred))

class Chronos():

    ################################################################################
    def __init__(self, 
                 G = 1, 
                 MU=50, 
                 LAMBDA=50, 
                 p_m=0.1, 
                 r_m = 0.01,
                 evaluation_function=None,
                 yearly_seasonality=1,
                 monthly_seasonality=1):
        self.G = G
        self.MU = MU
        self.LAMBDA = LAMBDA
        self.p_m = p_m
        self.r_m = r_m
        self.extra_regressors = None


        self.param_number = 2
        self.regression_const = 0
        self.regression_coef = 1

        if (yearly_seasonality != False):
            self.year_order = yearly_seasonality
            self.year_coef_start = self.param_number

            self.param_number += self.year_order + 1
            self.year_coef_end = self.param_number
        if (monthly_seasonality != False):
            self.month_order = monthly_seasonality
            self.month_coef_start = self.param_number
            self.param_number += self.month_order + 1
            self.month_coef_end = self.param_number
        
         


        self.population_fitnesses = np.empty((MU,))
        self.offspring_fitnesses = np.empty((LAMBDA,))

        self.best_individual = None

        if (evaluation_function is None):
            self.evaluation_function = CHRONOS_RMSE

        self.train_history_ = []
        self.validation_history_ = []

    
    ################################################################################
    def init_population(self, y_mean):
        
        if (self.extra_regressors is None):
            self.population = np.random.normal(scale=2.0, size=(self.MU, self.param_number))
            self.offspring = np.zeros(shape=(self.LAMBDA, self.param_number))

        else:
            raise Exception('Chronos::init_population. Can\'t deal with additional regressors yes.')

        self.population[:, 0] = y_mean

        
    
    ################################################################################
    def make_predictions(self, y_df, the_individual):
        predictions = the_individual[self.regression_const] \
                      + (y_df['ts'] * the_individual[self.regression_coef])

        # Yearly seasonality             
        order = 1
        for i in range(self.year_coef_start, self.year_coef_end):
            predictions += the_individual[i] * np.sin((order*y_df['ts']/365.25) + the_individual[self.year_coef_end-1])
            order += 1

        # Monthly seasonality             
        order = 1
        for i in range(self.month_coef_start, self.month_coef_end):
            predictions += the_individual[i] * np.sin((order * y_df['ts']/31) + the_individual[self.month_coef_end-1])
            order +=1
        

        return predictions
    ################################################################################
    def evaluate_population(self, 
                            the_population, 
                            population_fitnesses,
                            y_df):
        #predictions = np.full((self.MU, y_df.shape[0]), np.nan)
        #print(predictions.shape)

        for i in range(the_population.shape[0]): 
            predictions = self.make_predictions(y_df, the_population[i])
            population_fitnesses[i] = self.evaluation_function(y_df['y'], predictions)
        
        
        

        
    ################################################################################
    def perform_tournament_selection(self, 
                                     the_population, 
                                     population_fitnesses, 
                                     tournament_size=2):

        #tournament_size = the_population.shape[0]
        selected_fitnesses_indexes = np.random.choice(the_population.shape[0], 
                                                      size=tournament_size)
        
        fitnesses = population_fitnesses[selected_fitnesses_indexes]

        #print(fitnesses)
        

        winner = selected_fitnesses_indexes[np.argmin(fitnesses)]
        #print(winner)
        
        #assert(False)
        return winner
    ################################################################################
    def mutate_offspring(self):
        mutations = np.random.uniform(low=-self.r_m * self.df_mean,
                                      high=self.r_m * self.df_mean,
                                      size=self.offspring.shape)

        mutations_mask = np.random.binomial(1,
                                            p=self.p_m,
                                            size=self.offspring.shape)




        mutations_application = mutations * mutations_mask
        #print(mutations_application)
        #assert(False)

        self.offspring += mutations_application
    ################################################################################
    def find_best_individual(self):
        best_train_fitness_position = np.argmin(self.population_fitnesses)

        self.best_individual = self.population[best_train_fitness_position]

        return self.population_fitnesses[best_train_fitness_position]
    ################################################################################
    def best_individual_score(self, y_df, max_lag):
        
        predictions = np.full((y_df.shape[0],), np.nan)
        for i in range(max_lag, y_df.shape[0]):
            basic_prediction = np.dot(self.best_individual, 
                                       y_df['y'].iloc[i-max_lag:i].values)
            predictions[i] = basic_prediction

        return self.evaluation_function(y_df['y'], predictions), predictions
    ################################################################################
    def fit(self, 
            tsdf):

        self.train_df = tsdf.copy()
        self.train_df['ts'] = self.train_df['ds'].astype(np.int64)/(1e9*60*60)
        self.min_ts = self.train_df['ts'].min()
        self.train_df['ts'] = self.train_df['ts'] - self.min_ts

        self.df_mean = self.train_df['y'].std()
        print(self.df_mean)

        
        self.init_population(self.train_df['y'].mean())
        print("population initalized")

        base_rm = self.r_m
        for g in range(1, self.G+1):

            self.r_m = base_rm*abs(math.sin(g/2))
            
            self.evaluate_population(self.population, 
                                     self.population_fitnesses,
                                     self.train_df)

            print_string = f'g: {g}\t '

            
            train_best_fitness = self.find_best_individual()

            print_string += f'train_fitness: {round(train_best_fitness,2)}'
            #print_string += f'\tbi: {self.best_individual}'

            print(print_string, end="\r")


            for offspring_index in range(self.LAMBDA):
                selected_index = self.perform_tournament_selection(self.population, 
                                                                   self.population_fitnesses)
                self.offspring[offspring_index, :] = self.population[selected_index, :]
                #print(self.offspring)
            
            
            self.mutate_offspring()
            self.evaluate_population(self.offspring, 
                                     self.offspring_fitnesses,
                                     self.train_df)

            for population_index in range(self.MU):
                selected_index = self.perform_tournament_selection(self.offspring, 
                                                                   self.offspring_fitnesses)
                self.population[population_index, :] = self.offspring[selected_index, :]


        
        return self
        
        
    

    ################################################################################
    def make_prediction_df(self, period=30):
        return_df = pd.DataFrame({"ds": numpy.zeros((period,)),
                                  "y_hat": numpy.zeros((period,))})

        self.train_df

    ################################################################################
    def predict(self, y_hat_df):

        predict_df = y_hat_df.copy()
        predict_df['ts'] = predict_df['ds'].astype(np.int64)/(1e9*60*60)
        predict_df['ts'] = predict_df['ts'] - self.min_ts

        predictions = self.make_predictions(predict_df, self.best_individual)
        y_hat_df['yhat'] = predictions
        
        return y_hat_df
