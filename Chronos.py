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
                 evaluation_function=None):
        self.G = G
        self.MU = MU
        self.LAMBDA = LAMBDA
        self.p_m = p_m
        self.r_m = r_m
        self.extra_regressors = None


        self.regression_params_start, self.regression_params_end = 0, 1
        self.year_order = 1
        
        self.param_number = 2 + self.year_order + 1


        self.population_fitnesses = np.empty((MU,))
        self.offspring_fitnesses = np.empty((LAMBDA,))

        self.best_individual = None

        if (evaluation_function is None):
            self.evaluation_function = CHRONOS_RMSE

        self.train_history_ = []
        self.validation_history_ = []

    
    ################################################################################
    def init_population(self):
        
        if (self.extra_regressors is None):
            self.population = np.random.normal(scale=2.0, size=(self.MU, self.param_number))
            self.offspring = np.zeros(shape=(self.LAMBDA, self.param_number))

        else:
            raise Exception('Chronos::init_population. Can\'t deal with additional regressors yes.')

        
    
    ################################################################################
    def evaluate_population(self, 
                            the_population, 
                            population_fitnesses,
                            y_df):
        #predictions = np.full((self.MU, y_df.shape[0]), np.nan)
        #print(predictions.shape)

        for i in range(the_population.shape[0]): 
            predictions = the_population[i][0] + (y_df['ts'] * the_population[i][1])
            predictions += the_population[i][2]* np.sin((y_df['ts']/365.25) + the_population[i][3])
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

        self.df_mean = self.train_df['y'].max() - self.train_df['y'].min()

        
        self.init_population()
        print("population initalized")

        for g in range(1, self.G+1):

            self.r_m = 0.1*abs(math.sin(g))
            
            self.evaluate_population(self.population, 
                                     self.population_fitnesses,
                                     self.train_df)

            print_string = f'g: {g}\t '

            
            train_best_fitness = self.find_best_individual()

            print_string += f'\t train_fitness: {train_best_fitness}'

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
    def predict(y_hat_df):

        #_, predictions = self.best_individual_score(y_hat_df, max_lag)
        pass
