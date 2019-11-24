import numpy as np
import pandas as pd

def CHRONOS_RMSE(y_true, y_pred):
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
            self.population = np.random.normal(size=(self.MU, self.series_lag))
            self.offspring = np.zeros(shape=(self.LAMBDA, self.series_lag))
            return self.series_lag
        else:
            raise Exception('Chronos::init_population. Can\'t deal with additional regressors yes.')

        
    
    ################################################################################
    def evaluate_population(self, 
                            the_population, 
                            population_fitnesses,
                            y_df, max_lag):
        predictions = np.full((self.MU, y_df.shape[0]), np.nan)

        for i in range(max_lag, y_df.shape[0]):
            basic_predictions = np.dot(the_population, 
                                       y_df['y'].iloc[i-max_lag:i].values)
            predictions[:, i] = basic_predictions
        
        
        for i in range(population_fitnesses.shape[0]):
            population_fitnesses[i] = self.evaluation_function(y_df['y'], 
                                                                    predictions[i])
        #print(population_fitnesses)
        #print(np.argmin(population_fitnesses))
        #print(np.mean(population_fitnesses))

        
    ################################################################################
    def perform_tournament_selection(self, 
                                     the_population, 
                                     population_fitnesses, 
                                     tournament_size=2):

        selected_fitnesses = np.random.choice(the_population.shape[0], size=tournament_size)
        
        fitnesses = population_fitnesses[selected_fitnesses]

        winner = selected_fitnesses[np.argmin(fitnesses)]
        
        return winner
    ################################################################################
    def mutate_offspring(self):
        mutations = np.random.uniform(low=-self.r_m,
                                      high=self.r_m,
                                      size=self.offspring.shape)

        mutations_mask = np.random.binomial(1,
                                            p=self.p_m,
                                            size=self.offspring.shape)




        mutations_application = mutations * mutations_mask
        self.offspring += mutations_application
    ################################################################################
    def find_best_individual(self):
        best_train_fitness_position = np.argmin(self.population_fitnesses)

        self.best_individual = self.population[best_train_fitness_position]
    ################################################################################
    def best_individual_score(self, y_df, max_lag):
        
        predictions = np.full((y_df.shape[0],), np.nan)
        for i in range(max_lag, y_df.shape[0]):
            basic_prediction = np.dot(self.best_individual, 
                                       y_df['y'].iloc[i-max_lag:i].values)
            predictions[i] = basic_prediction

        return self.evaluation_function(y_df['y'], predictions)
    ################################################################################
    def fit(self, 
            y_df_train, 
            y_df_validation=None,
            lag=5):
        self.series_lag = lag

        max_lag = self.init_population()

        for g in range(1, self.G+1):
            
            self.evaluate_population(self.population, 
                                     self.population_fitnesses,
                                     y_df_train, 
                                     max_lag)

            print_string = f'g: {g}\t '
            
            self.find_best_individual()


            train_best_fitness = self.best_individual_score(y_df_train, max_lag)
            self.train_history_.append(train_best_fitness)
            print_string += f'\t train_fitness: {train_best_fitness}'
            if (y_df_validation is not None):
                validation_best_fitness = self.best_individual_score(y_df_validation, max_lag)
                self.validation_history_.append(validation_best_fitness)
                print_string += f'\t val_fitness: {validation_best_fitness}'

            print(print_string, end="\r")


            for offspring_index in range(self.LAMBDA):
                selected_index = self.perform_tournament_selection(self.population, 
                                                                   self.population_fitnesses,
                                                                   2)
                self.offspring[offspring_index, :] = self.population[selected_index, :]
                #print(self.offspring)
            
            
            self.mutate_offspring()
            self.evaluate_population(self.offspring, 
                                     self.offspring_fitnesses,
                                     y_df_train, 
                                     max_lag)

            for population_index in range(self.MU):
                selected_index = self.perform_tournament_selection(self.offspring, 
                                                                   self.offspring_fitnesses,
                                                                   2)
                self.population[population_index, :] = self.offspring[selected_index, :]


        
        return self.train_history_, self.validation_history_
        
        
    

    ################################################################################
    def predict(y_hat_df):
        pass