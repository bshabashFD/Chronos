import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def CHRONOS_RMSE(y_true, y_pred):
    return np.sqrt(np.nanmean((y_true - y_pred)**2, axis=1))

def CHRONOS_MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred), axis=1)

class Chronos():

    ################################################################################
    def __init__(self, 
                 G = 1000, 
                 MU=100, 
                 LAMBDA=100, 
                 p_m=0.1, 
                 r_m = 0.1,
                 evaluation_function=None,
                 yearly_seasonality=3,
                 weekly_seasonality=3):
        self.G = G
        self.MU = MU
        self.LAMBDA = LAMBDA
        self.p_m = p_m
        self.r_m = r_m
        self.extra_regressors = None


        self.param_number = 2
        self.regression_const = 0
        self.regression_coef = 1

        if (yearly_seasonality == False):
            yearly_seasonality = 0
        if (weekly_seasonality == False):
            weekly_seasonality = 0

        if (yearly_seasonality != 0):
            self.year_order = yearly_seasonality
            
        if (weekly_seasonality != 0):
            self.week_order = weekly_seasonality
            
         


        self.population_fitnesses = np.empty((MU,))
        self.offspring_fitnesses = np.empty((LAMBDA,))

        self.best_individual = None

        if (evaluation_function is None):
            self.evaluation_function = CHRONOS_MAE
            #self.evaluation_function = CHRONOS_RMSE

        self.train_history_ = []
        self.validation_history_ = []

    
    ################################################################################
    def init_population(self):
        

        if (self.extra_regressors is None):
            self.population = np.random.normal(scale=self.train_target.std(), size=(self.MU, self.train_df.shape[1]))
            self.offspring = np.zeros(shape=(self.LAMBDA, self.train_df.shape[1]))

        else:
            raise Exception('Chronos::init_population. Can\'t deal with additional regressors yes.')

        #self.population[:, 0] = y_mean

        
    
    ################################################################################
    def evaluate_population(self, 
                            the_population, 
                            population_fitnesses,
                            y_df):        

        predictions = self.train_df.values.dot(the_population.T)

        
        population_fitnesses[:] = self.evaluation_function(y_df.values, predictions.T)



        
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
        mutations = np.random.uniform(low=-self.r_m,
                                      high=self.r_m,
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

        self.best_individual = self.population[best_train_fitness_position].copy()

        return self.population_fitnesses[best_train_fitness_position]
    ################################################################################

    def create_internal_df(self, tsdf, learn=True):

        internal_df = tsdf.copy()
        internal_df['const'] = 1.0
        internal_df['ts'] = internal_df['ds'].astype(np.int64)/(1e9*60*60*24)

        if (learn == True):
            self.min_ts = internal_df['ts'].min()
            train_target = internal_df['y'].copy()
        else:
            train_target = None
        

        
        self.const_index = 0
        self.slope_index = 1
        self.year_index_start = 2
        self.year_index_end = self.year_index_start 
        

        year_series = internal_df['ds'].dt.dayofyear
        for o in range(1, self.year_order+1):
            internal_df[f'y_s_{o}'] = np.sin(o * 2 * math.pi * year_series/365.25)
            internal_df[f'y_c_{o}'] = np.cos(o * 2 * math.pi * year_series/365.25)
            self.year_index_end += 2

        self.week_index_start = self.year_index_end
        self.week_index_end = self.week_index_start

        week_series = internal_df['ds'].dt.dayofweek
        for o in range(1, self.week_order+1):
            internal_df[f'w_s_{o}'] = np.sin(o * 2 * math.pi * week_series/7)
            internal_df[f'w_c_{o}'] = np.cos(o * 2 * math.pi * week_series/7)
            self.week_index_end += 2

        # Put this here so that the time series starts from 0, but the 
        # seasonality reflects the true day of the series.
        # If this isn't done, the series can be very high numbers (~4000) and
        # so a slope of 0.1 gives a first value of 400
        internal_df['ts'] = internal_df['ts'] - self.min_ts

        internal_df.drop(['ds'], axis=1, inplace=True)
        if ('y' in internal_df):
            internal_df.drop(['y'], axis=1, inplace=True)

        #print(internal_df)
        #assert(False)

        return internal_df, train_target
        

    ################################################################################
    def fit(self, 
            tsdf):

        self.train_df, self.train_target = self.create_internal_df(tsdf)


        
        self.init_population()
        print("population initalized")

        base_rm = self.r_m
        for g in range(1, self.G+1):

            #self.r_m = base_rm*abs(math.sin(g/2))
            
            self.evaluate_population(self.population, 
                                     self.population_fitnesses,
                                     self.train_target)

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
                                     self.train_target)

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


        predict_df, _ = self.create_internal_df(y_hat_df)
        
        predictions = predict_df.values.dot(self.best_individual.reshape(1, -1).T)

        return_df = y_hat_df.copy()
        return_df['yhat'] = predictions
        
        return return_df

    ################################################################################
    def get_params(self):
        parameters = {}
        parameters['growth::const'] = self.best_individual[0]
        parameters['growth::coef'] = self.best_individual[1]

        

        o = 1
        for i in range(self.year_index_start, self.year_index_end, 2):
            parameters[f'yearly::order_{o}_coef'] = (self.best_individual[i], self.best_individual[i+1])
            o += 1
        
        o = 1
        for i in range(self.week_index_start, self.week_index_end, 2):
            parameters[f'weekly::order_{o}_coef'] = (self.best_individual[i], self.best_individual[i+1])
            o += 1

        return parameters

    ################################################################################
    def plot_components(self, the_df):

        predict_df, _ = self.create_internal_df(the_df)

        # Growth
        intercept = self.best_individual[self.const_index]
        slope = self.best_individual[self.slope_index]
        plt.figure()
        plt.plot(the_df['ds'],  intercept + slope * predict_df['ts'])
        plt.show()

        offset = 1

        #Yearly
        if (self.year_order > 0):
            plt.figure()
            X = np.array(range(365), dtype=np.float64)
            Y = X * 0
            o = 1
            for i in range(self.year_index_start, self.year_index_end, 2):
                Y += self.best_individual[i] * np.sin(o * 2 * math.pi * X/365.25)
                Y += self.best_individual[i+1] * np.cos(o * 2 * math.pi * X/365.25)
                o += 1

            plt.plot(X, Y)
            plt.show()

            offset += self.year_order

        #Yearly
        if (self.week_order > 0):
            plt.figure()
            X = np.array(range(7), dtype=np.float64)
            Y = X * 0
            o = 1
            for i in range(self.week_index_start, self.week_index_end, 2):
                Y += self.best_individual[i] * np.sin(o * 2 * math.pi * X/7)
                Y += self.best_individual[i+1] * np.cos(o * 2 * math.pi * X/7)
                o += 1

            plt.plot(X, Y)
            plt.show()

            offset += self.week_order
