import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def CHRONOS_RMSE(y_true, y_pred_matrix):
    '''
    Calculates the root mean squared errors between y_true
    and all predictions in y_pred_matrix.

    Parameters:
    -------------
    y_true - a numpy array or pandas series of shape (n_obs, )
    y_pred_matrix - a 2D numpy array or pandas dataframe of shape (n_individuals, n_obs), where
    n_individuals is the number of evolutionary genotypes being evaluated

    Returns:
    -------------
    rmses - a numpy array of shape (n_individuals), where
    n_individuals is the number of evolutionary genotypes being evaluated
    '''
    return np.sqrt(np.nanmean((y_true - y_pred_matrix)**2, axis=1))

################################################################################

def CHRONOS_MAE(y_true, y_pred_matrix):
    '''
    Calculates the mean absolute errors between y_true
    and all predictions in y_pred_matrix.

    Parameters:
    -------------
    y_true - a numpy array or pandas series of shape (n_obs, )
    y_pred_matrix - a 2D numpy array or pandas dataframe of shape (n_individuals, n_obs), where
    n_individuals is the number of evolutionary genotypes being evaluated

    Returns:
    -------------
    maes - a numpy array of shape (n_individuals), where
    n_individuals is the number of evolutionary genotypes being evaluated
    '''
    return np.nanmean(np.abs(y_true - y_pred_matrix), axis=1)

################################################################################

class Chronos():

    '''
    A time series analysis class powered by evolutionary algorithms. The algorithm
    starts with MU random estimates of all the parameters, and improves those estimates
    at every generation (iteration in many other algorithms).

    The major advantage of this method over Prophet at the moment is its ability
    to accept arbitrary cost functions which need not be differentiable or adhere
    to any statistical properties.

    The run-time complexity is in O(G*(MU + LAMBDA)* O(eval)), where G is the number
    of generations to run for, MU is the number of solutions to consider, LAMBDA is
    the number of offspring the MU solutions produce at each iteraiton, and O(eval)
    is the run-time complexity of the evaluation function.

    Parameters:
    ------------
    G -         int, in the range [1, inf]
                The number of generations the algorithm runs for. At every generation new
                LAMBDA solutions are produced, and some of them are used to replace the 
                original MU solutions
    
    MU -        int, in the range [1, inf]
                The number of solutions to consider at every iteration. This is NOT the 
                number of parameters which is considered, but rather how many configurations
                of those parameters are considered
    
    LAMBDA -    int, in the range [1, inf]
                The number of "offspring" solutions. At every iteration, LAMBDA solutions
                are produced from the original MU solutions through mutation and
                crossover. Then a new set of MU solutions are selected from the
                pool of LAMBDA solutions
    
    p_m -       float, in the range [0.0, 1.0]
                The probability of mutation. Larger probabilities provide faster intial
                improvement, but can slowdown convergence.

    r_m -       float, in the range [0.0, inf]
                Radius of mutation. How much coefficients can change at each mutation.
                This value will depend on the problem, but larger values can again slow
                down convergence.

    evaluation_function - ["MAE", "RMSE"], or callable
                The evaluation function for the data. If "MAE" uses the mean absolute 
                error between predictions and observations. If "RMSE", uses the root
                mean squared error between predictions and observations.
                If callable must confer to the following signature:
                def my_eval_func(y_true, y_pred_matrix) where y_true is of shape
                (n_obs, ) and y_pred_matrix is of shape (n_individuals, n_obs) where
                n_individuals is the number of evolutionary gentypes being evaluated
                and n_observations is the number of observations being evaluated.

    yearly_seasonality - int, in the range [0, inf]
                The number of yearly seasonality components. Higher number of components
                requires longer optimization, but also provides a higher degree of
                granularity for rapidly changing yearly trends

    
    weekly_seasonality - int, in the range [0, inf]
                The number of weekly seasonality components. Higher number of components
                requires longer optimization, but also provides a higher degree of
                granularity for rapidly changing yearly trends



    Examples:
    -----------
    >>> y_data = pd.date_range('2018-01-01', '2019-01-01').astype(np.int64)/(1e9*60*60*24)
    >>> train_df = pd.DataFrame(data={"ds": pd.date_range('2018-01-01', '2019-01-01'),
                                      "y" : np.sin(y_data*2*math.pi/365.25)})

    >>> my_cr = Chronos.Chronos(G=1000, evaluation_function="RMSE")
    >>> my_cr = my_cr.fit(train_df)
    >>> result_df = my_cr.predict(train_df)
    >>> print(result_df.head())

              ds             y      yhat
    0 2018-01-01 -1.175661e-14  0.981400
    1 2018-01-02  1.720158e-02  1.099839
    2 2018-01-03  3.439806e-02  1.248383
    3 2018-01-04  5.158437e-02  1.307322
    4 2018-01-05  6.875541e-02  1.222267

    '''
    ################################################################################
    def test_configurations(self):
        # self.G
        if (not isinstance(self.G, int)):
            raise TypeError("G must be an integer in the range [1,inf]")
        elif (self.G < 1):
            raise ValueError("G must be an integer in the range [1,inf]")

        # self.MU
        if (not isinstance(self.MU, int)):
            raise TypeError("MU must be an integer in the range [1,inf]")
        elif (self.MU < 1):
            raise ValueError("MU must be an integer in the range [1,inf]")

        # self.LAMBDA
        if (not isinstance(self.LAMBDA, int)):
            raise TypeError("LAMBDA must be an integer in the range [1,inf]")
        elif (self.LAMBDA < 1):
            raise ValueError("LAMBDA must be an integer in the range [1,inf]")


        # self.p_m 
        if (not isinstance(self.p_m, float)):
            raise TypeError("p_m must be an float larger than 0.0 and smaller than, or equal to, 1.0")
        elif (self.p_m > 1.0) or (self.p_m <= 0.0):
            raise ValueError("p_m must be an float larger than 0.0 and smaller than, or equal to, 1.0")


        # self.r_m
        if (not isinstance(self.r_m, float)):
            raise TypeError("r_m must be an float larger than 0.0")
        elif (self.p_m <= 0.0):
            raise ValueError("r_m must be an float larger than 0.0")

        # TODO: add extra regressors
        #self.extra_regressors = None

        #
        '''if (yearly_seasonality == False):
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

        if (evaluation_function == "MAE"):
            self.evaluation_function = CHRONOS_MAE
        elif (evaluation_function == "RMSE"):
            self.evaluation_function = CHRONOS_RMSE
        else:
            self.evaluation_function = evaluation_function

        self.train_history_ = []
        self.validation_history_ = []'''
    ################################################################################
    def __init__(self, 
                 G = 1000, 
                 MU=100, 
                 LAMBDA=100, 
                 p_m=0.1, 
                 r_m = 0.1,
                 evaluation_function="MAE",
                 yearly_seasonality=3,
                 weekly_seasonality=3,
                 AR_order=0):

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

        self.AR_order = AR_order
            
         
        self.test_configurations()

        self.population_fitnesses = np.empty((MU,))
        self.offspring_fitnesses = np.empty((LAMBDA,))

        self.best_individual = None

        if (evaluation_function == "MAE"):
            self.evaluation_function = CHRONOS_MAE
        elif (evaluation_function == "RMSE"):
            self.evaluation_function = CHRONOS_RMSE
        else:
            self.evaluation_function = evaluation_function

        self.train_history_ = []
        self.validation_history_ = []

        

    
    ################################################################################
    def _init_population(self):
        

        if (self.extra_regressors is None):
            self.population = np.random.normal(scale=self.train_target.std(), size=(self.MU, self.train_df.shape[1]))
            self.offspring = np.zeros(shape=(self.LAMBDA, self.train_df.shape[1]))

        else:
            raise Exception('Chronos::init_population. Can\'t deal with additional regressors yes.')

        #self.population[:, 0] = y_mean

        
    
    ################################################################################
    def _evaluate_population(self, 
                            the_population, 
                            population_fitnesses,
                            y_df):        

        predictions = self.train_df.values.dot(the_population.T)

        population_fitnesses[:] = self.evaluation_function(y_df.values, predictions.T)



        
    ################################################################################
    def _perform_tournament_selection(self, 
                                     the_population, 
                                     population_fitnesses, 
                                     tournament_size=2):

        selected_fitnesses_indexes = np.random.choice(the_population.shape[0], 
                                                      size=tournament_size)
        
        fitnesses = population_fitnesses[selected_fitnesses_indexes]

        

        winner = selected_fitnesses_indexes[np.argmin(fitnesses)]
        
        return winner
    ################################################################################
    def _mutate_offspring(self):
        mutations = np.random.uniform(low = -self.r_m,
                                      high = self.r_m,
                                      size = self.offspring.shape)

        mutations_mask = np.random.binomial(1,
                                            p=self.p_m,
                                            size=self.offspring.shape)




        mutations_application = mutations * mutations_mask

        self.offspring += mutations_application
    ################################################################################
    def _find_best_individual(self):
        best_train_fitness_position = np.argmin(self.population_fitnesses)

        self.best_individual = self.population[best_train_fitness_position].copy()

        return self.population_fitnesses[best_train_fitness_position]
    ################################################################################

    def _create_internal_df(self, tsdf, learn=True):

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

        for o in range(1, self.AR_order+1):
            internal_df[f'AR_{o}'] = internal_df['y'].shift(o)


        
        # Put this here so that the time series starts from 0, but the 
        # seasonality reflects the true day of the series.
        # If this isn't done, the series can be very high numbers (~4000) and
        # so a slope of 0.1 gives a first value of 400
        internal_df['ts'] = internal_df['ts'] - self.min_ts

        internal_df.drop(['ds'], axis=1, inplace=True)
        if ('y' in internal_df):
            internal_df.drop(['y'], axis=1, inplace=True)


        if (train_target is not None):
            train_target = train_target[self.AR_order:]
        internal_df.dropna(inplace=True)
        return internal_df, train_target
        

    ################################################################################
    def fit(self, 
            tsdf):

        self.train_df, self.train_target = self._create_internal_df(tsdf)


        
        self._init_population()
        print("population initalized")

        base_rm = self.r_m
        for g in range(1, self.G+1):

            #self.r_m = base_rm*abs(math.sin(g/2))
            
            self._evaluate_population(self.population, 
                                      self.population_fitnesses,
                                      self.train_target)

            print_string = f'g: {g}\t '

            
            train_best_fitness = self._find_best_individual()

            print_string += f'train_fitness: {round(train_best_fitness,2)}'
            #print_string += f'\tbi: {self.best_individual}'

            print(print_string, end="\r")


            for offspring_index in range(self.LAMBDA):
                selected_index = self._perform_tournament_selection(self.population, 
                                                                    self.population_fitnesses)
                self.offspring[offspring_index, :] = self.population[selected_index, :]
                #print(self.offspring)
            
            
            self._mutate_offspring()
            self._evaluate_population(self.offspring, 
                                      self.offspring_fitnesses,
                                      self.train_target)

            for population_index in range(self.MU):
                selected_index = self._perform_tournament_selection(self.offspring, 
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


        predict_df, prediction_target = self._create_internal_df(y_hat_df)

        if (self.AR_order > 0):

            prediction_target.iloc[self.AR_order:] = np.nan

            for o in range(1, self.AR_order+1):
                predict_df[f'AR_{o}'] = np.nan

            #print(predict_df)
            

            for i in range(self.AR_order, predict_df.shape[0]):
                #print(f'i={i}-------------------------------')
                #print(prediction_target)
                current_row = predict_df.iloc[i]

                for index, j in enumerate(range(self.AR_order, 0, -1)):
                    current_row.iloc[-j] = prediction_target.iloc[i-1-index]
                #print(current_row)

                y_hat_result = np.sum(current_row * self.best_individual)
                
                prediction_target.iloc[i] = y_hat_result

                
            prediction_target.iloc[:self.AR_order] = np.nan
            
            predictions = prediction_target.values
        else:
        
            predictions = predict_df.values.dot(self.best_individual.reshape(1, -1).T)

        #print(predictions)

        return_df = y_hat_df.copy()
        
        return_df['yhat'] = np.nan

        
        return_df.iloc[self.AR_order:, -1] = predictions
        

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

        for i in range(1, self.AR_order+1):
            parameters[f'AR::order_{i}_coef'] = self.best_individual[self.week_index_end+i-1]

        return parameters

    ################################################################################
    def plot_components(self, the_df):

        predict_df, _ = self._create_internal_df(the_df)

        # Growth
        intercept = self.best_individual[self.const_index]
        slope = self.best_individual[self.slope_index]
        plt.figure(figsize=(15,5))
        plt.plot(the_df['ds'].iloc[self.AR_order:],  intercept + slope * predict_df['ts'])
        plt.show()

        offset = 1

        #Yearly
        if (self.year_order > 0):
            plt.figure(figsize=(15,5))
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
            plt.figure(figsize=(15,5))
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

        # AR component
        if (self.AR_order > 0):
            plt.figure(figsize=(15,5))
            bar_values = []
            bar_height = []
            for i in range(1, self.AR_order+1):
                bar_values.append(f'AR_order_{i}_coef')
                bar_height.append(self.best_individual[self.week_index_end+i-1])
            plt.bar(bar_values, bar_height)
            plt.show()
