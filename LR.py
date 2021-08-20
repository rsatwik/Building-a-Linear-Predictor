import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)

try:
    glob_data = pd.read_csv('/kaggle/input/programming-assignment-1/train.csv')
    glob_targetsDf = glob_data.loc[:, glob_data.columns[-1]]
    glob_targets = np.array(glob_targetsDf)
    glob_std = glob_targets.std(axis=0)
    glob_mean = glob_targets.mean(axis=0)
except:
    glob_std = 1353.4364424352016
    glob_mean = 5590.366013621458


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        return
    
    def __call__(self, features, is_train=False):
        try:
            for i in range(features.shape[1]):
                features[:,i] = (features[:,i] - features[:,i].mean(axis=0)) / (features[:,i].std(axis=0)) #x-mu/sigma
            return features
        except IndexError:
            features = (features - features.mean(axis=0)) / (features.std(axis=0))
            return features
        raise NotImplementedError


def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''
    data = pd.read_csv(csv_path)
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # corr = pd.read_csv('/kaggle/input/cs725-autumn-2020-programming-assignment-1/dataset/train.csv').corr()
    # sns.heatmap(corr, 
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values)
    # plt.show()
    data.drop([' LDA_00',' global_sentiment_polarity',' global_rate_negative_words',' avg_negative_polarity',' n_unique_tokens',' n_non_stop_unique_tokens'],axis=1)
    if is_train:
        print('Training data')
        featureMatrixDf = data.loc[:, data.columns[0:data.shape[1]-1]]
    else:
        print('Test Data')
        featureMatrixDf = data.loc[:, data.columns[0:data.shape[1]]]
    featureMatrix = np.array(featureMatrixDf)
    if scaler:
        featureMatrix = scaler(featureMatrix)
    # bias column
    featureMatrix = np.insert(featureMatrix, 0, np.ones((featureMatrix.shape[0],)), axis = 1)
    return featureMatrix

    raise NotImplementedError # check if this has to commented out later... from the video

def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data = pd.read_csv(csv_path)
    
    targetsDf = data.loc[:, data.columns[-1]]
    targets = np.array(targetsDf)
    if scaler:
        targets = scaler(targets)
    return targets

    raise NotImplementedError
     

def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''
    x = feature_matrix
    y = targets
    w = x.T@x 
    wshape,qshape = w.shape
    w = w +C*np.eye(wshape)
    w = np.linalg.inv(w)@x.T
    return w@y

    raise NotImplementedError 

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''
    data = feature_matrix@weights
#     data = data*glob_std + glob_mean
    return data

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    n = feature_matrix.shape[0]
    predictions = get_predictions(feature_matrix, weights)
    loss = predictions - targets
    return np.sum(np.square(loss), axis = 0)/n

    raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    weights_sq = np.square(weights)
    return np.sum(weights_sq, axis=0)

    raise NotImplementedError

def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''
    return mse_loss(feature_matrix, weights, targets) + C*l2_regularizer(weights)

    raise NotImplementedError

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    diff = targets - get_predictions(feature_matrix, weights)
    grad = (-2*(feature_matrix.T@diff)/feature_matrix.shape[0]) + 2*C*weights  
    return grad

    raise NotImplementedError

def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''
    temp = list(range(feature_matrix.shape[0]))
    np.random.shuffle(temp)
    random_sample = temp[:32] 
    sampled_feature_matrix = feature_matrix[random_sample,:]
    sampled_targets = train_targets[random_sample]
    return (sampled_feature_matrix, sampled_targets)
    raise NotImplementedError
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''
    return np.zeros((n,))
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    ''' 
    return weights-(lr*gradients)
    raise NotImplementedError

def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
    if arg_1*arg_2 < 0:
        return True
    else:
        return False
    
#     if abs(abs(arg_1)-abs(arg_2)) < 1e-8:
#         return True
#     else:
#         return False
    # allowed to modify argument list as per your need
    # return True or False
    raise NotImplementedError
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    prev_dev_loss = 0
    prev_train_loss = 0
    prev_temp = 0
    temp = 0
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):

        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)  

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            temp = train_loss - prev_train_loss
            if step > 100:
                if early_stopping(temp, prev_temp):
                    break
            prev_temp = train_loss - prev_train_loss
            prev_train_loss = train_loss
            prev_dev_loss = dev_loss
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
        
        '''
        implement early stopping etc. to improve performance.
        '''

    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    #predictions = (predictions*glob_std) + glob_mean
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    scaler = False
    train_features, train_targets = get_features('/kaggle/input/programming-assignment-1/train.csv',True), get_targets('/kaggle/input/programming-assignment-1/train.csv')
    dev_features, dev_targets = get_features('/kaggle/input/programming-assignment-1/dev.csv',True), get_targets('/kaggle/input/programming-assignment-1/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=0.0024564564564564565)
    
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
#     devlossList=[]
#     Clist = np.linspace(1e-6,2e-6,1000)
#     #     Clist = [10**(-i) for i in range(6)]
#     for i in Clist:
#         a_solution = analytical_solution(train_features, train_targets, C=i)
#         dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
#         train_loss=do_evaluation(train_features, train_targets, a_solution)
#         print(i)
#         print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
#         devlossList.append(dev_loss)
    
    
    
    scaler = Scaler() #use of scaler is optional
    train_features, train_targets = get_features('/kaggle/input/programming-assignment-1/train.csv',True,scaler), get_targets('/kaggle/input/programming-assignment-1/train.csv')
    dev_features, dev_targets = get_features('/kaggle/input/programming-assignment-1/dev.csv',True,scaler), get_targets('/kaggle/input/programming-assignment-1/dev.csv')
    
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=1e-3,
                        C=1e-6,
                        batch_size=32,
                        max_steps=2000000,
                        eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
