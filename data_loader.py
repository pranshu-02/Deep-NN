"""
Data_Loader
To load the MNIST image data.
"""

import _pickle as cPickle
import gzip
import numpy as np

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10))
    e[j] = 1.0
    return e

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    tr_d,_,te_d = cPickle.load(f,encoding='iso-8859-1')
    f.close()    
    training_inputs = np.array(tr_d[0]).T
    test_inputs = np.array(te_d[0]).T
    training_results = [vectorized_result(y) for y in tr_d[1]]
    return (training_inputs,np.array(training_results).T,test_inputs,te_d[1])



