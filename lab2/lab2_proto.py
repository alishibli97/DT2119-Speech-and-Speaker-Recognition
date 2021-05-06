import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    # concatHMM = {}
    # start_prob = np.zeros(hmm1['startprob'].shape[0] + hmm2['startprob'].shape[0] - 1)
    # for i in range(start_prob.shape[0]):
    #     if i < hmm1['startprob'].shape[0] - 1:
    #         start_prob[i] = hmm1['startprob'][i]
    #     else:
    #         start_prob[i] = hmm1['startprob'][-1] * hmm2['startprob'][i - (hmm1['startprob'].shape[0] - 1)]
    
    # shape1 = hmm1['transmat'].shape
    # shape2 = hmm2['transmat'].shape
    # transition_matrix = np.zeros((shape1[0] + shape2[0] - 1, shape1[1] + shape2[1] - 1))
    # for i in range(transition_matrix.shape[0] - 1):
    #     for j in range(transition_matrix.shape[1]):
    #         if i < shape1[0] and j < shape1[1]:
    #             transition_matrix[i,j] = hmm1['transmat'][i,j]
    #         elif i < shape1[0] and j >= shape1[1]:
    #             transition_matrix[i,j] =  hmm1['transmat'][i, -1] * hmm2['startprob'][j - (hmm1['transmat'].shape[1] - 1)]
    #         elif i>=shape1[0] and j >= shape1[1]:
    #             transition_matrix[i, j] = hmm2['transmat'][i - (hmm1['transmat'].shape[0] - 1), j - (hmm1['transmat'].shape[1] - 1)]
   
    # transition_matrix[-1, -1] = 1
    # means = np.vstack((hmm1['means'], hmm2['means']))
    # covars = np.vstack((hmm1['covars'], hmm2['covars']))

    # concatHMM['startprob'] = start_prob
    # concatHMM['transmat'] = transition_matrix
    # concatHMM['means'] = means
    # concatHMM['covars'] = covars

    # return concatHMM
    
    concatedHMM = {}
    #M is the number of emitting states in each HMM model (could be different for each)
    #K is the sum of the number of emitting states from the input models
    
    M1 = hmm1['means'].shape[0]
    M2 = hmm2['means'].shape[0]
    K = M1 + M2
    
    concatedHMM['name'] = hmm1['name'] + hmm2['name']
    concatedHMM['startprob'] = np.zeros((K + 1, 1))
    concatedHMM['transmat'] = np.zeros((K + 1, K + 1))
    concatedHMM['means'] = np.vstack((hmm1['means'],hmm2['means']))
    concatedHMM['covars'] = np.vstack((hmm1['covars'],hmm2['covars']))
        
    
    start1 = hmm1['startprob'].reshape(-1,1)
    start2 = hmm2['startprob'].reshape(-1,1)
    
    concatedHMM['startprob'][:hmm1['startprob'].shape[0]-1,:] = start1[:-1,:]
    concatedHMM['startprob'][hmm1['startprob'].shape[0]-1:,:] = np.dot(start1[-1,0],start2)
    trans = concatedHMM['transmat']
    trans1 = hmm1['transmat']
    trans2 = hmm2['transmat']

    trans[:trans1.shape[0]-1,:trans1.shape[1]-1] = trans1[:-1,:-1]
    temp = trans1[:-1,-1].reshape(-1,1)
    trans[:trans1.shape[0]-1,trans1.shape[1]-1:] = \
                            np.dot(temp,start2.T)
    trans[trans1.shape[0]-1:,trans1.shape[1]-1:] = trans2
    concatedHMM['transmat'] = trans    
    
    return concatedHMM


# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    forward_prob = np.zeros(log_emlik.shape)
    forward_prob[0,:] = log_startprob.T + log_emlik[0]

    for i in range(1, forward_prob.shape[0]):
        for j in range(forward_prob.shape[1]):
            forward_prob[i,j] = logsumexp(forward_prob[i - 1,] + log_transmat[:,j]) + log_emlik[i,j]
    return forward_prob

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    # backward_prob = np.zeros(log_emlik.shape)
    # for i in reversed(range(backward_prob.shape[0] - 1)):
    #     for j in range(backward_prob.shape[1]):
    #         backward_prob[i,j] = logsumexp(log_transmat[j,:-1] + log_emlik[i + 1,:] + backward_prob[i + 1,:])
    # return backward_prob
    log_b = np.zeros(log_emlik.shape)# + 1.0 / log_emlik.shape[1]
    for n in reversed(range(log_emlik.shape[0] - 1)):
        for i in range(log_emlik.shape[1]):
            log_b[n, i] = logsumexp(log_transmat[i,:] + log_emlik[n + 1, :] + log_b[n + 1,:])
    return log_b

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    # viterbi_loglik = np.zeros(log_emlik.shape)
    # viterbi_b_matrix  = np.zeros(log_emlik.shape, dtype=int)

    # viterbi_loglik[0] = log_startprob[:-1] + log_emlik[0]

    # for i in range(1, viterbi_loglik.shape[0] - 1):
    #     for j in range(viterbi_loglik.shape[1]):
    #         viterbi_loglik[i,j] = np.max(viterbi_loglik[i-1,:] + log_transmat[:-1, j]) + log_emlik[i,j]
    #         viterbi_b_matrix[i,j] = np.argmax(viterbi_loglik[i - 1,:] + log_transmat[:-1,j])

    # viterbi_path = [np.argmax(viterbi_b_matrix[-1])]
    # for i in reversed(range(viterbi_b_matrix[0] - 1)):
    #     viterbi_path.append(viterbi_b_matrix[i, viterbi_path[-1]])
    # viterbi_path.reverse()

    # return np.max(viterbi_loglik[-1], np.array(viterbi_path))
    B = np.zeros(log_emlik.shape, dtype = int)
    V = np.zeros(log_emlik.shape)
    V[0] = log_startprob.flatten() + log_emlik[0]

    for n in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            V[n][j] = np.max(V[n - 1,:] + log_transmat[:,j]) + log_emlik[n, j]
            B[n][j] = np.argmax(V[n - 1,:] + log_transmat[:,j])

    # Backtrack to take viteri path
    # viterbi_path = viterbiBacktrack(B, np.argmax(V[ log_emlik.shape[0] - 1]))

    lastIdx = np.argmax(V[log_emlik.shape[0] - 1])

    viterbi_path = [lastIdx]
    for i in reversed(range(1, B.shape[0])):
        viterbi_path.append(B[i, viterbi_path[-1]])
    viterbi_path.reverse()
    viterbi_path = np.array(viterbi_path)

    return np.max(V[ log_emlik.shape[0] - 1]), viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    # log_gamma = log_alpha + log_beta - logsumexp(log_alpha[log_alpha.shape[0] - 1])
    # return log_gamma

    N = len(log_alpha)
    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[N - 1])
    return log_gamma



def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    
    gamma = np.exp(log_gamma)
    means = np.zeros((log_gamma.shape[1], X.shape[1]))
    covars = np.zeros(means.shape)

    for i in range(means.shape[0]):
        gamma_sum = np.sum(gamma[:,i])
        means[i] = np.sum(gamma[:,i].reshape(-1, 1) * X, axis = 0) / gamma_sum
        covars[i] = np.sum(gamma[:,i].reshape(-1, 1) * (X - means[i])**2, axis = 0) / gamma_sum
        covars[i, covars[i] < varianceFloor] = varianceFloor
    return (means, covars)