# functions from previous assignments
from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy.signal import hamming
from lab1_tools import *
from scipy.fftpack.realtransforms import dct

# Function given by the exercise ----------------------------------

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.
    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """

    # check if i+winlen > len(samples):

    result = []
    for i in range(0,len(samples),winshift):
        if(i+winlen > len(samples)): break
        result.append(samples[i:i+winlen])
    return np.array(result)
    # return np.array([samples[i:i+winlen] for i in range(0,len(samples),winshift)])
    
def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    return lfilter([1, -p], [1], input)

def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    return input * hamming(input.shape[1], sym=0)

def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    freq = fft(input, nfft)
    return freq.real**2 + freq.imag**2

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = input.shape[1]
    tr_filter = trfbank(samplingrate, nfft)
    return np.log(np.dot(input, tr_filter.transpose()))

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    return dct(input)[:,0:nceps]

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path thtough AD

    Note that you only need to define the first output for this exercise.
    """

    AD = np.zeros(dist.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            AD[i,j] = dist[i,j] + min(AD[i - 1, j], AD[i, j - 1], AD[i - 1, j - 1])
    
    d = AD[-1, -1]/(x.shape[0] + y.shape[0])

    return d, dist, AD

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

    B = np.zeros(log_emlik.shape, dtype = int)
    V = np.zeros(log_emlik.shape)
    V[0] = log_startprob.flatten() + log_emlik[0]

    for n in range(1, log_emlik.shape[0]):
        for j in range(log_emlik.shape[1]):
            V[n][j] = np.max(V[n - 1,:] + log_transmat[:,j]) + log_emlik[n, j]
            B[n][j] = np.argmax(V[n - 1,:] + log_transmat[:,j])

    lastIdx = np.argmax(V[log_emlik.shape[0] - 1])

    viterbi_path = [lastIdx]
    for i in reversed(range(1, B.shape[0])):
        viterbi_path.append(B[i, viterbi_path[-1]])
    viterbi_path.reverse()
    viterbi_path = np.array(viterbi_path)

    return np.max(V[ log_emlik.shape[0] - 1]), viterbi_path

def log_multivariate_normal_density_diag(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model

    Args:
        X: array like, shape (n_observations, n_features)
        means: array like, shape (n_components, n_features)
        covars: array like, shape (n_components, n_features)

    Output:
        lpr: array like, shape (n_observations, n_components)
    From scikit-learn/sklearn/mixture/gmm.py
    """
    n_samples, n_dim = X.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(X, (means / covars).T)
                  + np.dot(X ** 2, (1.0 / covars).T))
    return lpr

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