import librosa as lb
import numpy as np
from lab3_proto import *
from helper_functions import *
from lab1_tools import *
import os
# from pysndfile import sndio
import math
import pickle
import warnings

from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy.signal import hamming
from lab1_tools import *
from scipy.fftpack.realtransforms import dct
from sklearn.preprocessing import StandardScaler

def path2info(path):
    """
    path2info: parses paths in the TIDIGIT format and extracts information
               about the speaker and the utterance

    Example:
    path2info('tidigits/disc_4.1.1/tidigits/train/man/ae/z9z6531a.wav')
    """
    rest, filename = os.path.split(path)
    rest, speakerID = os.path.split(rest)
    rest, gender = os.path.split(rest)
    digits = filename[:-5]
    repetition = filename[-5]
    return gender, speakerID, digits, repetition

def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default pysndfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This is avoided by specifying
    the option dtype=np.int16 which keeps both the original data type and range
    of values.
    """
    sndobj = lb.load(filename)#, dtype=np.int16)
    samplingrate = sndobj[1]
    samples = np.array(sndobj[0])
    return samples, samplingrate

def frames2trans(sequence, outfilename=None, timestep=0.01):
    """
    Outputs a standard transcription given a frame-by-frame
    list of strings.

    Example (using functions from Lab 1 and Lab 2):
    phones = ['sil', 'sil', 'sil', 'ow', 'ow', 'ow', 'ow', 'ow', 'sil', 'sil']
    trans = frames2trans(phones, 'oa.lab')

    Then you can use, for example wavesurfer to open the wav file and the transcription
    """
    sym = sequence[0]
    start = 0
    end = 0
    trans = ''
    for t in range(len(sequence)):
        if sequence[t] != sym:
            trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
            sym = sequence[t]
            start = end
        end = end + timestep
    trans = trans + str(start) + ' ' + str(end) + ' ' + sym + '\n'
    if outfilename != None:
        with open(outfilename, 'w') as f:
            f.write(trans)
    return trans



# Function given by the exercise ----------------------------------

def lifter(mfcc, lifter=22):
    """
    Applies liftering to improve the relative range of MFCC coefficients.
       mfcc: NxM matrix where N is the number of frames and M the number of MFCC coefficients
       lifter: lifering coefficient
    Returns:
       NxM array with lifeterd coefficients
    """
    nframes, nceps = mfcc.shape
    cepwin = 1.0 + lifter/2.0 * np.sin(np.pi * np.arange(nceps) / lifter)
    l = np.multiply(mfcc, np.tile(cepwin, nframes).reshape((nframes,nceps)))
    return l

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def trfbank(fs, nfft, lowfreq=133.33, linsc=200/3., logsc=1.0711703, nlinfilt=13, nlogfilt=27, equalareas=False):
    """Compute triangular filterbank for MFCC computation.
    Inputs:
    fs:         sampling frequency (rate)
    nfft:       length of the fft
    lowfreq:    frequency of the lowest filter
    linsc:      scale for the linear filters
    logsc:      scale for the logaritmic filters
    nlinfilt:   number of linear filters
    nlogfilt:   number of log filters
    Outputs:
    res:  array with shape [N, nfft], with filter amplitudes for each column.
            (N=nlinfilt+nlogfilt)
    From scikits.talkbox"""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    if equalareas:
        heights = np.ones(nfilt)
    else:
        heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank


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


PATH = '/Users/nandakishorprabhu/Documents/Studies/DT2119/Code'
phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
phones = sorted(phoneHMMs.keys())
nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
# stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
# f = open("stateList.txt","w")
# for state in stateList:
#     f.write(f"{state}\n")
stateList = open("stateList.txt", "r").readlines()
stateList = [state.strip() for state in stateList]

example_data = np.load("lab3_example.npz", allow_pickle=True)['example']
example_data = dict(enumerate(example_data.flatten(), 1))[1]
filename = example_data['filename']
wordTrans = list(path2info(filename)[2])
# print(wordTrans)

prondict = {}
prondict['o'] = ['ow']
prondict['z'] = ['z', 'iy', 'r', 'ow']
prondict['1'] = ['w', 'ah', 'n']
prondict['2'] = ['t', 'uw']
prondict['3'] = ['th', 'r', 'iy']
prondict['4'] = ['f', 'ao', 'r']
prondict['5'] = ['f', 'ay', 'v']
prondict['6'] = ['s', 'ih', 'k', 's']
prondict['7'] = ['s', 'eh', 'v', 'ah', 'n']
prondict['8'] = ['ey', 't']
prondict['9'] = ['n', 'ay', 'n']

phoneTrans = words2phones(wordTrans, prondict)
# print(phoneTrans)

utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
# print(utteranceHMM)

# stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans for stateid in range(nstates[phone])]
# f = open("stateTrans.txt","w")
# for state in stateTrans:
#     f.write(f"{state}\n")
stateTrans = open("stateTrans.txt", "r").readlines()
stateTrans = [state.strip() for state in stateTrans]
# print(stateTrans[10])

def prepare_features(datapath, saveName):
    traindata = []
#     stateList = list()
#     with open(stateListPath) as f:
#         for line in f:
#             stateList.append(line.strip('\n'))

    totalfiles = 0
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith('.wav'):
                totalfiles += 1
    count = 0
    exit=False
    for root, dirs, files in os.walk(datapath):
        for file in files:
            if file.endswith('.wav'):
                filename = os.path.join(root, file)
                samples, samplingrate = loadAudio(filename)
                print(samples.shape)
                if count>10: 
                    exit = True
                    break
                count+=1
                lmfcc = mfcc(samples)
                mspecs = mspec(samples)
                wordTrans = list(path2info(filename)[2])
                phoneTrans = words2phones(wordTrans, prondict, addShortPause=False)
                targets = forcedAlignment(lmfcc, utteranceHMM, phoneTrans)

                traindata.append({'filename': filename,
                                 'lmfcc': lmfcc,
                                 'mspec': mspecs,
                                 'targets':targets})
        if exit: break
    return traindata


traindata = np.load('traindata.npz', allow_pickle = True)
traindata = dict(zip(("{}".format(k) for k in traindata), (traindata[k] for k in traindata)))
# print(traindata)
male_speakers = np.unique([traindata['traindata'][i]["filename"].split('/')[6] 
                      for i in range(len(traindata['traindata']))
                      if traindata['traindata'][i]["filename"].split('/')[4] == 'man'])
female_speakers = np.unique([traindata['traindata'][i]["filename"].split('/')[6] 
                      for i in range(len(traindata['traindata'])) 
                      if traindata['traindata'][i]["filename"].split('/')[4] == 'woman'])


m_train_speakers = male_speakers[0:math.floor(len(male_speakers)*0.9)]
m_valid_speakers = male_speakers[math.floor(len(male_speakers)*0.9):]

# print(len(m_train_speakers))
# print(m_valid_speakers)

m_train_data =  [traindata['traindata'][i] for i in range(len(traindata['traindata']))
                  if traindata['traindata'][i]["filename"].split('/')[6] in m_train_speakers]

m_valid_data =  [traindata['traindata'][i] for i in range(len(traindata['traindata'])) 
                  if traindata['traindata'][i]["filename"].split('/')[6] in m_valid_speakers]

w_train_speakers = female_speakers[0:math.floor(len(female_speakers)*0.9)]
w_valid_speakers = female_speakers[math.floor(len(female_speakers)*0.9):]

w_train_data =  [traindata['traindata'][i] for i in range(len(traindata['traindata'])) 
                  if traindata['traindata'][i]["filename"].split('/')[6] in w_train_speakers]

w_valid_data =  [traindata['traindata'][i] for i in range(len(traindata['traindata'])) 
                  if traindata['traindata'][i]["filename"].split('/')[6] in w_valid_speakers]

training_data = m_train_data + w_train_data
validation_data = m_valid_data + w_valid_data

# print("Training Data : ", (len(training_data)/len(traindata['traindata']))*100)
# print("Validation Data :", (len(validation_data)/len(traindata['traindata']))*100)


def dynamize_features(data, feature_type):
    for sample in (data):
        # print(sample)
        dynamic_features = []
        max_idx = len(sample[feature_type]) - 1
        for idx, feature in enumerate(sample[feature_type]):
            dynamic_feature = np.zeros((7, feature.shape[0]))

            dynamic_feature[0] = sample[feature_type][np.abs(idx - 3)]
            dynamic_feature[1] = sample[feature_type][np.abs(idx - 2)]
            dynamic_feature[2] = sample[feature_type][np.abs(idx - 1)]
            dynamic_feature[3] = sample[feature_type][idx]
            dynamic_feature[4] = sample[feature_type][max_idx - np.abs(max_idx - (idx + 1))]
            dynamic_feature[5] = sample[feature_type][max_idx - np.abs(max_idx - (idx + 2))]
            dynamic_feature[6] = sample[feature_type][max_idx - np.abs(max_idx - (idx + 3))]
            dynamic_features.append(dynamic_feature)
        sample['dynamic_'+feature_type] = np.array(dynamic_features)
    return data

training_data_lmfcc = dynamize_features(training_data[:], 'lmfcc')
training_data_mspec = dynamize_features(training_data[:], 'mspec')

validation_data_lmfcc = dynamize_features(validation_data[:], 'lmfcc')
validation_data_mspec = dynamize_features(validation_data[:], 'mspec')

print("Original Features ", training_data_lmfcc[0]["lmfcc"].shape)
print("Dynamic Features ", training_data_lmfcc[0]['dynamic_lmfcc'].shape)

print("Total number of data points is", len(training_data_lmfcc))


mspec_train_x = []
mspec_train_y = []

mspec_val_x = []
mspec_val_y = []

for train in training_data_mspec:
    mspec_train_x.extend(train['dynamic_mspec'])
    mspec_train_y.extend(train['targets'])

for val in validation_data_mspec:
    mspec_val_x.extend(val['dynamic_mspec'])
    mspec_val_y.extend(val['targets'])
    
mspec_train_x = np.array(mspec_train_x)
mspec_train_y = np.array(mspec_train_y)
mspec_val_x = np.array(mspec_val_x)
mspec_val_y = np.array(mspec_val_y)

from keras.utils import to_categorical
mspec_train_y = to_categorical(mspec_train_y, num_classes=40)
mspec_val_y = to_categorical(mspec_val_y, num_classes=40)


train_X = mspec_train_x
train_y = mspec_train_y

val_X = mspec_val_x
val_y = mspec_val_y


train_X = train_X.astype('float32')
val_X = val_X.astype('float32')

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(256, input_shape=(1,train_X.shape[1])))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(train_y.shape[1], activation="softmax"))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

print(train_X.shape,train_y.shape)
print(val_X.shape,val_y.shape)

history = model.fit(
    train_X, 
    train_y, 
    validation_data=(val_X, val_y), 
    epochs=10
)