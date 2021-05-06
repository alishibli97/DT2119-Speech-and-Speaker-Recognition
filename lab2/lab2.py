import numpy as np
from lab2_proto import *
from matplotlib import pyplot as plt
import psutil

def _verification(criteria,example,wordHMMs):
    if criteria=="concatenation":
        X = example['lmfcc']
        means = wordHMMs[0]['means']
        covars = wordHMMs[0]['covars']

        prac = example['obsloglik']
        theor = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs[0]['means'], wordHMMs[0]['covars'])

        # print(np.allclose(theor, example["obsloglik"]))
        plot_fig = True
        if plot_fig:
            plt.rcParams['figure.figsize'] = [15, 3]
            plt.subplot(121).set_title("Observation Likelihood")
            plt.pcolormesh(theor.T)
            plt.colorbar()
            plt.yticks(range(1,10), verticalalignment='top')
            plt.xlabel('Frames')
            plt.ylabel('States')
            plt.subplot(122).set_title("Example")
            plt.pcolormesh(example["obsloglik"].T)
            plt.show()

    elif criteria=="forward": # need to plot the forward and example verification
        obsloglik = log_multivariate_normal_density_diag(example['obsloglik'], wordHMMs[0]['means'], wordHMMs[0]['covars'])
        prac = forward(obsloglik, np.log(wordHMMs[0]['startprob'][:-1]),np.log(wordHMMs[0]['transmat'][:-1,:-1]))
        theor = example['logalpha']

        plot_fig = True
        if plot_fig:
            plt.rcParams['figure.figsize'] = [15, 3]
            plt.subplot(121).set_title("Observation Likelihood")
            plt.pcolormesh(theor.T)
            plt.colorbar()
            plt.yticks(range(1,10), verticalalignment='top')
            plt.xlabel('Frames')
            plt.ylabel('States')
            plt.subplot(122).set_title("Example")
            plt.pcolormesh(example["obsloglik"].T)
            plt.show()

    elif criteria=="viterbi":
        print("k")

    elif criteria=="backward":
        print("k")

def _forward(prondict,wordHMMs):
    correct = 0
    for key in prondict.keys():
        sequences = [x['lmfcc'] for x in data if x['digit']==key]
        for sequence in sequences:
            maxProb = None
            index = None
            for i in range(len(wordHMMs)):
                obsloglik = log_multivariate_normal_density_diag(sequence, wordHMMs[i]['means'], wordHMMs[i]['covars'])
                forward_probs = forward(obsloglik, np.log(wordHMMs[i]['startprob'][:-1]),np.log(wordHMMs[i]['transmat'][:-1,:-1]))
                obs_log_prob = logsumexp(forward_probs[-1])
                if(maxProb is None or obs_log_prob > maxProb):
                    maxProb = obs_log_prob
                    index = i
        pred = list(prondict.keys())[index] # predicted letter
        if key==pred: correct+=1
        print(f"For {key} predicted: {pred}")
    print(f"Total accuracy: {correct*100/len(prondict)}%")

def _viterbi(prondict,wordHMMs):
    correct = 0
    for key in prondict.keys():
        sequences = [x['lmfcc'] for x in data if x['digit']==key]
        for sequence in sequences:
            maxProb = None
            index = None
            for i in range(len(wordHMMs)):
                obsloglik = log_multivariate_normal_density_diag(sequence, wordHMMs[i]['means'], wordHMMs[i]['covars'])
                viterbi_loglik, viterbi_path = viterbi(obsloglik,np.log(wordHMMs[i]['startprob'][:-1]), np.log(wordHMMs[i]['transmat'][:-1,:-1]))
                if(maxProb is None or viterbi_loglik > maxProb):
                    index = i
                    maxProb = viterbi_loglik
        pred = list(prondict.keys())[index] # predicted letter
        if key==pred: correct+=1
        print(f"For {key} predicted: {pred}")
    print(f"Total accuracy: {correct*100/len(prondict)}%")

def _backward():
    obsloglik = log_multivariate_normal_density_diag(example['lmfcc'], wordHMMs[0]['means'], wordHMMs[0]['covars'])
    log_startprob = np.log(wordHMMs[0]['startprob'][:-1])
    log_transmat = np.log(wordHMMs[0]['transmat'][:-1,:-1])
    beta = backward(obsloglik, log_startprob, log_transmat)
    if np.allclose(beta, example['logbeta']):
        _ = plt.rcParams['figure.figsize'] = [15, 3]
    _ = plt.subplot(121).set_title("Log Beta")
    _ = plt.pcolormesh(beta.T)
    _ = plt.colorbar()
    _ = plt.yticks(range(1,10), verticalalignment='top')
    _ = plt.xlabel('Frames in sample')
    _ = plt.ylabel('sil-o-sil (9 states)')
    plt.show()

def _retrain():
    print("k")

if __name__=="__main__":
    data = np.load('lab2_data.npz', allow_pickle=True)['data']
    example = np.load('lab2_example.npz', allow_pickle=True)['example'].item()
    phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()
    phoneOneSpeaker = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()

    # discard sp model
    del phoneHMMs['sp']
    del phoneOneSpeaker['sp']

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

    wordHMMs = []
    wordHMMsOne = []
    for key in prondict.keys():
        prondict[key] = ['sil'] + prondict[key] + ['sil']
        wordHMMs.append(concatHMMs(phoneHMMs,prondict[key]))
        wordHMMsOne.append(concatHMMs(phoneHMMs,prondict[key]))

    # _concatination(example,wordHMMs)
    # _verification("forward", example, wordHMMs)
    # _forward(prondict,wordHMMsOne)
    

    _viterbi(prondict,wordHMMs)
    print('The CPU usage is: ', psutil.cpu_percent(4))

