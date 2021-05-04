import numpy as np
from lab2_proto import *


data = np.load('lab2_data.npz', allow_pickle=True)['data']
phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

# print(phoneHMMs)
# print(list(sorted(phoneHMMs.keys())))

# print(phoneHMMs['ah']['covars'].shape)   

isolated = {}
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

##Concatenating the phonemes with silence in the begining and end
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# print(isolated)

wordHMMs = {}
wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])
# print(wordHMMs['o'])

best_model = {}
accuracy = 0
for i,dp in enumerate(data):
    maxlog_lik = None
    for digit in wordHMMs.keys():
        obslogic = log_multivariate_normal_density_diag(dp['lmfcc'], wordHMMs[digit]['means'], wordHMMs[digit]['covars'])
        logalpha = forward(obslogic,np.log(wordHMMs[digit]['startprob']), np.log(wordHMMs[digit]['transmat']))
        loglik = logsumexp(logalpha[-1])

        if maxlog_lik is None or maxlog_lik < loglik:
            best_model[i] = digit
            maxlog_lik = loglik
        
        if dp['digit'] == best_model[i]:
            accuracy+=1

print("The accuracy of the predictions has been: " + str(np.round(accuracy / len(data) * 100, 2)) + "%")
