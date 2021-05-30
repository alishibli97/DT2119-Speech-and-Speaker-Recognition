import numpy as np
from lab3_proto import *
from helper_functions import *
from lab1_tools import *
import os

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

# X = example_data['lmfcc']
# means = utteranceHMM['means']
# covars = utteranceHMM['covars']
# log_likelihood = log_multivariate_normal_density_diag(X,means,covars)

# log_startprob = np.log(utteranceHMM['startprob'][:-1])
# log_transmat = np.log(utteranceHMM['transmat'][:-1,:-1])

# prob,viterbiStateTrans = viterbi(log_likelihood,log_startprob,log_transmat)
# # print(viterbiStateTrans)

# vpath = [stateTrans[i] for i in viterbiStateTrans]
# print(vpath)
# viterbiStateTrans = forcedAlignment(example_data['lmfcc'], utteranceHMM, phoneTrans)
# vpath = [stateTrans[i] for i in viterbiStateTrans]
# print(vpath)
# frames = frames2trans(vpath, outfilename='z43a.lab')
# print(frames)

filenames = ['Data/traindata.npz', 'Data/testdata.npz']
sets = ['/Users/nandakishorprabhu/Documents/Studies/DT2119/Code/tidigits/disc_4.1.1/tidigits/train', '/Users/nandakishorprabhu/Documents/Studies/DT2119/Code/tidigits/disc_4.2.1/tidigits/test']
for idx, file_name in enumerate(filenames):
    if not os.path.isfile(file_name):
        data = []
        for root, dirs, files in os.walk(sets[idx]):
            for file in files:
                if file.endswith('.wav'):
                    filename = os.path.join(root, file)
                    samples, samplingrate = loadAudio(filename)
                    lmfcc, mspec = mfcc(samples)
                    wordTrans = list(path2info(filename)[2])
                    phoneTrans = words2phones(wordTrans, prondict.prondict)
                    targets = forcedAlignment(lmfcc, utteranceHMM, phoneTrans)
                    data.append({'filename': filename, 'lmfcc': lmfcc, 'mspec': mspec, 'targets': targets})
            if file_name == 'traindata.npz':
                np.savez(PATH + file_name, traindata=data)
            elif file_name == 'testdata.npz':
                np.savez(PATH + file_name, testdata=data)

print("!TEST!")