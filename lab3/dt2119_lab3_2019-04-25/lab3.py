import numpy as np
from lab3_proto import *
from helper_functions import *
from lab1_tools import *
from prondict import prondict

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

print(example_data.keys())
# utteranceHMM

# log_multivariate_normal_density_diag()
# viterbi

# frames2trans(viterbiStateTrans, outfilename='z43a.lab')