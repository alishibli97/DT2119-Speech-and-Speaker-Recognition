import numpy as np

data = np.load('lab2_data.npz', allow_pickle=True)['data']


phoneHMMs = np.load('lab2_models_onespkr.npz', allow_pickle=True)['phoneHMMs'].item()
# phoneHMMs = np.load('lab2_models_all.npz', allow_pickle=True)['phoneHMMs'].item()

# print(list(sorted(phoneHMMs.keys())))

# print(phoneHMMs['ah'].keys())

prondict = {}
prondict['o'] = ['ow']
prondict['z'] = ['z', 'iy', 'r', 'ow']
prondict['1'] = ['w', 'ah', 'n']
# ...

isolated = {}
for digit in prondict.keys():
    isolated[digit] = ['sil'] + prondict[digit] + ['sil']

# print(isolated)

wordHMMs = {}
wordHMMs['o'] = concatHMMs(phoneHMMs, isolated['o'])
print(wordHMMs)