import numpy as np

# phoneHMMs = np.load('lab2_models_all.npz',allow_pickle=True)['phoneHMMs'].item()
# phones = sorted(phoneHMMs.keys())
# nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
# stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
# f = open("stateList.txt","w")
# for state in stateList:
#     f.write(f"{state}\n")

stateList = open("stateList.txt","r").readlines()
stateList = [state.strip() for state in stateList]

