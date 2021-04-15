import numpy as np
from lab1_proto import *
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.simplefilter("ignore")

example = np.load('lab1_example.npz', allow_pickle=True)['example'].item()
data = np.load('lab1_data.npz', allow_pickle=True)['data']

testing1 = False
testing2 = True

def plot_example(sample_dct):
    # when plotting do transpose of the matrix
    plt.pcolormesh(sample_dct.T)
    plt.show()

def plot_data():
    sample_dct1 = mfcc(data[0]['samples'])
    sample_dct2 = mfcc(data[1]['samples'])
    sample_dct3 = mfcc(data[2]['samples'])
    sample_dct4 = mfcc(data[3]['samples'])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.pcolormesh(sample_dct1.T)
    ax1.set(title='dct1')
    ax1.grid()

    ax2.pcolormesh(sample_dct2.T)
    ax2.set(title='dct2')
    ax2.grid()

    ax3.pcolormesh(sample_dct3.T)
    ax3.set(title='dct3')
    ax3.grid()

    ax4.pcolormesh(sample_dct4.T)
    ax4.set(title='dct4')
    ax4.grid()
    
    fig.tight_layout()
    plt.show()

def plot_meshes(corr_matrix_mfcc,corr_matrix_mel):
    fig, ((ax1, ax2)) = plt.subplots(2, 1)
    
    ax1.pcolormesh(corr_matrix_mfcc)
    ax1.set(title='mfcc correlation matrix')
    ax1.grid()

    ax2.pcolormesh(corr_matrix_mel)
    ax2.set(title='mel correlation matrix')
    ax2.grid()

    fig.tight_layout()
    plt.show()

def plot_posteriors(posteriors):
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.pcolormesh(posteriors[0])
    ax1.set(title='mfcc correlation matrix')
    ax1.grid()

    ax2.pcolormesh(posteriors[1])
    ax2.set(title='mel correlation matrix')
    ax2.grid()

    ax3.pcolormesh(posteriors[2])
    ax3.set(title='mel correlation matrix')
    ax3.grid()

    ax4.pcolormesh(posteriors[3])
    ax4.set(title='mel correlation matrix')
    ax4.grid()

    fig.tight_layout()
    plt.show()

if testing1:
    # speech = example
    speech = data[0]

    winlen = int(speech['samplingrate'] * 20e-3)
    winshift = int(speech['samplingrate'] * 10e-3)

    sample_enframe = enframe(speech['samples'], winlen=winlen, winshift=winshift)
    sample_preemp = preemp(sample_enframe)
    sample_windowing = windowing(sample_preemp)
    sample_fft = powerSpectrum(sample_windowing,512)
    sample_mel = logMelSpectrum(sample_fft,speech['samplingrate'])
    sample_dct = cepstrum(sample_mel,13)
    sample_dct = lifter(sample_dct)
    plot(sample_dct)

if testing2:
    # plot_data()

    result_mfcc = mfcc(data[0]['samples'])
    result_mel = mspec(data[0]['samples'])
    for i in range(1,len(data)):
        np.append(result_mfcc,mfcc(data[i]['samples']))
        np.append(result_mel,mspec(data[i]['samples']))
    
    corr_matrix_mfcc = np.corrcoef(result_mfcc)
    corr_matrix_mel = np.corrcoef(result_mel)

    # plot_meshes(corr_matrix_mfcc,corr_matrix_mel)

    num_components = [4,8,16,32]
    for num_comp in num_components:
        gmm = GMM(n_components=num_comp,random_state=0).fit(result_mfcc)

        if(num_comp==32):
            X = np.array(mfcc(data[16]['samples']))
            for i in [17,38,39]:
                X = np.append(X, mfcc(data[i]['samples']),axis=0)
            
            posteriors = gmm.predict_proba(X)

            # plt.pcolormesh(posteriors)
            # plt.show()
    
    # X = mfcc(data[0]['samples'])
    # Y = mfcc(data[1]['samples'])

    D = np.zeros(shape=(44,44))

    for ii,data1 in enumerate(data):
        print(f"Iteration {ii}")
        for jj,data2 in enumerate(data):
            
            X = mfcc(data1['samples'])
            Y = mfcc(data2['samples'])

            local_distances = np.zeros(shape=(X.shape[0],Y.shape[0]))
            for i,x in enumerate(X):
                for j,y in enumerate(Y):
                    local_distances[i,j] = np.linalg.norm(x-y)
            
            d,LD,AD = dtw(X, Y, local_distances)
            D[ii,jj]=d
    

    # plt.pcolormesh(D)
    # plt.show()

    Z = linkage(D,method='complete')
    dn = dendrogram(Z,labels=tidigit2labels(data))
    plt.show()