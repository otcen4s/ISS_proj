import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import math
from scipy.io import wavfile


from scipy.stats import pearsonr
from scipy.signal import spectrogram

#np.set_printoptions(threshold=np.inf)

#----------------------------------------------------------------------------#


def spectrogram3task():
    signalData, Fs = sf.read('../sentences/sa1.wav')

    signalData = signalData - signalData.mean() #averaging
    wlen = 25e-3 * Fs
    wshift = 10e-3 * Fs
    woverlap = wlen - wshift
    N = 512

    win = np.hamming(wlen)

    f, t, sgr = spectrogram(signalData, Fs, window=win, noverlap=woverlap, nfft=N-1)

    P = 10 * np.log10((abs(sgr) ** 2)+1e-20)

    plt.figure(figsize=(6,3))
    plt.pcolormesh(t, f, P)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvencia [Hz]')
    plt.gca().set_title('sa1')
    plt.plot()
    plt.show()

#----------------------------------------------------------------------------#



def getMatrixF(sentence, sentenceName):

    signalData, Fs = sf.read(sentence)
    
    signalData = signalData - signalData.mean() #averaging
    wlen = 25e-3 * Fs
    wshift = 10e-3 * Fs
    woverlap = wlen - wshift
    N = 512
    
    win = np.hamming(wlen)
    
    f, t, sgr = spectrogram(signalData, Fs, window=win, noverlap=woverlap, nfft=N-1)

    P = 10 * np.log10((abs(sgr) ** 2)+1e-20)

    t = np.arange(signalData.size) / Fs
    
    plt.figure(figsize=(6,3))
    plt.plot(t, signalData)
    plt.gca().set_ylabel('$Signal$')
    plt.gca().set_xlabel('$Čas [s]$')
    plt.gca().set_title('"precariously" and "privations" vs ' + sentenceName)
    plt.show()
    
    row = 16
    col = 256

    B = 0
    cnt = 0

    A = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            if B * 16 == j:
                while cnt != 16:
                    A[i][cnt + j] = 1
                    cnt += 1
                j = j + cnt
        B += 1
        cnt = 0
                    
    F = A @ P
    
    t = np.arange(F[0].size) / 100
    f = np.arange(np.shape(F)[0]) 

    plt.figure(figsize=(6,3))
    
    
    plt.pcolormesh(t,f,F)
  
    plt.gca().set_ylabel('Parametre')
    plt.gca().set_xlabel('$Čas [s]$')
    plt.gca().invert_yaxis()
    plt.plot()
    plt.show()
    
    return F

#----------------------------------------------------------------------------#

def query(queryWav, F, label, color, senetnceWav):

    signalData, Fs = sf.read(queryWav)

    signalData = signalData - signalData.mean() #averaging
    wlen = 25e-3 * Fs
    wshift = 10e-3 * Fs
    woverlap = wlen - wshift

    N = 512

    win = np.hamming(wlen)

    t, f, sgr = spectrogram(signalData, Fs, window=win, noverlap=woverlap, nfft=N-1)

    P = 10 * np.log10((abs(sgr) ** 2)+1e-20)

    row = 16
    col = 256

    B = 0
    cnt = 0

    A = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            if B * 16 == j:
                while cnt != 16:
                    A[i][cnt + j] = 1
                    cnt += 1
                j = j + cnt
        B += 1
        cnt = 0
                    
    Q = A @ P

    F = F.transpose()
    Q = Q.transpose()

    Qlen = len(Q)
    Flen = len(F)

    scores = []
    pp = 0

    while Flen > pp + Qlen:
        coefficients = []
        for k in range(0, len(Q)):
            if Qlen + pp > Flen:
                break
            coefficient = pearsonr(Q[k], F[k + pp])[0]
           # if not np.isnan(coefficient):
            coefficients.append(coefficient)
           # else:
             #   coefficients.append(0)
        score = np.sum(coefficients)
        scorePercent = score/len(Q)
        scores.append(scorePercent)
        pp += 5

    
    signalData, Fs = sf.read(senetnceWav)

    signalData = signalData - signalData.mean() #averaging
    
    scoresPerSec = (len(signalData) / len(scores)) / Fs
    
    t = (np.arange(len(scores))) * scoresPerSec
    
    
    plt.gca().set_ylabel('$Skore$')
    plt.gca().set_xlabel('$Čas [s]$')
    plt.ylim(top=1.0)
    plt.plot(t, scores, color, label=label)
    plt.legend(loc="lower right")
    
    return scores
    
#----------------------------------------------------------------------------#

def findHits(score, sentenceWav, queryWav, qName, sName):
    signalData, Fs = sf.read(sentenceWav)

    signalData = signalData - signalData.mean() #averaging
    
    signalDataQuery, Fs = sf.read(queryWav)

    signalDataQuery = signalDataQuery - signalDataQuery.mean() #averaging

    index = 0
    for i in range(0, len(score)):
        if score[i] >= 0.80:
            possibleHits = []
            for k in range(i * 160 * 5 , len(signalDataQuery) + i * 160 * 5):
                possibleHits.append(signalData[k])
            
            possibleHits = np.array(possibleHits)
            wavfile.write(qName+sName+'.wav', 16000, possibleHits)
            sName += str(index)
            index += 1
            




#----------------------------------------------------------------------------#
spectrogram3task()

F = getMatrixF('../sentences/sa1.wav', 'sa1')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sa1.wav')
findHits(score, '../sentences/sa1.wav', '../queries/q1.wav', 'q1', 'sa1')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sa1.wav')
findHits(score, '../sentences/sa1.wav', '../queries/q2.wav', 'q2', 'sa1')
plt.show()

F = getMatrixF('../sentences/sa2.wav', 'sa2')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sa2.wav')
findHits(score, '../sentences/sa2.wav', '../queries/q1.wav', 'q1', 'sa2')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sa2.wav')
findHits(score, '../sentences/sa2.wav', '../queries/q2.wav', 'q2', 'sa2')
plt.show()

F = getMatrixF('../sentences/si595.wav', 'si595')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/si595.wav')
findHits(score, '../sentences/si595.wav', '../queries/q1.wav', 'q1', 'si595')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/si595.wav')
findHits(score, '../sentences/si595.wav', '../queries/q2.wav', 'q2', 'si595')
plt.show()

F = getMatrixF('../sentences/sil1225.wav', 'sil1225')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sil1225.wav')
findHits(score, '../sentences/sil1225.wav', '../queries/q1.wav', 'q1', 'sil1225')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sil1225.wav')
findHits(score, '../sentences/sil1225.wav', '../queries/q2.wav', 'q2', 'sil1225')
plt.show()

F = getMatrixF('../sentences/sil1855.wav', 'sil1855')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sil1855.wav')
findHits(score, '../sentences/sil1855.wav', '../queries/q1.wav', 'q1', 'sil1855')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sil1855.wav')
findHits(score, '../sentences/sil1855.wav', '../queries/q2.wav', 'q2', 'sil1855')
plt.show()

F = getMatrixF('../sentences/sx55.wav', 'sx55')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sx55.wav')
findHits(score, '../sentences/sx55.wav', '../queries/q1.wav', 'q1', 'sx55')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sx55.wav')
findHits(score, '../sentences/sx55.wav', '../queries/q2.wav', 'q2', 'sx55')
plt.show()

F = getMatrixF('../sentences/sx145.wav', 'sx145')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sx145.wav')
findHits(score, '../sentences/sx145.wav', '../queries/q1.wav', 'q1', 'sx145')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sx145.wav')
findHits(score, '../sentences/sx145.wav', '../queries/q2.wav', 'q2', 'sx145')
plt.show()

F = getMatrixF('../sentences/sx235.wav', 'sx235')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sx235.wav')
findHits(score, '../sentences/sx235.wav', '../queries/q1.wav', 'q1', 'sx235')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sx235.wav')
findHits(score, '../sentences/sx235.wav', '../queries/q2.wav', 'q2', 'sx235')
plt.show()

F = getMatrixF('../sentences/sx325.wav', 'sx325')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sx325.wav')
findHits(score, '../sentences/sx325.wav', '../queries/q1.wav', 'q1', 'sx325')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sx325.wav')
findHits(score, '../sentences/sx325.wav', '../queries/q2.wav', 'q2', 'sx325')
plt.show()

F = getMatrixF('../sentences/sx415.wav', 'sx415')
score = query('../queries/q1.wav', F, "precariously", "-r", '../sentences/sx415.wav')
findHits(score, '../sentences/sx415.wav', '../queries/q1.wav', 'q1', 'sx415')

score = query('../queries/q2.wav', F, "privations", "-g", '../sentences/sx415.wav')
findHits(score, '../sentences/sx415.wav', '../queries/q2.wav', 'q2', 'sx415')
plt.show()
