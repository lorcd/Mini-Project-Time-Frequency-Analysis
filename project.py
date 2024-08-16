"""Scientific Computation Project 4
Your CID here:01818119
"""
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

#We first define some preliminary functions
def gabor(M,w,eta=82.5):
    """Gabor wavelet as defined in Alm & Walker, eqn 4.5
        Use if/as needed
    """
    '''M is the number of points centered around 0 that the function calculates
    gabor wavelet for.
    '''
    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / w
    wavelet = np.exp(1j * 2*np.pi * eta * x) * np.exp(-np.pi * x**2)
    output = (1/w) * wavelet
    return output

def gabor1(t, w=0.25, eta=20):
    '''This is the same as the given gabor function, except 
    this takes a time array as its input.'''
    t = t / w
    wavelet = np.exp(1j * 2*np.pi * eta * t) * np.exp(-np.pi * t**2)
    output = (1/w) * wavelet
    return output

def gn(tarray, s, k):
    '''This function will generate the sequence needed to do the CWT.
    tarray is the time points to be calculated for, s is the scale/dilation, 
    and k is the translation'''
    return (s**-1)*gabor1((s**-1)*(tarray - k*tarray[1]))

def signalGen(t, v0 = 80, v1 = 320, v2 = 640, v3 = 160):
    '''This function will generate the test signal given in equation
    4.7 in the paper for an array of time points. '''
    p1 = np.sin(2 * np.pi * v1 *t) * np.exp(-np.pi*((t-0.2)/0.1)**10)
    p2 = (np.sin(2 * np.pi * v1 *t) + 2* np.cos(2 * np.pi * v2 *t)) * np.exp(-np.pi*((t-0.5)/0.1)**10)
    p3 = (2*np.sin(2 * np.pi * v2 *t) - np.cos(2 * np.pi * v3 *t)) * np.exp(-np.pi*((t-0.8)/0.1)**10)
    return(p1+p2+p3)


def testSignal(inputs=()):
    """
    Compute spectrogram and scalogram of test signal from paper
    Use inputs to provide any needed information.
    """
    #First we plot the signal against time
    sr = 2000
    dur = 1
    N = int(sr * dur)
    tarray = np.linspace(0, dur, N)
    y= signalGen(tarray)
    
    plt.figure()
    plt.title("Test Signal (Eqn. 4.7)")
    plt.ylabel('test signal')
    plt.xlabel('time')
    plt.plot(tarray, y, 'x')
    
    #Compute and plot the spectogram
    f = plt.figure()
    freqs, ts, vals = sig.spectrogram(y, sr, window = np.linspace(0, 1) )
    
    plt.contourf(ts, freqs , vals, cmap = "Greys")
    plt.title('Test Signal (Eqn 4.7) - Spectogram')
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()
    
    cbar = plt.colorbar()
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    plt.show()
    
    #Compute and plot the scalogram   . I simply followed the logic in the notes. 
    #This produces a list of ten evenly spaced numbers in each octave.
    #These correspond to the frequencies in each octave, with octave stretch.
    ss = np.array([(2**(-0.1))**i for i in range(0, 41)]) 
    fgCorr = np.zeros((len(ss), N), float)

    for ns, s in enumerate(ss):
        for k in range(N):
            fgCorr[ns, k] = np.sum(y * np.conjugate(gn(tarray, s, k)))
            
    fig = plt.figure()
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    
    cp = plt.contourf(tarray, np.array([i for i in range(0, 41)]) , abs(fgCorr), cmap = "Greys")
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    
    positions = np.arange(0, 41, 10)
    labels = ("$v_{0} 2^{0}$",
              "$v_{0} 2^{1}$",
              "$v_{0} 2^{2}$",
              "$v_{0} 2^{3}$", 
              "$v_{0} 2^{4}$")
    plt.yticks(positions, labels) 
    ax.set_title('Test Signal (Eqn 4.7) - Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency: $v_{0}=80 Hz$')
    plt.show()
    
    #Alternatively we can use the sig.cwt function with the morlet wavelet fn
    #We include this to show this method gives identical results. In addition 
    #it is much faster than the previous method, so we use it in part 3.
    '''
    t, dt = np.linspace(0, 1, 2000, retstep=True)
    fs = 1/dt
    w = 50
    y = signalGen(t)
    freq = np.arange(80, 1281, 10)
    widths = w*fs / (2*freq*np.pi)
    cwtm = sig.cwt(y, sig.morlet2, widths, w=w)
    
    fig, ax = plt.subplots()
    plt.semilogy(base = 2)
    positions = np.array([0, 160, 320, 640, 1280])
    labels = ("$v_{0} 2^{0}$",
              "$v_{0} 2^{1}$",
              "$v_{0} 2^{2}$",
              "$v_{0} 2^{3}$", 
              "$v_{0} 2^{4}$")
    
    plt.yticks(positions, labels) 
    cp = plt.contourf(t, freq , np.abs(cwtm),  levels = 3)
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    ax.set_title('Test Signal (Eqn 4.7) - Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    plt.show()'''
    return None #modify as needed

def analyze(fname='data4.wav',inputs=()):
    """Analyze signal contained in fname
    Input:
        fname: filename containing signal to be analyzed
        inputs: can be used to provide additional needed information
    Output: add as needed
    """
    #Load in music excerpt
    fname='data4.wav'
    fs, y = wavfile.read(fname)
    print("sampling frequency=",fs)
    N = y.size
    dur = N/fs
    dt = 1/fs
    t = np.arange(0,dur,dt)
    
    #Plot the excerpt against time    
    plt.figure()
    plt.title("Music Excerpt")
    plt.ylabel('Signal')
    plt.xlabel('time')
    plt.grid()
    plt.plot(t, y, '-', linewidth = .5);
    
    #Scalogram for entire excerpt
    w = 5
    freq = np.arange(0, 2**14, 50)
    widths = w*fs / (2*freq*np.pi)
    cwtm = sig.cwt(y, sig.morlet2, widths, w=w)
    
    fig, ax = plt.subplots()
    plt.semilogy(base = 2)
    plt.ylim( (2**6,2**14))
    
    cp = plt.contourf(t, freq , np.abs(cwtm), levels = 4)
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    ax.set_title('Music Excerpt - Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    positions = np.array([2**i for i in range(6, 15)])
    labels = ("$2^{6}$",
              "$2^{7}$",
              "$2^{8}$",
              "$2^{9}$", 
              "$2^{10}$",
              "$2^{11}$",
              "$2^{12}$",
              "$2^{13}$",
              "$2^{14}$")
    plt.yticks(positions, labels) 
    plt.show()
    
    #scalogram for first part of excerpt
    y1 = y[:np.argwhere(t==4.)[0][0]]
    t1 = t[:np.argwhere(t==4.)[0][0]]
    
    #w = 5
    freq = np.arange(0, 2**14, 50)
    widths = w*fs / (2*freq*np.pi)
    cwtm = sig.cwt(y1, sig.morlet2, widths, w=w)
    
    fig, ax = plt.subplots()
    plt.semilogy(base = 2)
    plt.ylim( (2**6,2**14))
    
    cp = plt.contourf(t1, freq , np.abs(cwtm), levels=4)
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    ax.set_title('Music Excerpt - Part 1 - Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    positions = np.array([2**i for i in range(6, 15)])
    labels = ("$2^{6}$",
              "$2^{7}$",
              "$2^{8}$",
              "$2^{9}$", 
              "$2^{10}$",
              "$2^{11}$",
              "$2^{12}$",
              "$2^{13}$",
              "$2^{14}$")
    plt.yticks(positions, labels) 
    plt.show()
    
    #scalogram for second part of excerpt
    y2 = y[np.argwhere(t==4.)[0][0]:np.argwhere(t==10.)[0][0]]
    t2 = t[np.argwhere(t==4.)[0][0]:np.argwhere(t==10.)[0][0]]
    
    #w = 5
    freq = np.arange(0, 2**14, 50)
    widths = w*fs / (2*freq*np.pi)
    cwtm = sig.cwt(y2, sig.morlet2, widths, w=w)
    
    fig, ax = plt.subplots()
    plt.semilogy(base = 2)
    plt.ylim( (2**6,2**14))
    
    cp = plt.contourf(t2, freq , np.abs(cwtm),levels=4)
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    ax.set_title('Music Excerpt-Part 2- Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    positions = np.array([2**i for i in range(6, 15)])
    labels = ("$2^{6}$",
              "$2^{7}$",
              "$2^{8}$",
              "$2^{9}$", 
              "$2^{10}$",
              "$2^{11}$",
              "$2^{12}$",
              "$2^{13}$",
              "$2^{14}$")
    plt.yticks(positions, labels)
    plt.show()
    
    
    #scalogram for third part of excerpt
    y3 = y[np.argwhere(t==10.)[0][0]:]
    t3 = t[np.argwhere(t==10.)[0][0]:]
    
    #w = 5
    freq = np.arange(0, 2**14, 50)
    widths = w*fs / (2*freq*np.pi)
    cwtm = sig.cwt(y3, sig.morlet2, widths, w=w)
    
    fig, ax = plt.subplots()
    plt.semilogy(base = 2)
    plt.ylim( (2**6,2**14))
    
    cp = plt.contourf(t3, freq , np.abs(cwtm), levels = 4)
    plt.grid()
    cbar = plt.colorbar(cp)
    cbar.set_ticks([])
    cbar.set_ticklabels([])
    ax.set_title('Music Excerpt - Part 3 - Scalogram')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    positions = np.array([2**i for i in range(6, 15)])
    labels = ("$2^{6}$",
              "$2^{7}$",
              "$2^{8}$",
              "$2^{9}$", 
              "$2^{10}$",
              "$2^{11}$",
              "$2^{12}$",
              "$2^{13}$",
              "$2^{14}$")
    plt.yticks(positions, labels) 
    plt.show()

    
    return None #modify as needed

if __name__=='__main__':
    #Add code below to call testSignal and analyze so that they generate the figures
    #in your report.
    testSignal()
    analyze() 
    
    
    
    
    
    
    
    
