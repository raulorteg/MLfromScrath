import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from filtering.wiener import WienerFilterFIR

def main():
    
    # load the data
    signal_ = scipy.io.loadmat('data/wiener_test/signal.mat')['signal'][:,0]
    noise = scipy.io.loadmat('data/wiener_test/noise.mat')['noise'][:,0]
    x = signal_ + noise

    # plot the whole sequence
    plt.subplot(3,1,1)
    plt.plot(signal_, label="Original signal", color="blue")
    plt.title("Original signal")
    plt.subplot(3,1,2)
    plt.plot(noise, label="Noise", color="orange")
    plt.title("Noise")
    plt.subplot(3,1,3)
    plt.plot(x, label="Noisy signal", color="green")
    plt.title("Noisy signal")
    plt.tight_layout()
    plt.show()

    # plot a subset of the sequence (zoom in)
    plt.subplot(3,1,1)
    plt.plot(signal_[100:300], label="Original signal", color="blue")
    plt.title("Original signal")
    plt.subplot(3,1,2)
    plt.plot(noise[100:300], label="Noise", color="orange")
    plt.title("Noise")
    plt.subplot(3,1,3)
    plt.plot(x[100:300], label="Noisy signal", color="green")
    plt.title("Noisy signal")
    plt.tight_layout()
    plt.show()

    FIR = WienerFilterFIR(length=10)
    length_vec = [1,3,5,7,10,15,20,25,30,35,40,45,50,60,80,100]
    mse_vec = []
    for length in length_vec:
        FIR.fit(x=x, y=signal_, length=length)
        _, mse = FIR.transform(x)
        mse_vec.append(mse)
    plt.plot(length_vec, mse_vec, color="orange", linestyle='dashed', marker='x')
    plt.xlabel("Length of filter")
    plt.ylabel("MSE")
    plt.show()
    
    FIR = WienerFilterFIR(length=15)
    FIR.fit(x=x, y=signal_,)
    filtered, _ = FIR.transform(x)

    # plot a subset of the sequence (zoom in)
    plt.subplot(2,1,1)
    plt.plot(signal_[100:300], label="Original signal", color="blue", alpha=0.5)
    plt.plot(filtered[100:300], label="Denoised signal", color="green", alpha=0.5)
    plt.legend()
    plt.title("Original signal")
    plt.subplot(2,1,2)
    plt.plot(x[100:300], label="Noisy signal", color="orange")
    plt.plot(filtered[100:300], label="Denoised signal", color="green")
    plt.legend()
    plt.title("Noisy signal")
    plt.tight_layout()
    plt.show()

    norm_freq, freq_response = signal.freqz(FIR.w)
    fig = plt.figure()
    plt.title('filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.plot(norm_freq, 20 * np.log10(abs(freq_response)), 'b')
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [rad/sample]')
    plt.grid(axis='both', linestyle='-')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(freq_response))
    plt.plot(norm_freq, angles, 'g')
    plt.ylabel('Angle (radians)')
    plt.grid(axis='both', linestyle='-')
    plt.axis('tight')
    plt.show()

    freq_response = np.fft.fft(signal_)
    norm_freq = np.fft.fftfreq(signal_.shape[0])
    plt.subplot(3,1,1)
    plt.title('Original Signal spectrum')
    plt.plot(norm_freq[0:16000], abs(freq_response[0:16000]), 'b')
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Normalized frequency')
    plt.grid(axis='both', linestyle='-')

    freq_response = np.fft.fft(noise)
    norm_freq = np.fft.fftfreq(noise.shape[0])
    plt.subplot(3,1,2)
    plt.title('Noise spectrum')
    plt.plot(norm_freq[0:16000], abs(freq_response[0:16000]), 'orange')
    plt.ylim((0,500))
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Normalized frequency')
    plt.grid(axis='both', linestyle='-')
    
    plt.subplot(3,1,3)
    plt.title('Noisy signal and denoised signal spectrum')
    freq_response = np.fft.fft(x)
    norm_freq = np.fft.fftfreq(x.shape[0])
    plt.plot(norm_freq[0:16000], abs(freq_response[0:16000]), color='blue', label="Noisy signal", alpha=0.5)
    freq_response = np.fft.fft(filtered)
    norm_freq = np.fft.fftfreq(filtered.shape[0])
    plt.plot(norm_freq[0:16000], abs(freq_response[0:16000]), color='green', label="Denoised signal", alpha=0.5)
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Normalized frequency')
    plt.legend()
    plt.grid(axis='both', linestyle='-')
    plt.tight_layout()
    plt.show()
    

    # visualize

if __name__ == "__main__":
    main()