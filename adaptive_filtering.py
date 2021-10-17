import numpy as np
from scipy import signal
import scipy.io
import librosa
import librosa.display
import matplotlib.pyplot as plt
from filtering.adaptive import AdaptiveFilter

def main():
    # load the data
    s = scipy.io.loadmat('data/adaptive_test/speech.mat')['speech']
    lpir = scipy.io.loadmat('data/adaptive_test/lpir.mat')['lpir']
    hpir = scipy.io.loadmat('data/adaptive_test/hpir.mat')['hpir']
    bpir = scipy.io.loadmat('data/adaptive_test/bpir.mat')['bpir']

    noise = np.random.normal(loc=0, scale=0.5, size = len(s))
    d = signal.lfilter(lpir[:,0], [1], noise)
    s = s[:,0]
    x=s+d

    plt.subplot(3,1,1)
    plt.plot(s, label="Original signal", color="blue")
    plt.title("Original signal")
    plt.subplot(3,1,2)
    plt.plot(noise, label="Noise", color="orange")
    plt.title("Noise")
    plt.subplot(3,1,3)
    plt.plot(x, label="Noisy signal", color="green")
    plt.title("Noisy signal")
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.show()

    adpativefilter = AdaptiveFilter(method="nlms")
    length_vec = [5,7,10,15,20,25,30,35,40,45,50,60,80,100,200]
    for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mse_vec = []
        for length in length_vec:
            yhat, mse = adpativefilter.fit_transform(x=x, y=s, L=length, mu=mu, delta=0.01)
            mse_vec.append(mse.mean())
        plt.plot(length_vec, mse_vec, linestyle='dashed', marker='x', label=f"mu: {mu}")
    plt.legend()

    plt.xlabel("Length of filter")
    plt.ylabel("MSE")
    plt.show()

    yhat, _ = adpativefilter.fit_transform(x=x, y=s, L=80, mu=0.9, delta=0.01)
    plt.subplot(2,1,1)
    plt.plot(s, label="Original signal", color="blue", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="orange", alpha=0.5)
    plt.legend()
    plt.title("Original signal")
    plt.title("Noise")
    plt.subplot(2,1,2)
    plt.plot(x, label="Noisy signal", color="green", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="blue", alpha=0.5)
    plt.legend()
    plt.title("Noisy signal and denoised signal")
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.show()


    plt.subplot(1,3,1)
    spec_y = librosa.stft(s, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Original signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,2)
    spec_y = librosa.stft(x, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Noisy signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,3)
    spec_y = librosa.stft(yhat, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("nlms method. Denoised signal")
    librosa.display.specshow(y_db, sr=7800)
    plt.show()
    

    ######################### rls method #####################################
    beta = 0.997 # forget factor
    adpativefilter = AdaptiveFilter(method="rls")
    length_vec = [5,10,20,30,35,40,45,50,60,80,100,200]
    for lambda_ in [0.1, 0.2, 0.5, 0.7, 1.0]:
        mse_vec = []
        for length in length_vec:
            yhat, mse = adpativefilter.fit_transform(x=x, y=s, L=length, beta=beta, lambda_=lambda_, method="rls")
            mse_vec.append(mse.mean())
        plt.plot(length_vec, mse_vec, linestyle='dashed', marker='x', label=f"lambda_: {lambda_}")
    plt.legend()
    plt.xlabel("Length of filter")
    plt.ylabel("MSE")
    plt.show()
    
    adpativefilter = AdaptiveFilter(method="rls")
    yhat, _ = adpativefilter.fit_transform(x=x, y=s, L=80, beta=0.997, lambda_=0.7, method="rls")
    plt.subplot(2,1,1)
    plt.plot(s, label="Original signal", color="blue", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="orange", alpha=0.5)
    plt.legend()
    plt.title("Original signal")
    plt.title("Noise")
    plt.subplot(2,1,2)
    plt.plot(x, label="Noisy signal", color="green", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="blue", alpha=0.5)
    plt.legend()
    plt.title("Noisy signal and denoised signal")
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.tight_layout()
    plt.show()


    plt.subplot(1,3,1)
    spec_y = librosa.stft(s, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Original signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,2)
    spec_y = librosa.stft(x, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Noisy signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,3)
    spec_y = librosa.stft(yhat, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("rls method. Denoised signal")
    librosa.display.specshow(y_db, sr=7800)
    plt.show()

    ############## method mls #######################

    beta = 0.997 # forget factor
    adpativefilter = AdaptiveFilter(method="lms")
    length_vec = [10,30,40,50,60,80,100,200,300,400, 1000]
    for mu in [0.001, 0.003, 0.005, 0.007, 0.009, 0.02]:
        mse_vec = []
        for length in length_vec:
            yhat, mse = adpativefilter.fit_transform(x=x, y=s, L=length, mu=mu, delta=0.001, method="lms")
            mse_vec.append(mse.mean())
        plt.plot(length_vec, mse_vec, linestyle='dashed', marker='x', label=f"mu: {mu}")
    plt.legend()
    plt.xlabel("Length of filter")
    plt.ylabel("MSE")
    plt.show()

    adpativefilter = AdaptiveFilter(method="lms")
    yhat, _ = adpativefilter.fit_transform(x=x, y=s, L=200, mu=0.02, delta=0.01, method="lms")
    
    plt.subplot(2,1,1)
    plt.plot(s, label="Original signal", color="blue", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="orange", alpha=0.5)
    plt.legend()
    plt.title("Original signal")
    plt.title("Noise")
    plt.subplot(2,1,2)
    plt.plot(x, label="Noisy signal", color="green", alpha=0.5)
    plt.plot(yhat, label="Denoised signal", color="blue", alpha=0.5)
    plt.legend()
    plt.title("Noisy signal and denoised signal")
    plt.xlabel('index')
    plt.ylabel('amplitude')
    plt.show()


    plt.subplot(1,3,1)
    spec_y = librosa.stft(s, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Original signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,2)
    spec_y = librosa.stft(x, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("Noisy signal")
    librosa.display.specshow(y_db, sr=7800)

    plt.subplot(1,3,3)
    spec_y = librosa.stft(yhat, n_fft=512, hop_length=32, center=True)
    y_db = librosa.amplitude_to_db(abs(spec_y))
    plt.title("lms method. Denoised signal")
    librosa.display.specshow(y_db, sr=7800)
    plt.show()


if __name__ == "__main__":
    main()