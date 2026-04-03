import librosa
import numpy as np

def extract_mfcc(audio_path, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        pre_emphasis = 0.97
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
            hop_length=hop_length, fmax=sr // 2
        )
        return mfcc
    except Exception as e:
        print(f"提取MFCC失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mfcc, 100), dtype=np.float32)

def extract_temporal_features(audio_path, sr=16000, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        return rmse
    except Exception as e:
        print(f"提取时域特征失败: {audio_path}, 错误: {e}")
        return np.zeros(100, dtype=np.float32)

def extract_energy_features(audio_path, sr=16000, n_mels=40, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length,
            fmax=sr // 2
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    except Exception as e:
        print(f"提取能量特征失败: {audio_path}, 错误: {e}")
        return np.zeros((n_mels, 100), dtype=np.float32)
