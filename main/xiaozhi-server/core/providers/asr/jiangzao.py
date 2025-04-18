import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def denoise_with_noise_profile(target_wav_path, noise_wav_path, output_wav_path):
    """
    使用专门的噪声音频文件进行降噪
    
    参数:
    target_wav_path: 需要降噪的目标音频文件路径
    noise_wav_path: 包含噪声特征的音频文件路径
    output_wav_path: 输出降噪后音频的路径
    """
    # 读取目标音频文件
    target_rate, target_audio = wav.read(target_wav_path)
    
    # 读取噪声音频文件
    noise_rate, noise_audio = wav.read(noise_wav_path)
    
    # 确保噪声音频与目标音频采样率相同
    if target_rate != noise_rate:
        raise ValueError("噪声音频和目标音频的采样率必须相同")
    
    # 参数设置
    frame_size = 512
    hop_size = frame_size // 2
    window = np.hamming(frame_size)
    
    # 分帧函数
    def framing(signal, frame_size, hop_size):
        num_frames = 1 + (len(signal) - frame_size) // hop_size
        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            frames[i] = signal[i * hop_size:i * hop_size + frame_size]
        return frames
    
    # 重建信号
    def overlap_add(frames, hop_size):
        num_frames, frame_size = frames.shape
        output_length = (num_frames - 1) * hop_size + frame_size
        output_signal = np.zeros(output_length)
        for i in range(num_frames):
            output_signal[i * hop_size:i * hop_size + frame_size] += frames[i]
        return output_signal
    
    # 处理噪声音频以获取噪声特征
    noise_frames = framing(noise_audio, frame_size, hop_size)
    windowed_noise_frames = noise_frames * window
    fft_noise_frames = np.fft.rfft(windowed_noise_frames)
    noise_estimate = np.mean(np.abs(fft_noise_frames), axis=0)
    
    # 处理目标音频
    target_frames = framing(target_audio, frame_size, hop_size)
    windowed_target_frames = target_frames * window
    fft_target_frames = np.fft.rfft(windowed_target_frames)
    
    # 频谱减法（加入门限）
    magnitude = np.abs(fft_target_frames)
    phase = np.angle(fft_target_frames)
    clean_magnitude = magnitude - noise_estimate * 1.2  # 加入噪声放大系数
    clean_magnitude = np.maximum(clean_magnitude, 0.01)  # 设置门限，避免过度减法
    
    # 重建频谱
    clean_fft = clean_magnitude * np.exp(1j * phase)
    
    # IFFT
    clean_frames = np.fft.irfft(clean_fft)
    
    # 重叠相加
    clean_audio = overlap_add(clean_frames, hop_size)
    
    # 归一化
    clean_audio = np.int16(clean_audio / np.max(np.abs(clean_audio)) * 32767)
    
    # 保存结果
    wav.write(output_wav_path, target_rate, clean_audio)

# # 示例调用
# target_wav = "8.wav"          # 需要降噪的音频
# noise_wav = "8.wav"       # 纯噪声音频
# output_wav = "88.wav"

# # 调用函数进行降噪
# denoise_with_noise_profile(target_wav, noise_wav, output_wav)