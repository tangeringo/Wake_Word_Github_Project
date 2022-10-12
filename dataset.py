from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch


dataset_path, train_data, test_data = 'C:\\Users\\matos\\Desktop\\wake_word', 'Train_Data', 'Test_Data'
SAMPLE_RATE, NUM_SAMPLES = 22050, 22050
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Take each audio sample and make it the same length
class WakeWordDataset(Dataset):
    """Load and return the data"""
    def __init__(self, data_json, transformation, target_sample_rate, num_samples, device):
        self.data = pd.read_json(data_json)
        self.data_category = 'Train_Data' if data_json == 'DATA/train_data.json' else 'Test_Data'
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.data_category == 'Train_Data':
            audio_obj = self.data.Train_Data.loc[index]
            loaded_path = audio_obj['path']
            label = audio_obj['label']
            waveform, sr = torchaudio.load(loaded_path)
            waveform = self._return_waveform(waveform, sr)

            return waveform, label
        elif self.data_category == 'Test_Data':
            audio_obj = self.data.Test_Data.loc[index]
            loaded_path = audio_obj['path']
            label = audio_obj['label']
            waveform, sr = torchaudio.load(loaded_path)
            waveform = self._return_waveform(waveform, sr)

            return waveform, label

    def _return_waveform(self, waveform, sr):
        waveform = waveform.to(device)
        waveform = self._resample_if_necessary(waveform, sr)
        waveform = self._mix_down_if_necessary(waveform)
        waveform = self._cut_if_necessary(waveform)
        waveform = self._right_pad_if_necessary(waveform)
        waveform = self.transformation(waveform)
        return waveform

    def _resample_if_necessary(self, waveform, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform

    def _mix_down_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _cut_if_necessary(self, waveform):
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        return waveform

    def _right_pad_if_necessary(self, waveform):
        waveform_length = waveform.shape[1]
        if waveform_length < self.num_samples:
            num_missing_samples = self.num_samples - waveform_length
            last_dim_padding = (0, num_missing_samples)
            waveform = torch.nn.functional.pad(waveform, last_dim_padding)
        return waveform


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
