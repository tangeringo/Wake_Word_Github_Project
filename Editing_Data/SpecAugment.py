import torch
import torchaudio.transforms


class SpecAugment(torch.nn.Module):
    """Implement more data in the Time and Frequency domain"""

    def __init__(self, condition, time_mask=4, freq_mask=2, container_param=3):
        super(SpecAugment, self).__init__()

        self.condition = condition
        self.spec_augment_1 = torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask)
        )
        self.spec_augment_2 = torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=time_mask),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
        )

        container_dict = {1: self.container_func_1, 2: self.container_func_2, 3: self.container_func_3}
        self._forward = container_dict[container_param]

    def forward(self, x):
        return self._forward

    def container_func_1(self, x):
        probability = torch.rand(1).item()
        if self.condition > probability:
            return self.spec_augment_1(x)
        return x

    def container_func_2(self, x):
        probability = torch.rand(1).item()
        if self.condition < probability:
            return self.spec_augment_2(x)
        return x

    def container_func_3(self, x):
        probability = torch.rand(1).item()
        if probability > 0.5:
            return self.spec_augment_1(x)
        return self.spec_augment_2(x)
