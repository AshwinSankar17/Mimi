import numpy as np
import typing as tp
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

ADVERSARIAL_LOSSES = ['mse', 'hinge', 'hinge2']

AdvLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor], torch.Tensor]]
FeatLossType = tp.Union[nn.Module, tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]

def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)

@contextmanager
def readonly(model: torch.nn.Module):
    """Temporarily switches off gradient computation for the given model.
    """
    state = []
    for p in model.parameters():
        state.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, s in zip(model.parameters(), state):
            p.requires_grad_(s)

class AdversarialLoss(nn.Module):
    """Adversary training wrapper.

    Args:
        adversary (nn.Module): The adversary module will be used to estimate the logits given the fake and real samples.
            We assume here the adversary output is ``Tuple[List[torch.Tensor], List[List[torch.Tensor]]]``
            where the first item is a list of logits and the second item is a list of feature maps.
        loss (AdvLossType): Loss function for generator training.
        loss_real (AdvLossType): Loss function for adversarial training on logits from real samples.
        loss_fake (AdvLossType): Loss function for adversarial training on logits from fake samples.
        loss_feat (FeatLossType): Feature matching loss function for generator training.
        normalize (bool): Whether to normalize by number of sub-discriminators.

    Example of usage:
        adv_loss = AdversarialLoss(adversaries, loss, loss_real, loss_fake) 
        for real in loader:
            noise = torch.randn(...)
            fake = model(noise)
            adv_loss.train_adv(fake, real)
            loss, _ = adv_loss(fake, real)
            loss.backward()
    """
    def __init__(
        self,
        adversary: nn.Module,
        loss: AdvLossType,
        loss_real: AdvLossType,
        loss_fake: AdvLossType,
        loss_feat: tp.Optional[FeatLossType] = None,
        normalize: bool = True
    ):
        super().__init__()
        self.adversary: nn.Module = adversary
        self.loss = loss
        self.loss_real = loss_real
        self.loss_fake = loss_fake
        self.loss_feat = loss_feat
        self.normalize = normalize

    # def _save_to_state_dict(self, destination, prefix, keep_vars):
    #     # Add the optimizer state dict inside our own.
    #     super()._save_to_state_dict(destination, prefix, keep_vars)
    #     destination[prefix + 'optimizer'] = self.optimizer.state_dict()
    #     return destination

    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     # Load optimizer state.
    #     self.optimizer.load_state_dict(state_dict.pop(prefix + 'optimizer'))
    #     super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_adversary_pred(self, x):
        """Run adversary model, validating expected output format."""
        logits, fmaps = self.adversary(x)
        assert isinstance(logits, list) and all([isinstance(t, torch.Tensor) for t in logits]), \
            f'Expecting a list of tensors as logits but {type(logits)} found.'
        assert isinstance(fmaps, list), f'Expecting a list of features maps but {type(fmaps)} found.'
        for fmap in fmaps:
            assert isinstance(fmap, list) and all([isinstance(f, torch.Tensor) for f in fmap]), \
                f'Expecting a list of tensors as feature maps but {type(fmap)} found.'
        return logits, fmaps

    def discriminator_fwd(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Train the adversary with the given fake and real example.

        We assume the adversary output is the following format: Tuple[List[torch.Tensor], List[List[torch.Tensor]]].
        The first item being the logits and second item being a list of feature maps for each sub-discriminator.

        This will automatically synchronize gradients (with `flashy.distrib.eager_sync_model`)
        and call the optimizer.
        """
        loss = torch.tensor(0., device=fake.device)
        all_logits_fake_is_fake, _ = self.get_adversary_pred(fake.detach())
        all_logits_real_is_fake, _ = self.get_adversary_pred(real.detach())
        n_sub_adversaries = len(all_logits_fake_is_fake)
        for logit_fake_is_fake, logit_real_is_fake in zip(all_logits_fake_is_fake, all_logits_real_is_fake):
            loss += self.loss_fake(logit_fake_is_fake) + self.loss_real(logit_real_is_fake)

        if self.normalize:
            loss /= n_sub_adversaries

        return loss

    def generator_fwd(self, fake: torch.Tensor, real: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Return the loss for the generator, i.e. trying to fool the adversary,
        and feature matching loss if provided.
        """
        with self.adversary.no_sync():
            adv = torch.tensor(0., device=fake.device)
            feat = torch.tensor(0., device=fake.device)
            with readonly(self.adversary):
                all_logits_fake_is_fake, all_fmap_fake = self.get_adversary_pred(fake)
                all_logits_real_is_fake, all_fmap_real = self.get_adversary_pred(real)
                n_sub_adversaries = len(all_logits_fake_is_fake)
                for logit_fake_is_fake in all_logits_fake_is_fake:
                    adv += self.loss(logit_fake_is_fake)
                if self.loss_feat:
                    for fmap_fake, fmap_real in zip(all_fmap_fake, all_fmap_real):
                        feat += self.loss_feat(fmap_fake, fmap_real)

        if self.normalize:
            adv /= n_sub_adversaries
            feat /= n_sub_adversaries

        return adv, feat


def get_adv_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'hinge':
        return hinge_loss
    elif loss_type == 'hinge2':
        return hinge2_loss
    raise ValueError('Unsupported loss')


def get_fake_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_fake_loss
    elif loss_type in ['hinge', 'hinge2']:
        return hinge_fake_loss
    raise ValueError('Unsupported loss')


def get_real_criterion(loss_type: str) -> tp.Callable:
    assert loss_type in ADVERSARIAL_LOSSES
    if loss_type == 'mse':
        return mse_real_loss
    elif loss_type in ['hinge', 'hinge2']:
        return hinge_real_loss
    raise ValueError('Unsupported loss')


def mse_real_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(1., device=x.device).expand_as(x))


def mse_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, torch.tensor(0., device=x.device).expand_as(x))


def hinge_real_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))


def hinge_fake_loss(x: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.min(-x - 1, torch.tensor(0., device=x.device).expand_as(x)))


def mse_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return F.mse_loss(x, torch.tensor(1., device=x.device).expand_as(x))


def hinge_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0], device=x.device)
    return -x.mean()


def hinge2_loss(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor([0.0])
    return -torch.mean(torch.min(x - 1, torch.tensor(0., device=x.device).expand_as(x)))


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for adversarial training.

    Args:
        loss (nn.Module): Loss to use for feature matching (default=torch.nn.L1).
        normalize (bool): Whether to normalize the loss.
            by number of feature maps.
    """
    def __init__(self, loss: nn.Module = torch.nn.L1Loss(), normalize: bool = True):
        super().__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, fmap_fake: tp.List[torch.Tensor], fmap_real: tp.List[torch.Tensor]) -> torch.Tensor:
        assert len(fmap_fake) == len(fmap_real) and len(fmap_fake) > 0
        feat_loss = torch.tensor(0., device=fmap_fake[0].device)
        feat_scale = torch.tensor(0., device=fmap_fake[0].device)
        n_fmaps = 0
        for (feat_fake, feat_real) in zip(fmap_fake, fmap_real):
            assert feat_fake.shape == feat_real.shape
            n_fmaps += 1
            feat_loss += self.loss(feat_fake, feat_real)
            feat_scale += torch.mean(torch.abs(feat_real))

        if self.normalize:
            feat_loss /= n_fmaps

        return feat_loss

class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.

    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(torch.nn.Module):
    """L1 Loss on MelSpectrogram.

    Args:
        sample_rate (int): Sample rate.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                             log=log, normalized=normalized, floor_level=floor_level)

    def forward(self, x, y):
        self.melspec.to(x.device)
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return self.l1(s_x, s_y)


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).

    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 64, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 normalized: bool = False, alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, normalized=normalized, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, normalized=normalized, floor_level=floor_level))
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss