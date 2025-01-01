import typing as tp
import torch
import torch.nn as nn
import nemo.collections.asr as nemo_asr
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

from einops import rearrange

def mask_from_lens(lens, max_len: tp.Optional[int] = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask

class W2V2BertFeature(nn.Module):
    """
    A module to extract features using a Wav2Vec2-BERT model.

    Attributes:
        processor: An instance of AutoFeatureExtractor for pre-processing audio data.
        model: A Wav2Vec2BERT model for generating semantic embeddings.
    """
    def __init__(self, config):
        """
        Initializes the W2V2BertFeature module.

        Args:
            config: Configuration object containing model-related parameters,
                    including the `speech_encoder_model_name_or_path` (path or name of the pretrained model).
        """
        super(W2V2BertFeature, self).__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(config.speech_encoder_model_name_or_path)
        self.model = Wav2Vec2BertModel.from_pretrained(config.speech_encoder_model_name_or_path)
        self.freeze()

    def freeze(self):
        """
        Freezes the parameters of the Wav2Vec2-BERT model to prevent updates during training.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids):
        """
        Computes semantic embeddings for the input audio batch.

        Args:
            input_ids (torch.Tensor): Input audio tensor of shape (B, 1, T) or (B, T),
                                        where B is the batch size, T is the sequence length.

        Returns:
            tuple: A tuple containing:
                - last_hidden_state (torch.Tensor): The semantic embeddings of shape (B, ssl_dim, T).
                - attention_mask (torch.Tensor): Attention mask for the sequences last_hidden_state. Same as attention_mask from preprocessor.
        """
        if input_ids.ndim == 3:
            input_ids = rearrange(input_ids, 'b 1 t -> b t')
        assert input_ids.ndim == 2, "Expected input shape (B, T)."

        audios_list = [audio[audio != 0].cpu().numpy() for audio in input_ids]
        
        inputs = self.processor(
            audios_list, 
            sampling_rate=self.processor.sampling_rate, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        outputs = self.model(**inputs)

        last_hidden_state = outputs['last_hidden_state'].to(torch.float32)  # (B, ssl_dim, T)

        return last_hidden_state, inputs.attention_mask

class IndicConformerFeature(nn.Module):
    """
    A module to extract features using the Indic Conformer ASR model.

    Attributes:
        model: A NeMo ASR model for generating semantic embeddings.
    """
    def __init__(self, config):
        """
        Initializes the IndicConformerFeature module.

        Args:
            config: Configuration object containing model-related parameters,
                    including the `speech_encoder_model_name_or_path` (path or name of the pretrained model).
        """
        super(IndicConformerFeature, self).__init__()
        self.model = nemo_asr.models.ASRModel.from_pretrained(config.speech_encoder_model_name_or_path)
        self.freeze()

    def freeze(self):
        """
        Freezes the parameters of the Indic Conformer model to prevent updates during training.
        """
        self.model.freeze()

    @torch.no_grad()
    def forward(self, input_ids):
        """
        Computes semantic embeddings for the input audio batch.

        Args:
            input_ids (torch.Tensor): Input audio tensor of shape (B, 1, T) or (B, T),
                                        where B is the batch size, T is the sequence length.

        Returns:
            tuple: A tuple containing:
                - last_hidden_state (torch.Tensor): The semantic embeddings of shape (B, ssl_dim, T).
                - attention_mask (torch.Tensor): Attention mask created from the output lengths of the ssl embeddings.
        """
        if input_ids.ndim == 3:
            input_ids = rearrange(input_ids, 'b 1 t -> b t')
        assert input_ids.ndim == 2, "Expected input shape (B, T)."
        audio_lengths = (input_ids != 0).sum(dim=1).to(input_ids.device)
        
        *_, last_hidden_state, last_hidden_state_lens = model(input_signal=input_ids, input_signal_length=audio_lengths)
        last_hidden_state = last_hidden_state.to(torch.float32)
        
        return last_hidden_state, mask_from_lens(last_hidden_state_lens)
