import torch
from training.discriminators import MultiScaleSTFTDiscriminator
from training.discriminator_config import DiscriminatorConfig

discriminator_conf = DiscriminatorConfig()
discriminator_model = MultiScaleSTFTDiscriminator(config=discriminator_conf)
outputs = discriminator_model(torch.rand(32, 1, 48_000))

logits, fmaps = outputs

for logit in logits:
    print(logit.shape)

for fmap in fmaps:
    for fmp in fmap:
        print(fmp.shape)

# discriminator_model.save_pretrained("/home/tts/ttsteam/repos/Mimi/logs/first_checkpoint/discriminator")
discriminator_model.push_to_hub("AshwinSankar/Mimi-v1-multilingual-discriminator", private=True)