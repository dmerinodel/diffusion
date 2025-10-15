"""
Prueba con el modelo UNet como Denoising Probabilistic Function
"""
import torch
import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs).__init__()

        conv1 = [nn.Conv2d(1, 2, 5, 2), nn.ReLU()]  # 1x128x42 -> 2x62x19
        conv2 = [nn.Conv2d(2, 4, 5, 2), nn.ReLU()]  # 2x62x19 -> 4x29x8
        conv3 = [nn.Conv2d(4, 4, 5, 2), nn.ReLU()]  # 4x29x8 -> 4x13x2
        enc_convs = conv1 + conv2 + conv3
        self.encoder = nn.Sequential(*enc_convs)  # latent_dim = 4*13*2 = 102
        # Cuidado en esta parte con la forma de los datos, nn.Upsample es especialito
        upsamp1 = [nn.Upsample((29, 8)), nn.Conv2d(4, 4,
                                                   5, padding='same'), nn.MaxPool2d(3)]
        upsamp2 = [nn.Upsample((62, 19)), nn.Conv2d(4, 2,
                                                    5, padding='same'), nn.MaxPool2d(3)]
        upsamp3 = [nn.Upsample((128, 42)), nn.Conv2d(2, 1,
                                                     5, padding='same'), nn.ReLU()]
        dec_ups = upsamp1+upsamp2+upsamp3
        self.decoder = nn.Sequential(*dec_ups)

    def forward(self, imgs):
        latent_vec = self.encoder(imgs)
        output = self.decoder(latent_vec)

        return output


batch = torch.randn(4, 1, 128, 42)
model = SimpleAutoEncoder()
recon = model(batch)
print(recon.shape)

