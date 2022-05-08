import torch
import torch.nn as nn
from .basic import BasicModule


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=True):
    layers = []
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=4, stride=2, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    flatten = nn.Flatten()
    layers += [flatten]
    return nn.Sequential(*layers)


class CVAE(BasicModule):
    def __init__(self, name, input_dim, hidden_dim1, hidden_dim2, z_dim):
        super(CVAE, self).__init__()
        self.name = name
        # encoder part
        self.fc1 = nn.Linear(input_dim + 64, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc31 = nn.Linear(hidden_dim2, z_dim)
        self.fc32 = nn.Linear(hidden_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim + 64, 96)
        self.fc5 = nn.Linear(96, 64)
        self.fc6 = nn.Linear(64, input_dim)

        self.relu = nn.ReLU()

        self.embed_feat = [16, 16, 32, 32, 16, 16]
        self.embed = make_layers(self.embed_feat, in_channels=1)

        initialize_weights(self)

    def encoder(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = self.relu(self.fc4(z))
        h = self.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def embed_fea(self, img):
        emb = self.embed(img)
        return emb

    def forward(self, x, img):
        emb = self.embed(img)
        i_z = torch.cat([x, emb], 1)
        mu, log_var = self.encoder(i_z)
        z = self.sampling(mu, log_var)
        o_z = torch.cat([z, emb], 1)
        return self.decoder(o_z), mu, log_var


def cvae(modul_name, model_name, **kwargs):
    name = modul_name + '_' + model_name
    return CVAE(name, **kwargs)
