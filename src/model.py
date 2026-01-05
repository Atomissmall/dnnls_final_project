import torch
import torch.nn as nn
import torch.nn.functional as F

# Text autoencoder (LSTM)
class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        outputs, (h, c) = self.lstm(x)
        return outputs, h, c


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, h, c):
        x = self.embedding(input_ids)
        outputs, (h, c) = self.lstm(x, (h, c))
        logits = self.fc(outputs)
        return logits, h, c


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_input_ids, tgt_input_ids):
        _, h, c = self.encoder(src_input_ids)
        logits, _, _ = self.decoder(tgt_input_ids, h, c)
        return logits

# Visual autoencoder (CNN)
class EncoderCNN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 4 * 8, latent_dim)  # assumes input 60x125

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        z = self.fc(x)
        return z


class DecoderCNN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 8)
        self.unflatten = nn.Unflatten(1, (128, 4, 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = EncoderCNN(latent_dim)
        self.decoder = DecoderCNN(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
# Sequence predictor (Attention + GRU)
class Attention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        return out


class SequencePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.attn = Attention(latent_dim)
        self.gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.fc = nn.Linear(latent_dim, latent_dim)

    def forward(self, seq_latents):
        x = self.attn(seq_latents)
        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred
