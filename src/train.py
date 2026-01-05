import os
import torch
import torch.nn as nn

from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer

from src.models import (
    EncoderLSTM, DecoderLSTM, Seq2SeqLSTM,
    VisualAutoencoder,
    SequencePredictor
)

from src.utils import (
    parse_gdi_text,
    save_checkpoint, load_checkpoint,
    validation_panel,
    edge_loss,
    make_transforms
)

import torchvision.transforms.functional as TF


#DATASETS
class SequencePredictionDataset(torch.utils.data.Dataset):
    """
    Returns:
      sequence_tensor:    (4, C, H, W)
      description_tensor: (4, T)
      image_target:       (C, H, W)
      target_ids:         (1, T)
    """
    def __init__(self, hf_dataset, tokenizer, train_transform, eval_transform, mode="train", max_len=120):
        super().__init__()
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def _transform(self, img):
        if self.mode == "train":
            return self.train_transform(img)
        return self.eval_transform(img)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        frames = item["images"]
        attrs = parse_gdi_text(item["story"])

        frame_tensors = []
        desc_tensors = []

        for i in range(4):
            img = TF.equalize(frames[i])
            frame_tensors.append(self._transform(img))

            desc = attrs[i]["description"]
            ids = self.tokenizer(
                desc, return_tensors="pt",
                padding="max_length", truncation=True, max_length=self.max_len
            ).input_ids.squeeze(0)
            desc_tensors.append(ids)

        target_img = TF.equalize(frames[4])
        target_img = self._transform(target_img)

        target_desc = attrs[4]["description"]
        target_ids = self.tokenizer(
            target_desc, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_len
        ).input_ids  # (1,T)

        return (
            torch.stack(frame_tensors),          # (4,C,H,W)
            torch.stack(desc_tensors),           # (4,T)
            target_img,                          # (C,H,W)
            target_ids                            # (1,T)
        )


class TextTaskDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        attrs = parse_gdi_text(self.dataset[idx]["story"])
        frame_idx = torch.randint(0, min(5, len(attrs)), (1,)).item()
        return attrs[frame_idx]["description"]


class AutoEncoderTaskDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        num_frames = len(frames)
        frame_idx = torch.randint(0, max(1, num_frames), (1,)).item()
        return self.transform(frames[frame_idx])

def run(
    ckpt_dir="/content/gdrive/MyDrive/DL_Checkpoints",
    image_hw=(60, 125),
    max_len=120,
    use_augmentation=True,
    freeze_text_encoder=True,
    text_emb_dim=16,
    text_hidden_dim=16,
    visual_latent_dim=64,
    gru_hidden_dim=64,
    batch_train=8,
    batch_val=4,
    lr=1e-3,
    epochs_ae=10,
    epochs_seq=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Using device:", device)

    #dataset
    train_hf = load_dataset("daniel3303/StoryReasoning", split="train")
    test_hf  = load_dataset("daniel3303/StoryReasoning", split="test")

    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    train_tf, eval_tf = make_transforms(image_hw=image_hw, use_aug=use_augmentation)

    #split
    full_train = SequencePredictionDataset(train_hf, tokenizer, train_tf, eval_tf, mode="train", max_len=max_len)

    n = len(full_train)
    perm = torch.randperm(n).tolist()
    n_train = int(0.80 * n)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = Subset(SequencePredictionDataset(train_hf, tokenizer, train_tf, eval_tf, mode="train", max_len=max_len), train_idx)
    val_ds   = Subset(SequencePredictionDataset(train_hf, tokenizer, train_tf, eval_tf, mode="val",   max_len=max_len), val_idx)
    test_ds  = SequencePredictionDataset(test_hf, tokenizer, train_tf, eval_tf, mode="test", max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_val, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_val, shuffle=False)

    #text AE
    encoder = EncoderLSTM(tokenizer.vocab_size, text_emb_dim, text_hidden_dim, num_layers=1, dropout=0.1).to(device)
    decoder = DecoderLSTM(tokenizer.vocab_size, text_emb_dim, text_hidden_dim, num_layers=1, dropout=0.1).to(device)
    text_ae = Seq2SeqLSTM(encoder, decoder).to(device)

    text_ckpt = os.path.join(ckpt_dir, "text_autoencoder.pth")
    if os.path.exists(text_ckpt):
        text_ae, _, _, _ = load_checkpoint(text_ae, None, text_ckpt)
        print("Loaded text autoencoder checkpoint.")
    else:
        print("No text autoencoder ckpt found. (You can train it separately if needed.)")

    if freeze_text_encoder:
        for p in text_ae.encoder.parameters():
            p.requires_grad = False
        # decoder can remain trainable (helps generation)
        for p in text_ae.decoder.parameters():
            p.requires_grad = True

    #visual AE
    visual_ae = VisualAutoencoder(latent_dim=visual_latent_dim).to(device)

    visual_ckpt = os.path.join(ckpt_dir, "visual_autoencoder_pretrained.pth")
    if os.path.exists(visual_ckpt):
        visual_ae, _, _, _ = load_checkpoint(visual_ae, None, visual_ckpt)
        print("Loaded visual AE checkpoint.")
    else:
        # Pretrain visual AE quickly (from notebook intent)
        print("Pretraining visual AE...")
        ae_dataset = AutoEncoderTaskDataset(train_hf, transform=eval_tf)
        ae_loader = DataLoader(ae_dataset, batch_size=32, shuffle=True)

        opt_ae = torch.optim.Adam(visual_ae.parameters(), lr=lr)
        crit_ae = nn.L1Loss()

        visual_ae.train()
        for ep in range(epochs_ae):
            running = 0.0
            for imgs in ae_loader:
                imgs = imgs.to(device)
                opt_ae.zero_grad()
                recon = visual_ae(imgs)               # recon: (B,C,H,W)
                loss = crit_ae(recon, imgs)
                loss.backward()
                opt_ae.step()
                running += loss.item() * imgs.size(0)

            ep_loss = running / len(ae_loader.dataset)
            print(f"[AE] epoch {ep+1}/{epochs_ae} loss={ep_loss:.4f}")

        save_checkpoint(visual_ae, opt_ae, epochs_ae, ep_loss, visual_ckpt)

    # -------- sequence predictor --------
    model = SequencePredictor(
        visual_autoencoder=visual_ae,
        text_autoencoder=text_ae,
        visual_latent_dim=visual_latent_dim,
        text_hidden_dim=text_hidden_dim,
        gru_hidden_dim=gru_hidden_dim
    ).to(device)

    # Losses
    PAD_ID = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    crit_img = nn.L1Loss()
    crit_txt = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # Optimizer includes only trainable params
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    #training loop (sequence)
    print("Training sequence predictor...")
    train_losses = []
    val_losses = []

    for epoch in range(epochs_seq):
        model.train()
        running = 0.0

        for frames, descriptions, image_target, text_target in train_loader:
            frames = frames.to(device)
            descriptions = descriptions.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            pred_image, pred_text_logits, h0, c0 = model(frames, descriptions, text_target)
            # pred_text_logits

            loss_im = crit_img(pred_image, image_target)

            # text alignment
            targets = text_target[:, 0, 1:]  # (B, T-1)
            logits  = pred_text_logits.reshape(-1, tokenizer.vocab_size)
            labels  = targets.reshape(-1)
            loss_txt = crit_txt(logits, labels)

            # optional anti-blur
            loss_ed = edge_loss(pred_image, image_target)

            loss = loss_im + loss_txt + 0.5 * loss_ed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * frames.size(0)

        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        #val loss
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for frames, descriptions, image_target, text_target in val_loader:
                frames = frames.to(device)
                descriptions = descriptions.to(device)
                image_target = image_target.to(device)
                text_target = text_target.to(device)

                pred_image, pred_text_logits, h0, c0 = model(frames, descriptions, text_target)

                loss_im = crit_img(pred_image, image_target)
                targets = text_target[:, 0, 1:]
                logits  = pred_text_logits.reshape(-1, tokenizer.vocab_size)
                labels  = targets.reshape(-1)
                loss_txt = crit_txt(logits, labels)

                loss = loss_im + loss_txt
                val_running += loss.item() * frames.size(0)

        val_loss = val_running / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"[SEQ] epoch {epoch+1}/{epochs_seq} | train={train_loss:.4f} | val={val_loss:.4f}")

        print("Validation panel sample:")
        validation_panel(model, val_loader, tokenizer, device, max_len=80)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1, val_loss,
                os.path.join(ckpt_dir, f"sequence_predictor_epoch_{epoch+1}.pth")
            )

    # final plot
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Train vs Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return model
