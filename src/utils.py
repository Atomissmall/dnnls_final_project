import os
import re
import textwrap
from bs4 import BeautifulSoup

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

#CHECKPOINT UTILITIES
def save_checkpoint(model, optimizer, epoch, loss, ckpt_path: str):
    """
    Save checkpoint to ckpt_path.
    """
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "loss": loss,
    }
    torch.save(payload, ckpt_path)
    print(f"[CKPT] Saved: {ckpt_path} (epoch={epoch}, loss={loss})")


def load_checkpoint(model, optimizer=None, ckpt_path: str = ""):
    """
    Load checkpoint from ckpt_path into model (+ optimizer if provided).
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    epoch = payload.get("epoch", 0)
    loss = payload.get("loss", None)
    print(f"[CKPT] Loaded: {ckpt_path} (epoch={epoch}, loss={loss})")
    return model, optimizer, epoch, loss

#PARSERS
def parse_gdi_text(text: str):
    """
    Parse the dataset 'story' field containing GDI tags.
    Returns a list of dicts (one per frame) with:
    - description
    - objects
    - actions
    - locations
    """
    soup = BeautifulSoup(text, "html.parser")
    frames = []

    for gdi in soup.find_all("gdi"):
        # try to infer image_id if present
        image_id = None
        if gdi.attrs:
            for attr_name in gdi.attrs.keys():
                if "image" in attr_name.lower():
                    image_id = attr_name.lower().replace("image", "")
                    break
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r"<gdi\s+image(\d+)", tag_str)
            if match:
                image_id = match.group(1)
        if not image_id:
            image_id = str(len(frames) + 1)

        content = gdi.get_text().strip()
        objects = [x.get_text().strip() for x in gdi.find_all("gdo")]
        actions = [x.get_text().strip() for x in gdi.find_all("gda")]
        locations = [x.get_text().strip() for x in gdi.find_all("gdl")]

        frames.append(
            {
                "image_id": image_id,
                "description": content,
                "objects": objects,
                "actions": actions,
                "locations": locations,
                "raw_text": str(gdi),
            }
        )

    return frames


def show_image(ax, image_tensor):
    """
    image_tensor: (C,H,W) in [0,1]
    """
    ax.imshow(image_tensor.permute(1, 2, 0))
    ax.axis("off")

#TEXT GENERATION (LSTM)
def generate_text(decoder, h0, c0, tokenizer, device, max_len=80, temperature=0.9):
    """
    Autoregressive generation using DecoderLSTM.
    h0/c0 should be shape: (1, B, H). We generate only for B=1.
    """
    decoder.eval()
    dec_input = torch.tensor([[tokenizer.cls_token_id]], device=device)

    generated = []
    hidden, cell = h0, c0

    for _ in range(max_len):
        with torch.no_grad():
            logits, hidden, cell = decoder(dec_input, hidden, cell)  # (1,1,V)

        logits = logits.squeeze(1)  # (1,V)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

        token_id = next_token.item()
        if token_id == tokenizer.sep_token_id:
            break
        if token_id in [tokenizer.pad_token_id, tokenizer.cls_token_id]:
            dec_input = next_token
            continue

        generated.append(token_id)
        dec_input = next_token

    return tokenizer.decode(generated, skip_special_tokens=True)

#VALIDATION VISUALS
def validation_panel(model, data_loader, tokenizer, device, max_len=80):
    """
    Displays:
    - 4 conditioning frames + their descriptions
    - target frame + target text
    - predicted frame + generated text
    Assumes model forward returns: pred_image, pred_text_logits, h0, c0
    """
    model.eval()
    with torch.no_grad():
        frames, descriptions, image_target, text_target = next(iter(data_loader))

        frames = frames.to(device)
        descriptions = descriptions.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)

        pred_image, pred_text_logits, h0, c0 = model(frames, descriptions, text_target)

        fig, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={"height_ratios": [2, 1.5]})

        #input frames + input text
        for i in range(4):
            show_image(ax[0, i], frames[0, i].cpu())
            txt = tokenizer.decode(descriptions[0, i], skip_special_tokens=True)
            ax[1, i].text(0.5, 0.95, textwrap.fill(txt, 40), ha="center", va="top", fontsize=10)
            ax[1, i].axis("off")

        #target
        show_image(ax[0, 4], image_target[0].cpu())
        ax[0, 4].set_title("Target")
        tgt_txt = tokenizer.decode(text_target[0, 0], skip_special_tokens=True)
        ax[1, 4].text(0.5, 0.95, textwrap.fill(tgt_txt, 40), ha="center", va="top", fontsize=10)
        ax[1, 4].axis("off")

        #predicted
        show_image(ax[0, 5], pred_image[0].cpu())
        ax[0, 5].set_title("Predicted")

        gen_txt = generate_text(model.text_decoder, h0[:, 0:1, :], c0[:, 0:1, :], tokenizer, device, max_len=max_len)
        ax[1, 5].text(0.5, 0.95, textwrap.fill(gen_txt, 40), ha="center", va="top", fontsize=10)
        ax[1, 5].axis("off")

        plt.tight_layout()
        plt.show()

#EDGE LOSS(ANTI-BLUR)
def edge_loss(pred, target):
    """
    Sobel-based edge loss to reduce blur in reconstructions.
    pred/target: (B,C,H,W)
    """
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=pred.device).reshape(1, 1, 3, 3)

    sobel_x = sobel_x.repeat(pred.shape[1], 1, 1, 1)
    sobel_y = sobel_y.repeat(pred.shape[1], 1, 1, 1)

    edges_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
    edges_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
    edges_tgt_x  = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
    edges_tgt_y  = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

    edges_pred = torch.sqrt(edges_pred_x**2 + edges_pred_y**2 + 1e-8)
    edges_tgt  = torch.sqrt(edges_tgt_x**2 + edges_tgt_y**2 + 1e-8)

    return F.l1_loss(edges_pred, edges_tgt)

def make_transforms(image_hw=(60, 125), use_aug=True):
    """
    Returns (train_transform, eval_transform)
    """
    H, W = image_hw

    if use_aug:
        train_transform = T.Compose([
            T.Resize((H, W)),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.35),
            T.RandomApply([T.RandomAffine(degrees=0, translate=(0.05, 0.05))], p=0.25),
            T.ToTensor(),
            T.Lambda(lambda x: x + 0.03 * torch.randn_like(x) if torch.rand(1).item() < 0.25 else x),
            T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        train_transform = T.Compose([
            T.Resize((H, W)),
            T.ToTensor()
        ])

    eval_transform = T.Compose([
        T.Resize((H, W)),
        T.ToTensor()
    ])

    return train_transform, eval_transform
