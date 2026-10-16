import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.tensorboard import SummaryWriter

SEED = 2077
torch.manual_seed(SEED)

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs" / "09_cyberpunk_gpt"
CHECKPOINT_DIR = ROOT / "checkpoints"
CORPUS_PATH = DATA_DIR / "nightcity_corpus.txt"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Exécution sur : {device}")

# ==========================================
# 1. CRÉATION / CHARGEMENT DU CORPUS
# ==========================================
SAMPLE_DATA = """<|STATUT:CORPO|>
CLIENT: J'ai besoin d'informations sur un convoi de Militech.
FIXER: Regardez-moi ce costume d'Arasaka qui vient s'encanailler à Kabuki... Tu crois que mes infos sont gratuites, gonk ? Crache le double d'eddies tout de suite ou je revends ta position aux Désosseurs. Ici c'est Night City, choom, ton assurance Trauma Team te sauvera pas si tu me prends pour un pigeon.
<|FIN|>

<|STATUT:NETRUNNER|>
CLIENT: J'ai besoin d'informations sur un convoi de Militech.
FIXER: Yo le runner, garde ton cyberdeck au frais. Le convoi passe par Arroyo à minuit, mais leur sous-réseau est blindé de GLACE noire. Si tu tentes un piratage rapide sans mon démon de décryptage, ton cortex va flatline direct. Je te file les codes d'accès, tu me récupères la puce de données. Preem ?
<|FIN|>

<|STATUT:EDGERUNNER|>
CLIENT: J'ai besoin d'informations sur un convoi de Militech.
FIXER: Hahaha, je savais que l'odeur du chrome brûlé t'attirerait, choom ! Quatre blindés, deux méchas, plein de jouets militaires à piller. Vas-y avec ton Sandevistan, fais tout péter, mais touche pas à la caisse rouge à l'arrière sinon mes solos te transforment en passoire chez le charcudoc. Allez, delta !
<|FIN|>

<|STATUT:MERC|>
CLIENT: Tu as un contrat qui paie bien ?
FIXER: T'es qui toi, encore de la chair à canon fraîchement débarquée à Heywood ? J'ai un sale boulot de nettoyage dans un repaire de Maelstrom. Pas de question, tu rentres, tu refroidis tout ce qui bouge et tu me ramènes la mallette. C'est 2000 eddies, à prendre ou à laisser, gonk.
<|FIN|>
"""

if not CORPUS_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Générateur de secours si le corpus riche n'a pas encore été généré
    CORPUS_PATH.write_text(SAMPLE_DATA * 500, encoding="utf-8")
    print(f"Corpus de démo généré dans {CORPUS_PATH} (Pense à l'enrichir avec generate_corpus.py !)")

text = CORPUS_PATH.read_text(encoding="utf-8")

# Vocabulaire caractère (Script 01 & 06)
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s if c in stoi]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"Taille du corpus : {len(text):,} caractères | vocab_size : {vocab_size}")

# ==========================================
# 2. HYPERPARAMÈTRES & DATALOADERS (Script 06)
# ==========================================
batch_size = 32
block_size = 256  # Fenêtre élargie pour lire [STATUT] + Question + Réponse !
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 1e-3
max_steps = 5_000
eval_every = 250
val_batches = 50


class CharDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, block_size_: int) -> None:
        self.data = data_tensor
        self.block_size = block_size_

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]


train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)


def make_train_loader(steps: int) -> DataLoader:
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=steps * batch_size)
    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=RandomSampler(val_dataset, replacement=True, num_samples=val_batches * batch_size),
)


# ==========================================
# 3. ARCHITECTURE TRANSFORMER (Script 06 & 07)
# ==========================================
class Head(nn.Module):
    def __init__(self, n_embd_: int, head_size: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd_, head_size, bias=False)
        self.query = nn.Linear(n_embd_, head_size, bias=False)
        self.value = nn.Linear(n_embd_, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size_, block_size_)))
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(n_embd_, n_embd_ // n_head_, block_size_, dropout_) for _ in range(n_head_)
        )
        self.proj = nn.Linear(n_embd_, n_embd_)
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, n_embd_: int, dropout_: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_, 4 * n_embd_),
            nn.ReLU(),
            nn.Linear(4 * n_embd_, n_embd_),
            nn.Dropout(dropout_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd_, n_head_, block_size_, dropout_)
        self.ffwd = FeedForward(n_embd_, dropout_)
        self.ln1 = nn.LayerNorm(n_embd_)
        self.ln2 = nn.LayerNorm(n_embd_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class CyberpunkGPT(nn.Module):
    def __init__(
        self,
        vocab_size_: int,
        block_size_: int,
        n_embd_: int,
        n_head_: int,
        n_layer_: int,
        dropout_: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size_
        self.token_embedding_table = nn.Embedding(vocab_size_, n_embd_)
        self.position_embedding_table = nn.Embedding(block_size_, n_embd_)
        self.blocks = nn.Sequential(
            *[Block(n_embd_, n_head_, block_size_, dropout_) for _ in range(n_layer_)]
        )
        self.ln_f = nn.LayerNorm(n_embd_)
        self.lm_head = nn.Linear(n_embd_, vocab_size_)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))


# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT (Script 04 & 06)
# ==========================================
@torch.no_grad()
def evaluate(model_: nn.Module, loader: DataLoader) -> float:
    model_.eval()
    total = sum(model_(X.to(device), Y.to(device))[1].item() for X, Y in loader)
    return total / len(loader)


def train_model(model_: nn.Module, run_name: str, steps: int) -> dict[str, float]:
    log_dir = RUNS_DIR / run_name
    shutil.rmtree(log_dir, ignore_errors=True)
    optimizer = torch.optim.AdamW(model_.parameters(), lr=learning_rate)
    train_losses = []
    start = time.perf_counter()
    tr_loss = 0.0
    va_loss = 0.0

    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for step, (X, Y) in enumerate(make_train_loader(steps), start=1):
            model_.train()
            _, loss = model_(X.to(device), Y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping (Script 04) pour éviter les explosions sur T=256
            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if step % eval_every == 0 or step == steps:
                tr_loss = sum(train_losses) / len(train_losses)
                va_loss = evaluate(model_, val_loader)
                writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, step)
                print(f"Pas {step:>5}/{steps} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}")
                train_losses = []

    print(f"Entraînement terminé en {time.perf_counter() - start:.1f}s")
    return {"train": tr_loss, "val": va_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne CyberpunkGPT sur le corpus Night City")
    parser.add_argument("--steps", type=int, default=max_steps, help="Nombre de pas d'entraînement")
    args = parser.parse_args()

    config = {
        "vocab_size_": vocab_size,
        "block_size_": block_size,
        "n_embd_": n_embd,
        "n_head_": n_head,
        "n_layer_": n_layer,
        "dropout_": dropout,
    }
    model = CyberpunkGPT(**config).to(device)
    print(f"Paramètres du modèle : {sum(p.numel() for p in model.parameters()):,}")

    losses = train_model(model, "fixer_v1", args.steps)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "chars": chars,
            "val_loss": losses["val"],
        },
        CHECKPOINT_DIR / "cyberpunk_fixer.pt",
    )
    print(f"Checkpoint sauvegardé dans {CHECKPOINT_DIR / 'cyberpunk_fixer.pt'}")
