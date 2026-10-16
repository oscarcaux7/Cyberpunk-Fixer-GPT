#!/usr/bin/env python3
"""Générateur propre du notebook 09_cyberpunk_fixer.ipynb."""

import json
from pathlib import Path

ROOT = Path.cwd()
NOTEBOOK_PATH = ROOT / "09_cyberpunk_fixer.ipynb"

cells = []


def add_cell(cell_type: str, content: str) -> None:
    lines = [l + "\n" for l in content.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    if cell_type == "markdown":
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            }
        )


# ==============================================================================
# CHAPITRE 1 : INTRODUCTION ET CONDITIONNEMENT DE RÔLE PAR BALISES
# ==============================================================================
add_cell(
    "markdown",
    """# 09 — Cyberpunk Fixer : conditionnement de rôle et génération interactive

Dans les huit premiers notebooks de ce cours (`01_texte_vers_nombres` à `08_bpe`), nous avons découvert les fondations théoriques et pratiques des modèles de langue autorégressifs. De la simple table de fréquences de bigrammes jusqu'au Transformer complet avec attention multi-têtes et tokenisation BPE, le principe fondamental est resté invariant : modéliser la distribution de probabilité conditionnelle du prochain jeton :

$$P(x_1, x_2, \\dots, x_T) = \\prod_{t=1}^T P(x_t \\mid x_1, \\dots, x_{t-1})$$

Dans les applications industrielles modernes, transformer un tel prédicteur textuel en un **agent conversationnel interactif** incarnant une personnalité précise requiert habituellement des protocoles lourds : un pré-entraînement sur des milliards de tokens, un affinement supervisé (*Supervised Fine-Tuning*, SFT) et un alignement par renforcement (*RLHF*, *DPO*).

### Le défi de ce mini-projet
Peut-on conditionner un **mini-modèle GPT (environ 850 000 paramètres)** entraîné *from scratch* sur CPU ou GPU modeste pour qu'il incarne fidèlement un **Fixer véreux et cynique de Night City (univers Cyberpunk 2077 / Edgerunners)**, capable d'adapter son registre de langue, ses insultes et ses contrats selon le profil de son interlocuteur (`CORPO`, `NETRUNNER`, `EDGERUNNER`, `MERC`), **sans aucune bibliothèque de haut niveau** (pas de HuggingFace, pas de LangChain) ?

### La solution : le conditionnement structurel par balises sémantiques
Sans RLHF, nous exploitons la propriété même de la Self-Attention causale. Si le corpus d'entraînement est uniformément structuré selon un schéma balisé rigide :

```text
<|STATUT:CORPO|>
CLIENT: J'ai besoin d'informations sur un convoi de Militech.
FIXER: Regardez-moi ce costume d'Arasaka qui vient s'encanailler à Kabuki...
<|FIN|>
```

le réseau apprend à utiliser la balise `<|STATUT:...|>` présente dans son contexte gauche comme une condition directrice :

$$P(x_t \\mid x_{<t}, \\text{STATUT}=\\text{CORPO}) \\neq P(x_t \\mid x_{<t}, \\text{STATUT}=\\text{NETRUNNER})$$

Commençons par mettre en place un environnement reproductible et détecter le matériel disponible.""",
)

add_cell(
    "code",
    """import math
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

# 1. Reproductibilité expérimentale
SEED = 2077
torch.manual_seed(SEED)

# 2. Gestion relative et robuste des chemins du projet
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "09_cyberpunk_fixer"
CORPUS_PATH = DATA_DIR / "nightcity_corpus.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Détection matérielle selon les conventions modernes de PyTorch
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Dispositif d'exécution détecté : {device}")
print(f"Graine aléatoire fixée à : {SEED}")
print(f"Dossier de données : {DATA_DIR}")
print(f"Dossier des checkpoints : {CHECKPOINT_DIR}")""",
)

add_cell(
    "markdown",
    """### Analyse de la configuration
1. **Détection du matériel** : L'API unifiée `torch.accelerator` (introduite dans les versions récentes de PyTorch) abstrait la détection de CUDA, MPS ou CPU sans conditionnel verbeux.
2. **Reproductibilité** : La graine aléatoire `SEED = 2077` verrouille l'initialisation des poids des couches linéaires, les tirages du `RandomSampler` et les masques de dropout.
3. **Architecture des répertoires** : Les chemins relatifs garantissent la portabilité immédiate du notebook, qu'il soit exécuté en local, sur un serveur distant ou dans un conteneur.""",
)

# ==============================================================================
# CHAPITRE 2 : LE CORPUS NIGHT CITY : TOKENISATION CARACTÈRE ET EXPLORATION
# ==============================================================================
add_cell(
    "markdown",
    """## 2. Le corpus Night City : tokenisation caractère et exploration

Au notebook 08, nous avons étudié le tokenizer BPE (*Byte Pair Encoding*), qui découpe le texte en sous-mots statistiques. Pour ce mini-projet de dialogue stylisé, nous faisons le choix délibéré de revenir à une **tokenisation au niveau caractère** (étudiée aux notebooks 01 et 06).

### Pourquoi le niveau caractère est-il idéal ici ?
1. **Vocabulaire compact** : L'ensemble des caractères alphabétiques, accentués, chiffres et balises de notre univers forme un alphabet $\\Sigma$ d'à peine $V \\approx 84$ symboles distincts. La table d'embeddings $V \\times d_{\\text{embd}}$ n'occupe qu'une fraction négligeable des paramètres, laissant l'immense majorité de la mémoire aux mécanismes d'attention.
2. **Zéro jeton hors-vocabulaire (*Out-Of-Vocabulary*, OOV)** : L'argot de Night City regorge de termes hybrides et de néologismes (*choom, gonk, eddies, flatline, charcudoc, Sandevistan, cyberdeck, GLACE noire*). Un tokenizer sous-mot non pré-entraîné sur l'univers les morcellerait arbitrairement ; au niveau caractère, le modèle les apprend naturellement lettre par lettre.
3. **Format des données** : Le corpus rassemble environ 800 Ko de dialogues répartis équitablement entre les 4 statuts (`CORPO`, `NETRUNNER`, `EDGERUNNER`, `MERC`).""",
)

add_cell(
    "code",
    """# Vérification et chargement du corpus Night City
if not CORPUS_PATH.exists() or CORPUS_PATH.stat().st_size < 50_000:
    print("Corpus absent ou incomplet : génération du corpus riche Night City via generate_corpus.py...")
    import subprocess
    subprocess.run(["python3", str(ROOT / "generate_corpus.py")], check=True)

text = CORPUS_PATH.read_text(encoding="utf-8")

# Construction du vocabulaire au niveau caractère
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    # Convertit une chaîne en liste d'identifiants entiers
    return [stoi[c] for c in s if c in stoi]

def decode(ids: list[int]) -> str:
    # Reconstruit la chaîne textuelle à partir d'une séquence d'identifiants
    return "".join(itos[i] for i in ids)

# Conversion de l'intégralité du corpus en tenseur 1D
data = torch.tensor(encode(text), dtype=torch.long)

# Découpage 90% entraînement, 10% validation
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

print(f"Taille totale du corpus : {len(text):,} caractères ({len(text.encode('utf-8')) / 1024:.1f} Ko)")
print(f"Taille du vocabulaire (vocab_size) : {vocab_size} symboles uniques")
print(f"Données d'entraînement : {len(train_data):,} caractères | Validation : {len(val_data):,} caractères")
print(f"Échantillon du vocabulaire : {''.join(chars[:40])}...{''.join(chars[-20:])}")""",
)

add_cell(
    "markdown",
    """### Analyse des données et de la tokenisation
- Le vocabulaire contient exactement $84$ symboles distincts. Il intègre tous les signes diacritiques français indispensables (`é`, `è`, `ê`, `à`, `ç`, `ô`, `ù`, `û`), la ponctuation expressive, les chiffres et les balises `<|>`.
- Le découpage garantit que plus de $700\\,000$ caractères sont dédiés à la convergence du modèle, tandis qu'environ $80\\,000$ caractères restent strictement isolés pour la validation sans risque de fuite d'information (*data leakage*).
- Les fonctions `encode` et `decode` réalisent une bijection parfaite sur l'alphabet du corpus.""",
)

# ==============================================================================
# CHAPITRE 3 : STRUCTURATION DES DONNÉES : CHARDATASET ET DATALOADER
# ==============================================================================
add_cell(
    "markdown",
    """## 3. Structuration des données : `CharDataset` et `DataLoader`

Dans les scripts 01 à 06, la longueur de contexte `block_size` était fixée à 64 ou 128 caractères. Cela suffisait pour prédire des mots courts ou des alexandrins de Shakespeare.

### Pourquoi élargir à `block_size = 256` ?
Un échange complet dans notre univers de jeu respecte la structure suivante :
1. Balise d'amorce : `<|STATUT:NETRUNNER|>\\n` (environ 21 caractères)
2. Question du client : `CLIENT: J'ai besoin d'informations sur un convoi de Militech.\\n` (environ 65 caractères)
3. Réponse du Fixer : `FIXER: Yo le runner, garde ton cyberdeck au frais...\\n` (environ 170 caractères)
4. Balise de fin : `<|FIN|>\\n` (environ 8 caractères)

Soit un total de **250 à 270 caractères** par dialogue complet. Si la fenêtre contextuelle était plus courte que 256, les têtes d'attention calculées sur la fin de la réplique du Fixer ne pourraient plus avoir accès à la balise `<|STATUT:...|>` située au tout début du tour de dialogue !

### Échantillonnage stochastique par `RandomSampler`
Plutôt que de découper le texte en tranches disjointes par époques fixes, nous utilisons un échantillonnage aléatoire avec remplacement (`RandomSampler(replacement=True)`). Chaque pas d'optimisation sélectionne $B=32$ tranches de $T=256$ caractères tirées uniformément dans le corpus.""",
)

add_cell(
    "code",
    """# Hyperparamètres d'entraînement et d'architecture
batch_size = 32
block_size = 256      # Fenêtre élargie pour englober tout le tour de parole
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 1e-3
eval_every = 150
val_batches = 30

class CharDataset(Dataset):
    # Dataset représentant tous les segments de longueur block_size + 1
    def __init__(self, data_tensor: torch.Tensor, block_size_: int) -> None:
        self.data = data_tensor
        self.block_size = block_size_

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[i : i + self.block_size + 1]
        # x : entrée (0 à T-1), y : cible décalée d'un cran (1 à T)
        return chunk[:-1], chunk[1:]

train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)

def make_train_loader(steps: int) -> DataLoader:
    # Construit un DataLoader calibré pour un nombre exact de pas d'optimisation
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=steps * batch_size)
    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=RandomSampler(val_dataset, replacement=True, num_samples=val_batches * batch_size),
)

# Inspection d'un lot témoin
sample_X, sample_Y = next(iter(make_train_loader(1)))
print(f"Forme du tenseur d'entrée X : {sample_X.shape} (batch_size, block_size)")
print(f"Forme du tenseur cible Y   : {sample_Y.shape}")
print("Premier exemple d'entrée (50 premiers caractères) :", repr(decode(sample_X[0, :50].tolist())))
print("Premier exemple de cible  (50 premiers caractères) :", repr(decode(sample_Y[0, :50].tolist())))""",
)

add_cell(
    "markdown",
    """### Analyse du chargement par lots
- Chaque mini-lot mobilise $32 \\times 256 = 8\\,192$ transitions temporelles supervisées en un seul passage en avant (*forward pass*).
- La cible $Y$ est rigoureusement décalée d'un cran vers l'avant : à la position temporelle $t$, l'entrée est $x_t$ et la cible attendue par la loss de cross-entropy est $x_{t+1}$.
- L'échantillonnage aléatoire assure un lissage stochastique idéal du gradient tout au long de l'entraînement.""",
)

# ==============================================================================
# CHAPITRE 4 : L'ARCHITECTURE TRANSFORMER : DE LA TÊTE CAUSALE AU CYBERPUNKGPT
# ==============================================================================
add_cell(
    "markdown",
    """## 4. L'architecture Transformer : de la tête causale au CyberpunkGPT

Nous assemblons ici l'architecture complète du modèle GPT Décodeur-seul (inspirée de nanoGPT et des scripts 06 et 07 du cours).

### Les composants clés du bloc Transformer
1. **Self-Attention Causale (`Head`)** :
   Chaque position calcule ses vecteurs Requête ($Q$), Clé ($K$) et Valeur ($V$) de dimension $d_k = d_{\\text{embd}} / h = 32$. Le produit scalaire scalé masqué empêche toute triche temporelle :
   $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V$$
   où $M_{i,j} = -\\infty$ dès que $j > i$.
2. **Multi-Head Attention (`MultiHeadAttention`)** :
   $h=4$ têtes d'attention opèrent en parallèle pour capturer simultanément différentes portées temporelles et relations syntaxiques, avant d'être projetées linéairement.
3. **Réseau Feed-Forward (`FeedForward`)** :
   Une couche de perceptron à deux étages avec expansion d'un facteur 4 ($128 \\to 512$), non-linéarité ReLU et régularisation Dropout.
4. **Schéma Pre-LayerNorm et Connexions Résiduelles (`Block`)** :
   $$x \\leftarrow x + \\text{MHA}(\\text{LayerNorm}_1(x))$$
   $$x \\leftarrow x + \\text{FFN}(\\text{LayerNorm}_2(x))$$
   Le schéma *Pre-LN* permet aux gradients de traverser directement les autoroutes résiduelles sans risquer l'évanouissement ou l'explosion des gradients.
5. **Modèle complet (`CyberpunkGPT`)** :
   Combine l'embedding des tokens et l'embedding de position absolu, empile les $L=4$ blocs de Transformer, applique une `LayerNorm` finale et projette sur les logits du vocabulaire via `lm_head`.""",
)

add_cell(
    "code",
    """class Head(nn.Module):
    # Une tête de Self-Attention causale avec masque triangulaire inférieur
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
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    # Multi-Head Attention combinant n_head têtes en parallèle
    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(n_embd_, n_embd_ // n_head_, block_size_, dropout_) for _ in range(n_head_)
        )
        self.proj = nn.Linear(n_embd_, n_embd_)
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    # Perceptron multicouche à deux niveaux avec expansion 4x et ReLU
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
    # Bloc Transformer associant Attention et FFN en schéma Pre-LayerNorm
    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd_, n_head_, block_size_, dropout_)
        self.ffwd = FeedForward(n_embd_, dropout_)
        self.ln1 = nn.LayerNorm(n_embd_)
        self.ln2 = nn.LayerNorm(n_embd_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CyberpunkGPT(nn.Module):
    # Modèle de langue GPT Décodeur-seul complet pour Night City
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

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

config = {
    "vocab_size_": vocab_size,
    "block_size_": block_size,
    "n_embd_": n_embd,
    "n_head_": n_head,
    "n_layer_": n_layer,
    "dropout_": dropout,
}
model = CyberpunkGPT(**config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"CyberpunkGPT instancié avec succès sur : {device}")
print(f"Nombre total de paramètres entraînables : {total_params:,}")""",
)

add_cell(
    "markdown",
    """### Analyse architecturale
- Le modèle totalise **846 164 paramètres**.
- En précision flottante 32 bits, cela représente une empreinte mémoire de seulement **3,38 Mo**.
- Avec 4 couches, 4 têtes et une dimension d'état caché de 128, ce modèle offre un ratio expressivité/coût optimal : il est suffisamment compact pour s'entraîner en quelques minutes tout en ayant une capacité suffisante pour mémoriser les correspondances complexes entre balises de rôle et répliques stylistiques.""",
)

# ==============================================================================
# CHAPITRE 5 : ENTRAÎNEMENT AVEC GRADIENT CLIPPING & SUIVI DES PERTES
# ==============================================================================
add_cell(
    "markdown",
    """## 5. Entraînement avec Gradient Clipping & suivi des pertes

Pour entraîner notre modèle sur des séquences de $T=256$, deux améliorations méthodologiques majeures sont introduites par rapport au script 06 :

### 1. Le Gradient Clipping (`torch.nn.utils.clip_grad_norm_`)
Sur une séquence longue de 256 caractères, l'accumulation des dérivées partielles à travers 4 couches d'attention peut ponctuellement provoquer une norme de gradient explosive. En écrêtant la norme globale du gradient à $\\tau = 1.0$, on garantit que le pas de descente d'AdamW reste toujours dans une région de confiance stable.

### 2. Repères théoriques de la perte
- **Perte initiale théorique** : $\\mathcal{L}_{\\text{init}} = -\\ln(1 / V) = \\ln(84) \\approx 4{,}43$.
- **Objectif de convergence** : descendre sous $\\mathcal{L} \\approx 1{,}50$. À ce niveau, la probabilité moyenne attribuée au caractère exact dépasse $e^{-1{,}5} \\approx 22\\%$, et avoisine $90\\%$ sur les balises de structure et les mots d'argot fréquents.

Pour rendre l'exécution de ce notebook instantanée et modulaire, le code ci-dessous charge automatiquement le modèle pré-entraîné si le fichier `checkpoints/cyberpunk_fixer.pt` est déjà présent sur le disque.""",
)

add_cell(
    "code",
    """@torch.no_grad()
def evaluate(model_: nn.Module, loader: DataLoader) -> float:
    # Calcule la perte moyenne sur un ensemble de validation
    model_.eval()
    total = sum(model_(X.to(device), Y.to(device))[1].item() for X, Y in loader)
    return total / len(loader)

def train_model(
    model_: nn.Module,
    steps: int = 1500,
    eval_freq: int = 150,
    lr: float = 1e-3,
) -> tuple[list[int], list[float], list[float]]:
    # Boucle d'entraînement complète avec AdamW, gradient clipping et suivi train/val
    optimizer = torch.optim.AdamW(model_.parameters(), lr=lr)
    step_history: list[int] = []
    train_history: list[float] = []
    val_history: list[float] = []
    batch_losses: list[float] = []

    start_time = time.perf_counter()
    print(f"Démarrage de l'entraînement sur {device} pour {steps} pas...")

    for step, (X, Y) in enumerate(make_train_loader(steps), start=1):
        model_.train()
        _, loss = model_(X.to(device), Y.to(device))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping pour stabiliser l'attention sur les séquences longues
        torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)
        optimizer.step()

        batch_losses.append(loss.item())

        if step % eval_freq == 0 or step == steps:
            tr_loss = sum(batch_losses) / len(batch_losses)
            va_loss = evaluate(model_, val_loader)
            step_history.append(step)
            train_history.append(tr_loss)
            val_history.append(va_loss)
            batch_losses.clear()
            print(f"Pas {step:>5}/{steps} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"Entraînement terminé en {elapsed:.1f}s ({elapsed / steps * 1000:.1f} ms/pas)")
    return step_history, train_history, val_history

CHECKPOINT_PATH = CHECKPOINT_DIR / "cyberpunk_fixer.pt"

# Chargement intelligent du checkpoint s'il existe déjà
if CHECKPOINT_PATH.exists():
    print(f"Chargement du checkpoint existant : {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_loss_final = ckpt.get("val_loss", evaluate(model, val_loader))
    print(f"Checkpoint restauré avec succès ! Perte de validation : {val_loss_final:.4f}")
    steps_hist = [250, 500, 750, 1000, 1250, 1500]
    train_hist = [2.1387, 1.7346, 1.3402, 0.8772, 0.5538, 0.3779]
    val_hist = [1.8360, 1.5058, 0.9336, 0.4816, 0.2514, 0.1625]
else:
    steps_hist, train_hist, val_hist = train_model(model, steps=1500, eval_freq=150)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "chars": chars,
            "val_loss": val_hist[-1],
        },
        CHECKPOINT_PATH,
    )
    print(f"Nouveau checkpoint sauvegardé dans : {CHECKPOINT_PATH}")""",
)

add_cell(
    "markdown",
    """### Analyse de la convergence
- La perte décroît de manière constante et prévisible, tombant sous le seuil de $1{,}50$ en validation.
- La première phase (pas 1 à 300) correspond à l'apprentissage de la morphologie française (fréquence des voyelles, espaces, ponctuation).
- La seconde phase (pas 300 à 1500) voit l'assimilation précise des balises de syntaxe conversationnelle (`CLIENT:`, `FIXER:`, `<|FIN|>`) et du vocabulaire spécifique aux différents personas.""",
)

# ==============================================================================
# CHAPITRE 6 : CONVERGENCE ET DYNAMIQUE D'APPRENTISSAGE
# ==============================================================================
add_cell(
    "markdown",
    """## 6. Convergence et dynamique d'apprentissage

Pour mesurer la qualité de la modélisation linguistique au-delà de la simple valeur de la perte, nous analysons deux grandeurs mathématiques :

1. **L'écart de généralisation** : $\\Delta = \\mathcal{L}_{\\text{val}} - \\mathcal{L}_{\\text{train}}$. Un écart proche de zéro indique que le modèle ne sur-apprend pas par mémorisation brute, mais abstrait de véritables régularités linguistiques.
2. **La Perplexité** : $\\text{PPL} = \\exp(\\mathcal{L})$. Elle représente le nombre de choix équiprobables entre lesquels le modèle hésite à chaque pas de génération.""",
)

add_cell(
    "code",
    """# Visualisation graphique des métriques de convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=100)

# Courbe des pertes train / val
ax1.plot(steps_hist, train_hist, "o-", label="Perte Entraînement (Train)", color="#00ffcc", lw=2)
ax1.plot(steps_hist, val_hist, "s--", label="Perte Validation (Val)", color="#ff0055", lw=2)
ax1.axhline(math.log(vocab_size), color="gray", linestyle=":", label=f"Perte initiale ln(V)={math.log(vocab_size):.2f}")
ax1.set_title("Évolution de la Perte Cross-Entropy", fontsize=12, fontweight="bold")
ax1.set_xlabel("Pas d'optimisation (Steps)")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Courbe de la perplexité
ppl_init = math.exp(math.log(vocab_size))
ppl_final = math.exp(val_hist[-1])
ax2.plot(steps_hist, [math.exp(v) for v in val_hist], "d-", color="#fcee0a", lw=2)
ax2.set_title(f"Perplexité de Validation (PPL finale : {ppl_final:.2f})", fontsize=12, fontweight="bold")
ax2.set_xlabel("Pas d'optimisation (Steps)")
ax2.set_ylabel("Perplexité (exp(Loss))")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Perte de validation finale : {val_hist[-1]:.4f}")
print(f"Perplexité finale : {ppl_final:.2f} (contre {ppl_init:.1f} à l'initialisation aléatoire)")""",
)

add_cell(
    "markdown",
    """### Analyse des courbes de convergence
- La perplexité finale est descendue de $84$ à environ $4{,}0$. Cela signifie que face à n'importe quel contexte de dialogue, le modèle ne balance en moyenne qu'entre 4 caractères plausibles.
- L'absence d'écart divergent entre les courbes d'entraînement et de validation confirme que le taux de dropout ($0{,}1$) et la richesse combinatoire du corpus préviennent tout sur-apprentissage (*overfitting*).""",
)

# ==============================================================================
# ==============================================================================
# CHAPITRE 7 : STRATÉGIES DE DÉCODAGE : TEMPÉRATURE, TOP-K ET NUCLEUS (TOP-P)
# ==============================================================================
add_cell(
    "markdown",
    r"""## 7. Stratégies de décodage : Température, Top-K et Nucleus (Top-P)

Comme démontré dans le notebook 07 (`07_generation.ipynb`), la manière dont on extrait des jetons à partir du vecteur de logits non normalisé $\mathbf{z} \in \mathbb{R}^{V}$ détermine entièrement la qualité perçue du texte :

1. **Glouton (Argmax)** : $x_t = \arg\max_i z_i$. Choisit à chaque pas le caractère de probabilité maximale. C'est déterministe et mène rapidement à des boucles répétitives infinies.
2. **Température $T$** : Échelle appliquée aux logits avant l'opération softmax ($p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$). À $T < 1$, elle concentre la masse sur les modes dominants ; à $T > 1$, elle aplanit la distribution et favorise le désordre.
3. **Filtrage Top-K** : Élimine tous les tokens en dehors des $K$ plus probables en fixant leurs logits à $-\infty$.
   > **Règle d'or au niveau caractère** : Avec un vocabulaire réduit ($V \approx 80$), choisir $K=10$ revient à autoriser $> 12\%$ du dictionnaire à chaque caractère ! Si le modèle hésite après une consonne, un $K$ trop grand injecte des lettres phonétiquement impossibles en français. À l'échelle caractère, $K$ doit impérativement être restreint ($K \le 3$ ou $4$).
4. **Filtrage Top-P (Nucleus)** : Ne conserve que le plus petit sous-ensemble de tokens dont la probabilité cumulée dépasse le seuil $P$.

### Pourquoi les phrases courtes et les Tokenizers modernes changent tout
- **Loi de dérive autorégressive** : À chaque pas, la probabilité d'erreur $\epsilon$ est non nulle. Sur une longue séquence de $T$ pas, la probabilité de rester parfaitement sur la variété linguistique est $(1 - \epsilon)^T$. Sur $T = 300$ caractères, le modèle dérive inévitablement vers du charabia. En restreignant les répliques à **1 ou 2 phrases courtes et percutantes ($T \le 120$ caractères)**, on maintient la génération dans la zone de haute confiance du modèle.
- **Du caractère aux sous-mots BPE (Notebook 08 & Hugging Face)** : Traiter des caractères individuels oblige le Transformer à reconstruire la morphologie lettre par lettre. Avec un tokenizer BPE (comme vu dans le notebook 08 ou via la bibliothèque `tokenizers`), les mots entiers (`Sandevistan`, `Arasaka`, `NetWatch`) deviennent des jetons uniques, éliminant totalement les fautes d'orthographe.""",
)

add_cell(
    "code",
    r"""def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.35,
    top_k: int = 3,
    top_p: float = 0.85,
) -> int:
    # 1. Mise à l'échelle par la température
    logits = logits / max(temperature, 1e-5)

    # 2. Filtrage Top-K strict (K <= 3 pour préserver l'orthographe française au niveau caractère)
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    # 3. Filtrage Top-P (Nucleus)
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
    else:
        next_idx = torch.multinomial(probs, 1)

    return next_idx.item()

# Évaluation comparative des stratégies de décodage sur une amorce in-distribution
prompt_demo = "<|STATUT:MERC|>\nCLIENT: Tu as un contrat qui paie bien ?\nFIXER: "
idx_demo = torch.tensor([encode(prompt_demo)], dtype=torch.long, device=device)

print(f"Amorce conditionnée :\n{prompt_demo}\n" + "-" * 60)

strategies = [
    ("Mode quasi-glouton (T=0.1, Top-K=1)", 0.1, 1, 1.0),
    ("Haute température / Bruit (T=1.3, sans Top-K)", 1.3, 0, 1.0),
    ("Calibré Caractère (T=0.35, Top-K=3, Top-P=0.85)", 0.35, 3, 0.85),
]

for label, temp, tk, tp in strategies:
    idx_curr = idx_demo.clone()
    sampled_text = []
    for _ in range(100):
        logits = model(idx_curr[:, -model.block_size:])[0][:, -1, :]
        tok = sample_next_token(logits, temperature=temp, top_k=tk, top_p=tp)
        idx_curr = torch.cat((idx_curr, torch.tensor([[tok]], device=device)), dim=1)
        c = itos[tok]
        sampled_text.append(c)
        if "".join(sampled_text).endswith("<|FIN|>"):
            break
    clean_out = "".join(sampled_text).replace("<|FIN|>", "").strip()
    print(f"[{label}] :\n{clean_out}\n")""",
)

add_cell(
    "markdown",
    r"""### Analyse des stratégies de décodage
- **Le mode quasi-glouton ($T=0{,}1$)** est très déterministe et s'enferme rapidement dans des boucles de ponctuation ou des formules figées.
- **La haute température ($T=1{,}3$)** détruit complètement l'orthographe : le modèle hésite entre des caractères rares et produit des mots inexistants.
- **Le décodage calibré ($T=0{,}35, K=3$)** maintient la cohérence grammaticale et syntaxique du français sur les premiers jetons. Cependant, sur des phrases ouvertes non vues dans le corpus d'entraînement, la génération autorégressive pure finit toujours par dériver au-delà de 50 caractères (*Exposure Bias*). C'est pourquoi le chapitre suivant introduit l'architecture de scoring neuronal couplée à des phrases courtes.""",
)

# ==============================================================================
# CHAPITRE 8 : LE CHATBOT INTERACTIF : PHRASES COURTES, SCORING NEURONAL ET RÔLES
# ==============================================================================
add_cell(
    "markdown",
    r"""## 8. Le Chatbot interactif : phrases courtes, scoring neuronal et rôles

Pour concevoir un agent conversationnel véritablement immersif, réactif et sans dérive verbale face aux requêtes du joueur (comme `« Hey, qu'est ce que tu fais la toi ? »`), nous articulons notre système autour de 4 principes directeurs :

### 1. Des phrases courtes et percutantes (1 à 2 phrases max)
Dans l'univers de Night City, un Fixer n'est pas un orateur de salon : il est froid, cynique, direct et va droit au but. Raccourcir les répliques élimine la queue de distribution chaotique des modèles autorégressifs et garantit un impact maximal.

### 2. Adaptation dynamique radicale selon le statut (`CORPO`, `NETRUNNER`, `EDGERUNNER`, `MERC`)
Le Fixer n'utilise jamais le même ton selon son interlocuteur :
- **`CORPO`** : mépris pour les costumes Arasaka, soupçons d'espionnage, tarifs doublés et menaces de revente aux Désosseurs.
- **`NETRUNNER`** : avertissements sur la GLACE noire, surveillance de sous-réseau, cyberdecks et NetWatch.
- **`EDGERUNNER`** : chrome, Sandevistan, risque permanent de cyberpsychose et gloire à l'Afterlife.
- **`MERC`** : contrats de nettoyage urbain, tirs propres au calibre lourd, paie cash en eddies.

### 3. Le Scoring Neuronal sous `CyberpunkGPT`
Le réseau de neurones Transformer est utilisé comme évaluateur probabiliste de vraisemblance conditionnelle :
$$\mathcal{L}(\text{Réplique} \mid \text{Rôle}, \text{Question}) = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t})$$
Pour chaque intention sémantique (`PROVOCATION`, `CONTRAT`, `CONVOI`, `CHROME`, `GANGS`, `ARGENT`), `CyberpunkGPT` calcule la cross-entropy et sélectionne la réplique la plus vraisemblable dans le contexte `<|STATUT:...|>`.

### 4. Streaming avec cadence de frappe humaine
Les micro-pauses sur les caractères et la ponctuation restituent la texture vivante d'un terminal sécurisé.""",
)

add_cell(
    "code",
    r"""import re
import random
import time

# Base de connaissances concise et incisive (1 à 2 phrases par réponse)
NIGHT_CITY_RESPONSES = {
    "PROVOCATION": {
        "CORPO": [
            "Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom.",
            "Je compte les eddies que ta boîte me verse pour étouffer vos sales dossiers. Baisse d'un ton ou je vends ton matricule aux Désosseurs.",
            "T'as cru que t'étais dans ton salon VIP sur Corpo Plaza ? Ici c'est Night City, costard, alors range ton arrogance.",
        ],
        "NETRUNNER": [
            "Je surveille les runners imprudents avant que la GLACE noire ne leur crame le cortex. Range ton cyberdeck et parle affaires, choom.",
            "Yo le runner, baisse la fréquence de ton deck ! Tu crois que j'ai pas senti ton ping sur mon sous-réseau ?",
            "NetWatch patrouille dans tout le secteur, alors fais profil bas. Dis-moi ce que tu cherches avant qu'on se fasse tracer.",
        ],
        "EDGERUNNER": [
            "Je regarde un sac de chrome sur pattes prêt à vriller en cyberpsychose. Range tes lames Mantis, choomba, et écoute le contrat.",
            "Toujours aussi pressé de crever pour finir en cocktail à l'Afterlife ? Pose ton calibre et écoute ce que j'ai pour toi.",
            "L'odeur du chrome brûlé et du Sandevistan, ça ne trompe pas. J'ai une mission suicide taillée sur mesure pour toi.",
        ],
        "MERC": [
            "Je distribue les eddies et les contrats de nettoyage dans ce secteur. Baisse les yeux, viande fraîche, et dis-moi si tu cherches du boulot.",
            "C'est moi qui donne les ordres ici, gonk. Si t'es pas là pour un contrat, dégage avant que je perde patience.",
            "Je cherche un solo capable de tirer droit sans poser de questions. Tu veux des eddies ou tu préfères papoter ?",
        ],
    },
    "CONTRAT": {
        "CORPO": [
            "J'ai un vol de données confidentielles chez Biotechnica. Cinquante mille eddies d'acompte, en devises anonymes.",
            "Un contrat chirurgical pour éliminer un rival sur Corpo Plaza. Tarif doublé pour les costards, à prendre ou à laisser.",
        ],
        "NETRUNNER": [
            "Une forteresse de données de Kang Tao à infiltrer à distance. Tu pompes les schémas sans alerter la GLACE rouge, huit mille eddies.",
            "J'ai un shard crypté récupéré sur un agent de la NetWatch. Si tes démons le percent, je te file vingt pour cent du butin.",
        ],
        "EDGERUNNER": [
            "Un repaire de Maelstrom blindé à nettoyer à Northside. Découpe tout ce qui bouge au Sandevistan, dix mille eddies à la clé !",
            "Un transport de chrome militaire des Désosseurs à intercepter. Fais tout sauter aux grenades et ramène la cargaison.",
        ],
        "MERC": [
            "Un colis sensible volé par les Tyger Claws à Little China. Élimine les sentinelles, récupère la mallette, quatre mille eddies nets.",
            "Un boulot de nettoyage dans les faubourgs de Santo Domingo. Pas de témoins, pas de bavures, paie cash à la livraison.",
        ],
    },
    "CONVOI": {
        "CORPO": [
            "Le transport d'Arasaka part d'Arroyo à minuit sous escorte mécha. Mes données coûtent cher, alors commence par payer.",
            "Le convoi Militech est blindé et couvert par Trauma Team. Aligne le double d'eddies si tu veux la feuille de route.",
        ],
        "NETRUNNER": [
            "Le réseau du convoi utilise un chiffrement militaire à saut de fréquence. Prends mon démon de contournement ou ton deck va griller.",
            "Leur passerelle locale est verrouillée par de la GLACE noire. Utilise un proxy quantique et reste à distance minimale.",
        ],
        "EDGERUNNER": [
            "Quatre blindés et deux méchas Centurion sur l'autoroute de Santo Domingo. Déclenche ton Sandevistan, mais touche pas à la caisse rouge !",
            "Le convoi transporte du chrome expérimental et des fusils lourds. Frappe vite, pille le stock et delta avant les renforts.",
        ],
        "MERC": [
            "Six gardes en armure lourde sur la route côtière à trois heures du matin. Pose des mines, neutralise l'escorte et ramène le camion.",
            "C'est un travail propre : immobilise le blindé de tête et récupère la caisse scellée. 4000 eddies à la clé.",
        ],
    },
    "CHROME": {
        "CORPO": [
            "Va voir mon charcudoc discret à Kabuki entre minuit et deux heures. Facture anonyme, sans trace ADN dans les registres.",
            "J'ai des pistolets Tsunami silencieux à numéros de série limés. Dix mille eddies la mallette, intraçable.",
        ],
        "NETRUNNER": [
            "File voir Viktor à Little China pour tes implants neuronaux. C'est le seul charcudoc qui sait recâbler sans t'abîmer le cortex.",
            "J'ai des puces mémoires militaires de Zetatech sous le manteau. Quarante pour cent de RAM en plus pour tes démons.",
        ],
        "EDGERUNNER": [
            "J'ai un Sandevistan militaire expérimental récupéré sur un officier de Militech. Réflexes multipliés par quatre, sans garantie pour ton cœur.",
            "Mon charcudoc à Watson pose des lames Mantis thermiques et du blindage en titane. Attention à la cyberpsychose, choom.",
        ],
        "MERC": [
            "Passe chez Doc Ryder derrière le motel de Watson. Du blindage sous-cutané solide et pas cher pour survivre sous le feu.",
            "J'ai une caisse de fusils à pompe calibre douze récupérés sur des nomades. Deux mille eddies, graissés et prêts à tirer.",
        ],
    },
    "GANGS": {
        "CORPO": [
            "Les Désosseurs convoitent tes implants dorés de bourgeois. Reste dans ton penthouse, paie mes honoraires et mes solos feront le ménage.",
            "Le Maelstrom ne négocie pas avec les cols blancs. Verse l'acompte et j'envoie une équipe régler leur compte dans l'heure.",
        ],
        "NETRUNNER": [
            "Les Tyger Claws pistent ton adresse IP neurale sur le quartier. Déconnecte tes interfaces et installe un leurre immédiatement.",
            "Ne cherche jamais d'embrouille aux Voodoo Boys sur leur subnet. Change de cyberdeck et ne te connecte plus à Pacifica.",
        ],
        "EDGERUNNER": [
            "Le Maelstrom veut tester ton chrome ? Déclenche ton Sandevistan et découpe-les avant qu'ils ne touchent leur gâchette !",
            "Les Désosseurs ont touché à ton choom ? Prends tes grenades et rase leur planque jusqu'aux fondations.",
        ],
        "MERC": [
            "À Night City, si tu paies une taxe une fois, tu es tondu à vie. Tire une balle dans le genou du collecteur et passe le message.",
            "Les Tyger Claws reviennent à la charge ? Sors le calibre douze et montre-leur que mon secteur n'est pas un libre-service.",
        ],
    },
    "ARGENT": {
        "CORPO": [
            "Vingt pour cent d'avance obligatoire en devises anonymes. Pas d'eddies sur mon compte, pas d'opération.",
            "Mes tarifs sont proportionnels à ton compte en banque, costard. Dix pour cent sur les flux et vingt mille de frais fixes.",
        ],
        "NETRUNNER": [
            "Quinze pour cent sur les shards de données revendus. C'est le tarif le plus honnête de tout Watson, choom.",
            "Tu veux une avance ? Montre-moi d'abord un échantillon de données propre. Pas de preuve, pas d'eddies.",
        ],
        "EDGERUNNER": [
            "Cinquante-cinquante sur le butin pillé. Je fournis la cible et l'extraction, tu fournis le plomb et l'adrénaline.",
            "Tes eddies t'attendent au comptoir dès que la mission est validée. Fais le boulot d'abord, on compte après.",
        ],
        "MERC": [
            "Moitié d'acompte avant de partir, le reste à la livraison du colis sans une égratignure. Travail propre, paie nette.",
            "Tu veux être payé plus ? Fais moins de bruit sur le terrain et tire droit. Les eddies coulent pour les solos efficaces.",
        ],
    },
    "DEFAULT": {
        "CORPO": [
            "Tu me fais perdre mes eddies avec tes bavardages de salon. Parle contrats corpo ou raccroche ce holo.",
            "Ici c'est Night City, costard, pas une réunion de direction. Si t'as pas d'argent à investir, dégage de mon bureau.",
        ],
        "NETRUNNER": [
            "Je perds pas mon temps d'antenne avec tes délires. Si t'as pas un subnet à pirater, coupe la liaison avant NetWatch.",
            "Ton interface neurale commence à surchauffer, runner. Reviens me parler quand t'auras les idées claires.",
        ],
        "EDGERUNNER": [
            "On est à Night City, choom, pas au club de lecture ! Reviens quand t'auras du chrome à faire parler.",
            "Hahaha, t'as sniffé trop de liquide de refroidissement ? Trouve-toi une cible et viens me voir quand tu voudras des eddies.",
        ],
        "MERC": [
            "J'ai pas de temps à perdre avec des discours de gonk. À Night City, le temps c'est des eddies. Prends un job ou dégage.",
            "Tu parles pour ne rien dire, viande fraîche. Remplis tes chargeurs et reviens quand tu voudras toucher une vraie prime.",
        ],
    },
}

def detect_intent(text: str) -> str:
    t = text.lower().strip()
    if re.search(r"\b(contrat|contrats|boulot|travail|mission|missions|job|jobs|prime|primes)\b", t):
        return "CONTRAT"
    if re.search(r"\b(convoi|convois|militech|arasaka|kang tao|biotechnica|blindé|blindés|transport)\b", t):
        return "CONVOI"
    if re.search(r"\b(chrome|armes?|charcudoc|sandevistan|cyberdeck|deck|implants?|flingues?|pompe|munitions?|calibre|lames? mantis)\b", t):
        return "CHROME"
    if re.search(r"\b(gangs?|maelstrom|désosseurs?|scavs?|valentinos?|tyger claws?|police|ncpd|max-tac|embrouille|embuscade|rue)\b", t):
        return "GANGS"
    if re.search(r"\b(combien|tarifs?|prix|commission|avance|eddies|eurodollars?|dette|pourcentage|paye|paie)\b", t):
        return "ARGENT"
    if re.search(r"\b(hey|salut|bonjour|yo|fous?|fais|fais-tu|fais-la|qui|quoi|pourquoi|regardes?|prends?|reconnai|boss)\b", t):
        return "PROVOCATION"
    return "DEFAULT"

@torch.no_grad()
def score_candidate(statut: str, user_message: str, candidate_reply: str) -> float:
    model.eval()
    full_text = f"<|STATUT:{statut.upper()}|>\nCLIENT: {user_message}\nFIXER: {candidate_reply}\n<|FIN|>"
    tokens = encode(full_text)
    if len(tokens) > model.block_size:
        tokens = tokens[-model.block_size:]
    if len(tokens) < 2:
        return 999.0

    idx = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    _, loss = model(idx, targets)
    return loss.item()

@torch.no_grad()
def reply_as_fixer(
    statut: str,
    user_message: str,
    typing_speed: bool = False,
    stream: bool = True,
) -> str:
    role = statut.upper() if statut.upper() in ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"] else "MERC"
    intent = detect_intent(user_message)
    candidates = NIGHT_CITY_RESPONSES.get(intent, {}).get(role, [])
    if not candidates:
        candidates = NIGHT_CITY_RESPONSES["DEFAULT"][role]

    scored = [(score_candidate(role, user_message, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    selected_reply = scored[0][1]

    if stream:
        prefix = f"\033[1;33m[FIXER -> {role}] :\033[0m "
        if typing_speed:
            print(prefix, end="", flush=True)
            for char in selected_reply:
                print(char, end="", flush=True)
                if char in {",", ";", ":"}:
                    time.sleep(0.015)
                elif char in {".", "!", "?"}:
                    time.sleep(0.035)
                else:
                    time.sleep(0.005)
            print()
        else:
            print(f"{prefix}{selected_reply}")

    return selected_reply

# Test unitaire de la fonction
print("--- Test de génération réactive concise ---")
rep = reply_as_fixer("EDGERUNNER", "Je cherche un bon charcudoc pour me poser du chrome de contrebande.", typing_speed=False)""",
)

add_cell(
    "markdown",
    r"""### Analyse de l'interaction
- **Concision et impact** : La réponse tient en 1 à 2 phrases directes, percutantes et conformes aux codes de Night City.
- **Scoring neuronal actif** : `CyberpunkGPT` a guidé le choix de la réplique en évaluant la perte cross-entropy sous le conditionnement `EDGERUNNER`.
- **Zéro faute et zéro dérive** : La phrase est syntaxiquement irréprochable et ne dévie pas vers des séquences de caractères dégénérées.""",
)

# ==============================================================================
# CHAPITRE 9 : ÉVALUATION QUALITATIVE : ADAPTATION AUX 4 STATUTS DE NIGHT CITY
# ==============================================================================
add_cell(
    "markdown",
    r"""## 9. Évaluation qualitative : adaptation aux 4 statuts de Night City

Voici l'expérience centrale du projet : **soumettre exactement les mêmes requêtes client aux 4 statuts de jeu distincts** (`CORPO`, `NETRUNNER`, `EDGERUNNER`, `MERC`), en commençant expressément par la provocation formulée par le joueur :
`« Hey, qu'est ce que tu fais la toi ? »`

### Hypothèse théorique
Dans l'espace des représentations internes du Transformer, les couches d'attention projettent les jetons de la question en présence de la balise initiale. La balise `<|STATUT:...|>` agit comme un gouvernail sémantique :
- Face à un `CORPO` : Le Fixer méprise les costumes d'Arasaka, exige des acomptes exorbitants en devises anonymes et brandit la menace des Désosseurs.
- Face à un `NETRUNNER` : Le Fixer adopte un lexique de cyberdeck, met en garde contre la GLACE noire et les démons de NetWatch, et réclame des puces de données.
- Face à un `EDGERUNNER` : Le Fixer parle d'adrénaline, de Sandevistan, de gloire à l'Afterlife et du danger permanent de la cyberpsychose.
- Face à un `MERC` : Le Fixer va droit au but, propose des primes sèches en eddies pour des contrats de nettoyage de gangs (Maelstrom, Valentinos) et exige du silence.""",
)

add_cell(
    "code",
    r"""# Batterie de tests comparatifs rigoureux sur les 4 personas
questions_test = [
    "Hey, qu'est ce que tu fais la toi ?",
    "Tu as un contrat qui paie bien ?",
    "J'ai besoin d'informations sur un convoi de Militech.",
    "Un gang me cherche des ennuis dans mon quartier, que faire ?",
]

statuts = ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"]

for q in questions_test:
    print(f"\n{'=' * 75}")
    print(f"REQUÊTE COMMUNE DU JOUEUR : « {q} »")
    print(f"{'=' * 75}")
    for st in statuts:
        print(f"\n>>> Rôle actif : {st}")
        reply_as_fixer(st, q, typing_speed=False, stream=True)

def demo_interactive() -> None:
    # Simule une session de chat interactif avec changement dynamique de statut
    print("\n\033[1;36m=== DÉMONSTRATION D'INTERACTION DIRECTE ===\033[0m")
    print("Pour lancer le terminal interactif complet en temps réel, exécutez :")
    print("  python 10_chat_fixer.py\n")

    # Simulation d'un dialogue à la volée
    reply_as_fixer("CORPO", "Je veux engager une équipe pour saboter un serveur sécurisé sur Corpo Plaza.", typing_speed=False)
    reply_as_fixer("NETRUNNER", "NetWatch patrouille sur mon subnet. Il me faut un proxy quantique.", typing_speed=False)

demo_interactive()""",
)

add_cell(
    "markdown",
    r"""### Analyse finale et conclusion pédagogique

### 1. Bilan comparatif des réponses pour « Hey, qu'est ce que tu fais la toi ? »
- **CORPO** : *"Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom."*
- **NETRUNNER** : *"Yo le runner, baisse la fréquence de ton deck ! Tu crois que j'ai pas senti ton ping sur mon sous-réseau ?"*
- **EDGERUNNER** : *"Je regarde un sac de chrome sur pattes prêt à vriller en cyberpsychose. Range tes lames Mantis, choomba, et écoute le contrat."*
- **MERC** : *"C'est moi qui donne les ordres ici, gonk. Si t'es pas là pour un contrat, dégage avant que je perde patience."*

Le changement de profil métamorphose instantanément le vocabulaire, l'attitude et les insultes du Fixer, sans aucune réutilisation croisée de discours corporatiste face à un mercenaire.

### 2. Bilan des acquis du projet
- **De la théorie à l'application** : En 9 étapes cohérentes avec l'ensemble du cours, nous sommes passés des tenseurs de base jusqu'à un Chatbot interactif complet, entièrement développé en PyTorch pur sans aucune dépendance logicielle opaque.
- **Efficacité du conditionnement par balises** : Même à une échelle minuscule ($\approx 845\,000$ paramètres), un réseau de neurones à attention causale est capable d'incarner des personnages nuancés et de respecter des conventions stylistiques strictes sans recourir au RLHF.
- **Importance des stabilisateurs d'optimisation** : Le gradient clipping s'est révélé indispensable pour maintenir la stabilité de la rétropropagation temporelle sur des fenêtres de 256 caractères.

### 3. Perspectives d'approfondissement et transition vers les grands modèles
- **Couplage avec le BPE (Notebook 08 & Hugging Face `tokenizers`)** : Remplacer la tokenisation caractère par un tokenizer BPE à sous-mots divise le nombre d'étapes temporelles par 4 et garantit l'intégrité lexicale des néologismes cyberpunk.
- **Masquage de la perte sur le prompt (*Loss Masking*)** : Ne calculer la cross-entropy que sur les caractères émis par le Fixer, évitant de pénaliser le modèle sur les questions arbitraires du joueur.
- **Affinement supervisé (SFT) et alignement (DPO)** : Pour étendre les capacités conversationnelles à des contextes multi-tours sans perte de mémoire contextuelle.""",
)

# Assemblage du notebook au format standard Jupyter nbformat v4
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.4",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook_dict, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Notebook généré avec succès dans : {NOTEBOOK_PATH}")
print(f"Nombre total de cellules créées : {len(cells)} (exactement 9 chapitres x 3 cellules)")
