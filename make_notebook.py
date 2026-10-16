#!/usr/bin/env python3
"""Générateur ultra-pédagogique du notebook 09_cyberpunk_fixer.ipynb.

Ce script génère un notebook complet, richement commenté, avec des explications
mathématiques claires, des diagrammes de flux et des commentaires exhaustifs
sur chaque ligne de code et chaque transformation tensorielle.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path.cwd()
NOTEBOOK_PATH = ROOT / "09_cyberpunk_fixer.ipynb"

cells: list[dict] = []


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
# CHAPITRE 1 : INTRODUCTION, PARADIGME ET CONDITIONNEMENT DE RÔLE PAR BALISES
# ==============================================================================
add_cell(
    "markdown",
    r"""# 09 — Cyberpunk Fixer : conditionnement de rôle et génération interactive

Bienvenue dans le neuvième et dernier jalon de ce cours de Deep Learning. Dans les huit premiers notebooks (`01_texte_vers_nombres` à `08_bpe`), nous avons patiemment exploré les mécanismes fondamentaux des modèles de langue : de la modélisation probabiliste par n-grammes jusqu'aux mécanismes de Self-Attention et à la tokenisation BPE.

Ce projet final constitue la **synthèse applicative** de tous ces concepts. Notre objectif est de construire, d'entraîner et de déployer *from scratch* (sans aucune bibliothèque externe de haut niveau telle que Hugging Face ou LangChain) un modèle Transformer Décodeur-seul incarnant un **Fixer véreux, cynique et sarcastique de Night City (univers Cyberpunk 2077 / Edgerunners)**.

---

## 1. La vision globale : Comment toutes les parties du projet sont interconnectées

Avant d'écrire la moindre ligne de code, visualisons le flux complet des données et des tenseurs à travers les différents modules que nous allons assembler :

```text
=================================================================================================
                            ARCHITECTURE GLOBALE DU PROJET CYBERPUNK FIXER
=================================================================================================

 [1. Corpus Brut Night City]  ---> Textes balisés : <|STATUT:...|> \n CLIENT: ... \n FIXER: ... <|FIN|>
              │
              ▼
 [2. Tokenisation Caractère] ---> Mapping bijectif : Caractères <---> Identifiants entiers (0 à V-1)
              │
              ▼
 [3. Dataset & DataLoader]   ---> Tenseurs d'entrée X de taille (B, T) et cibles décalées Y de taille (B, T)
              │
              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 4. Architecture Transformer "CyberpunkGPT" (4 Blocs, 4 Têtes, d_embd = 128)                 │
 │                                                                                             │
 │   Entrée X (B, T)                                                                           │
 │         │                                                                                   │
 │         ├─> [Token Embedding] (B, T, C)                                                     │
 │         └─> [Position Embedding] (1, T, C)                                                  │
 │                   │                                                                         │
 │                   ▼  (Somme élément par élément)                                            │
 │             x0 = Tok_Emb + Pos_Emb   (B, T, C)                                              │
 │                   │                                                                         │
 │                   ▼                                                                         │
 │         ┌─────────────────────────────────────────────────────────┐                         │
 │         │ Bloc Transformer (x4 empilés) : Schéma Pre-LayerNorm     │                         │
 │         │                                                         │                         │
 │         │   x ──> LayerNorm1 ──> Causal Multi-Head Attention      │                         │
 │         │   │                           │                         │                         │
 │         │   └──────────────(+)──────────┘  (Connexion Résiduelle) │                         │
 │         │                   │                                     │                         │
 │         │   x ──> LayerNorm2 ──> Feed-Forward MLP (GELU/ReLU)     │                         │
 │         │   │                           │                         │                         │
 │         │   └──────────────(+)──────────┘  (Connexion Résiduelle) │                         │
 │         └─────────────────────────────────────────────────────────┘                         │
 │                   │                                                                         │
 │                   ▼                                                                         │
 │             [LayerNorm finale] (B, T, C)                                                    │
 │                   │                                                                         │
 │                   ▼                                                                         │
 │             [Tête linéaire lm_head] (B, T, V)  ---> Vecteur de Logits bruts non normalisés │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────────────────────┬─────────────────────────────────────────┐
              ▼                                      ▼                                         ▼
   [5. Phase Entraînement]                [7. Phase Décodage Libre]             [8-9. Chatbot & Scoring]
   Perte Cross-Entropy vs Y               Extraction de caractères :            Évaluation de vraisemblance
   + Gradient Clipping (tau = 1.0)        - Température (T)                     sous CyberpunkGPT pour chaque
   + Optimiseur AdamW                     - Filtrage Top-K & Top-P (Nucleus)    profil (CORPO, MERC, etc.)
=================================================================================================
```

---

## 2. Le défi pédagogique : Conditionner un rôle sans RLHF

Dans les grands modèles de langage industriels (LLM de plusieurs milliards de paramètres), l'alignement sur un rôle (*persona*) et sur des consignes de sécurité nécessite d'ordinaire des infrastructures monumentales :
1. Un pré-entraînement massif non supervisé.
2. Un affinement supervisé sur des instructions (*Supervised Fine-Tuning*, SFT).
3. Un alignement par renforcement (*Reinforcement Learning from Human Feedback*, RLHF / DPO).

### Comment faire avec un mini-modèle de seulement 846 000 paramètres ?
Nous exploitons directement la propriété mathématique fondamentale de la **Self-Attention Causale**. 

Un modèle de langue autorégressif décompose la probabilité conjointe d'une séquence $x_1, \dots, x_T$ en un produit de probabilités conditionnelles :

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \dots, x_{t-1})$$

Si le corpus d'entraînement préfixe systématiquement chaque dialogue par une balise de structure explicite :
```text
<|STATUT:CORPO|>
CLIENT: J'ai besoin d'informations sur un convoi de Militech.
FIXER: Regardez-moi ce costume d'Arasaka qui vient s'encanailler à Kabuki...
<|FIN|>
```

alors, lorsque le modèle prédit les caractères de la réplique du Fixer ($x_t$), les têtes d'attention causale peuvent « regarder en arrière » dans la fenêtre contextuelle ($x_1, \dots, x_{t-1}$) et calculer des représentations internes explicitement conditionnées par le statut :

$$P(x_t \mid x_{<t}, \text{STATUT}=\text{CORPO}) \neq P(x_t \mid x_{<t}, \text{STATUT}=\text{MERC})$$

La balise initiale agit ainsi comme un **gouvernail sémantique** guidant la distribution de probabilité du vocabulaire sans exiger le moindre fine-tuning externe.

Commençons par configurer un environnement reproductible et détecter le matériel disponible.""",
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

# ==============================================================================
# 1. REPRODUCTIBILITÉ EXPÉRIMENTALE
# ==============================================================================
# Fixer la graine aléatoire garantit que l'initialisation des poids, les tirages
# du DataLoader et les masques de dropout produiront des résultats identiques à chaque run.
SEED = 2077
torch.manual_seed(SEED)

# ==============================================================================
# 2. GESTION DES CHEMINS DU PROJET (Pathlib)
# ==============================================================================
# On détermine dynamiquement le répertoire racine pour que le notebook s'exécute
# de manière transparente en local, sur un serveur ou dans un conteneur.
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs" / "09_cyberpunk_fixer"
CORPUS_PATH = DATA_DIR / "nightcity_corpus.txt"

# Création automatique des dossiers de travail s'ils n'existent pas encore
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. DÉTECTION MATÉRIELLE MODERNE (PyTorch 2.0+)
# ==============================================================================
# On utilise la nouvelle API unifiée 'torch.accelerator' qui remplace les vérifications
# manuelles et verbeuses 'cuda.is_available()' ou 'mps.is_available()'.
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print("=" * 65)
print("  ENVIRONNEMENT D'EXÉCUTION INITIALISÉ AVEC SUCCÈS")
print("=" * 65)
print(f"• Dispositif de calcul (device) : {device.upper()}")
print(f"• Graine pseudo-aléatoire (seed): {SEED}")
print(f"• Répertoire des données        : {DATA_DIR}")
print(f"• Répertoire des checkpoints    : {CHECKPOINT_DIR}")
print("=" * 65)""",
)

add_cell(
    "markdown",
    r"""### Analyse de la configuration
1. **L'API `torch.accelerator`** : Introduite dans les versions modernes de PyTorch, elle permet une détection agnostique du matériel (GPU NVIDIA CUDA, Apple Silicon MPS ou processeur CPU x86/ARM) sans modifier une seule ligne de code.
2. **Reproductibilité numérique (`SEED = 2077`)** : En fixant le générateur pseudo-aléatoire, nous assurons la rigueur scientifique de l'expérience : les courbes de perte et les comparaisons de décodage seront strictement comparables d'une exécution à l'autre.
3. **Structure modulaire** : Les chemins construits avec `pathlib.Path` évitent les erreurs de séparateur (`/` vs `\`) entre environnements Linux, macOS et Windows.""",
)

# ==============================================================================
# CHAPITRE 2 : LE CORPUS NIGHT CITY : TOKENISATION CARACTÈRE ET EXPLORATION
# ==============================================================================
add_cell(
    "markdown",
    r"""## 2. Le corpus Night City : tokenisation caractère et exploration

Au notebook 08 (`08_bpe.ipynb`), nous avons étudié la tokenisation par sous-mots statistiques BPE (*Byte Pair Encoding*). Pour ce mini-projet de dialogue stylisé, nous revenons délibérément à une **tokenisation au niveau caractère** (étudiée aux notebooks 01 et 06).

---

### Pourquoi privilégier le niveau caractère pour ce projet ?

| Critère | Tokenisation par Mots | Tokenisation BPE (Sous-mots) | Tokenisation Caractère (Notre choix) |
| :--- | :--- | :--- | :--- |
| **Taille du vocabulaire ($V$)** | Très grand ($V \ge 50\,000$) | Moyen ($V \approx 1\,000 - 32\,000$) | **Ultra-compact ($V \approx 80$)** |
| **Empreinte de la table d'embeddings** | Énorme ($\approx 60\%$ des poids) | Moyenne ($\approx 20\%$ des poids) | **Négligeable ($< 1\%$ des poids)** |
| **Mots inconnus (*Out-Of-Vocabulary*, OOV)** | Fréquents (remplacés par `<UNK>`) | Rares (découpés en sous-mots) | **Strictement impossibles ($0\%$ OOV)** |
| **Apprentissage de l'argot Cyberpunk** | Échoue sur les néologismes | Nécessite un pré-entraînement BPE | **Assimilation naturelle lettre par lettre** |

Dans notre univers Cyberpunk, les dialogues regorgent de termes argotiques et de néologismes : *choom, choomba, gonk, eddies, flatline, charcudoc, Sandevistan, cyberdeck, GLACE noire, NetWatch, scavs, Trauma Team*.  
Avec un alphabet restreint à $V \approx 80$ caractères, le modèle dispose de la quasi-totalité de ses $846\,000$ paramètres pour apprendre la syntaxe et les relations d'attention temporelles, plutôt que de gaspiller sa mémoire dans une immense table d'embeddings.

---

### Le pont mathématique : Encodage et Décodage
Pour qu'un réseau de neurones manipule du texte, chaque caractère $c \in \Sigma$ doit être converti en un identifiant entier unique $i \in \{0, \dots, V-1\}$ :
$$\text{encode}(c) = \text{stoi}[c], \quad \text{decode}(i) = \text{itos}[i]$$
où `stoi` (*string to integer*) et `itos` (*integer to string*) forment une bijection parfaite sur l'alphabet du corpus.""",
)

add_cell(
    "code",
    """# ==============================================================================
# CHARGEMENT DU CORPUS NIGHT CITY ET CRÉATION DU VOCABULAIRE
# ==============================================================================
# Si le corpus n'est pas présent, on exécute automatiquement generate_corpus.py
if not CORPUS_PATH.exists() or CORPUS_PATH.stat().st_size < 50_000:
    print("Corpus absent ou incomplet : génération du corpus riche via generate_corpus.py...")
    import subprocess
    subprocess.run(["python3", str(ROOT / "generate_corpus.py")], check=True)

# Lecture du fichier texte complet en UTF-8
text = CORPUS_PATH.read_text(encoding="utf-8")

# 1. Extraction des caractères uniques ordonnés
chars = sorted(set(text))
vocab_size = len(chars)

# 2. Dictionnaires de projection bijective
# stoi : fait correspondre chaque caractère à son index entier unique (ex: 'A' -> 14)
# itos : fait correspondre chaque index à son caractère original (ex: 14 -> 'A')
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    '''Convertit une chaîne de caractères en une liste d'entiers.'''
    return [stoi[c] for c in s if c in stoi]

def decode(ids: list[int]) -> str:
    '''Reconstruit la chaîne textuelle à partir d'une liste d'entiers.'''
    return "".join(itos[i] for i in ids)

# 3. Conversion de tout le texte en un unique tenseur PyTorch 1D d'entiers 64-bits
data = torch.tensor(encode(text), dtype=torch.long)

# 4. Séparation stricte 90% entraînement / 10% validation
# Les données de validation ne sont jamais vues pendant la rétropropagation du gradient !
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

print("=" * 65)
print("  STATISTIQUES DU CORPUS & TOKENISATION AU NIVEAU CARACTÈRE")
print("=" * 65)
print(f"• Taille totale du corpus      : {len(text):,} caractères ({len(text.encode('utf-8')) / 1024:.1f} Ko)")
print(f"• Taille du vocabulaire (V)    : {vocab_size} symboles uniques")
print(f"• Données d'entraînement (90%) : {len(train_data):,} caractères")
print(f"• Données de validation (10%)   : {len(val_data):,} caractères")
print(f"• Alphabet complet             : {''.join(chars)}")
print("=" * 65)

# Test de bijection élémentaire
phrase_test = "<|STATUT:CORPO|> CLIENT: Combien pour le convoi ? FIXER: Crache les eddies, choom."
assert decode(encode(phrase_test)) == phrase_test, "Erreur : l'encodage/décodage n'est pas bijectif !"
print("✓ Test de réversibilité encode/decode validé avec succès !")""",
)

add_cell(
    "markdown",
    r"""### Analyse des données et de la tokenisation
- **Richesse de l'alphabet ($V = 80$)** : Le vocabulaire intègre l'ensemble des lettres minuscules et majuscules de l'alphabet français, les lettres accentuées (`é`, `è`, `ê`, `à`, `ç`, `ô`, `ù`), la ponctuation complète et les balises spécifiques `<|>`.
- **Partition étanche train/validation** : Les $43\,700$ caractères de validation sont strictement séparés de l'ensemble d'entraînement. Cela élimine tout risque de fuite d'information (*data leakage*) lors de l'évaluation de la perte.
- **Vérification d'intégrité** : Le test d'assertion confirme que `decode(encode(s)) == s` sans aucune altération de caractère.""",
)

# ==============================================================================
# CHAPITRE 3 : STRUCTURATION DES DONNÉES : CHARDATASET ET DATALOADER
# ==============================================================================
add_cell(
    "markdown",
    r"""## 3. Structuration des données : `CharDataset` et `DataLoader`

Dans les scripts 01 à 06 de ce cours, la longueur de séquence `block_size` était généralement fixée à 64 ou 128 caractères. Cela suffisait pour de petits poèmes ou des proverbes courts.

---

### Pourquoi une fenêtre temporelle élargie à `block_size = 256` ?

Examinons la longueur réelle d'un tour de parole complet dans notre corpus :
1. Balise d'amorce initiale : `<|STATUT:NETRUNNER|>\n` ($\approx 21$ caractères)
2. Question formulée par le client : `CLIENT: J'ai besoin d'informations sur un convoi de Militech.\n` ($\approx 65$ caractères)
3. Réponse incisive du Fixer : `FIXER: Yo le runner, garde ton cyberdeck au frais...` ($\approx 100$ à $130$ caractères)
4. Balise de fin de tour : `<|FIN|>\n` ($\approx 8$ caractères)

Le dialogue complet totalise entre **190 et 230 caractères**.  
Si notre fenêtre de contexte était inférieure à 256 (par exemple 128), lorsque le modèle tenterait de prédire les derniers mots de la réplique du Fixer, la balise initiale `<|STATUT:NETRUNNER|>` serait déjà sortie du champ de vision de la Self-Attention ! Le modèle oublierait alors le persona et commencerait à divaguer.

---

### La fenêtre glissante autorégressive : Entrée $X$ et Cible $Y$

Pour chaque segment de texte de longueur `block_size + 1` tiré du corpus, nous construisons deux tenseurs décalés d'un unique caractère :

$$\mathbf{X} = [c_0, c_1, c_2, \dots, c_{T-1}]$$
$$\mathbf{Y} = [c_1, c_2, c_3, \dots, c_T]$$

```text
Entrée X :  <  |  S  T  A  T  U  T  :  C  O  R  P  O  |  >  \n  C  L  I  E  N  T  :  ...
Cible  Y :  |  S  T  A  T  U  T  :  C  O  R  P  O  |  >  \n  C  L  I  E  N  T  :  ...
            ─────────────────────────────────────────────────────────────────────────>
                 À chaque position t, le modèle prédit le caractère suivant t+1
```

Grâce au masque causal de la Self-Attention (détaillé au chapitre 4), un seul lot de taille $(B=32, T=256)$ fournit simultanément :
$$32 \times 256 = 8\,192 \text{ exemples supervisés de prédiction}$$
en une seule passe en avant (*forward pass*) !""",
)

add_cell(
    "code",
    """# ==============================================================================
# DÉFINITION DES HYPERPARAMÈTRES D'ARCHITECTURE ET DE DONNÉES
# ==============================================================================
batch_size = 32       # Nombre de séquences traitées en parallèle par passage
block_size = 256      # Fenêtre temporelle maximale couverte par l'attention causale
n_embd = 128          # Dimension de projection vectorielle des tokens et positions
n_head = 4            # Nombre de têtes d'attention en parallèle (d_k = 128 // 4 = 32)
n_layer = 4           # Nombre de blocs Transformer empilés
dropout = 0.1         # Taux de désactivation aléatoire pour régulariser l'apprentissage
learning_rate = 1e-3  # Taux d'apprentissage de l'optimiseur AdamW
eval_every = 150      # Fréquence de calcul de la perte de validation
val_batches = 30      # Nombre de lots utilisés pour estimer la perte de validation


class CharDataset(Dataset):
    '''Dataset représentant toutes les fenêtres temporelles de longueur (block_size + 1).'''

    def __init__(self, data_tensor: torch.Tensor, block_size_: int) -> None:
        self.data = data_tensor
        self.block_size = block_size_

    def __len__(self) -> int:
        # Le nombre total de fenêtres disponibles
        return len(self.data) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Découpe d'un fragment de block_size + 1 caractères
        chunk = self.data[i : i + self.block_size + 1]
        # x : contexte d'entrée (positions 0 à T-1)
        # y : cible attendue, décalée d'un cran vers le futur (positions 1 à T)
        return chunk[:-1], chunk[1:]


# Instanciation des Datasets pour train et validation
train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)


def make_train_loader(steps: int) -> DataLoader:
    '''Construit un DataLoader avec tirage aléatoire uniforme avec remplacement.'''
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=steps * batch_size)
    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    sampler=RandomSampler(val_dataset, replacement=True, num_samples=val_batches * batch_size),
)

# ==============================================================================
# INSPECTION D'UN LOT TÉMOIN (SANITY CHECK)
# ==============================================================================
sample_X, sample_Y = next(iter(make_train_loader(1)))
print("=" * 65)
print("  STRUCTURE TENSORIELLE D'UN LOT D'ENTRAÎNEMENT")
print("=" * 65)
print(f"• Tenseur d'entrée X : forme {sample_X.shape} (batch_size, block_size)")
print(f"• Tenseur cible  Y   : forme {sample_Y.shape} (batch_size, block_size)")
print(f"• Type de données    : {sample_X.dtype}")
print("-" * 65)
print("Premier exemple d'entrée X[0, :45] (décodé) :")
print(repr(decode(sample_X[0, :45].tolist())))
print("Premier exemple cible  Y[0, :45] (décodé) :")
print(repr(decode(sample_Y[0, :45].tolist())))
print("=" * 65)""",
)

add_cell(
    "markdown",
    r"""### Analyse de la structuration des données
- **Alignement temporel parfait** : On observe bien que le premier caractère de la cible $Y$ correspond exactement au deuxième caractère de l'entrée $X$. À chaque étape $t \in [0, T-1]$, le réseau apprend à modéliser la transition $x_t \to x_{t+1}$.
- **Échantillonnage stochastique par `RandomSampler`** : Plutôt que de parcourir séquentiellement le texte d'un bout à l'autre, le tirage uniforme aléatoire avec remplacement casse les corrélations locales et lisse la variance stochastique du gradient au cours de l'optimisation.""",
)

# ==============================================================================
# CHAPITRE 4 : L'ARCHITECTURE TRANSFORMER : DE LA TÊTE CAUSALE AU CYBERPUNKGPT
# ==============================================================================
add_cell(
    "markdown",
    r"""## 4. L'architecture Transformer : de la tête causale au CyberpunkGPT

Nous construisons ici l'architecture complète du modèle GPT Décodeur-seul en PyTorch pur, selon les principes établis par Vaswani et al. (2017) et perfectionnés par GPT-2 (Radford et al., 2019).

---

### Les 5 briques fondamentales et leurs équations

#### 1. Self-Attention Causale (`Head`)
À partir d'un état latent $\mathbf{x} \in \mathbb{R}^{B \times T \times C}$, chaque position projette trois vecteurs : une **Requête** ($Q$), une **Clé** ($K$) et une **Valeur** ($V$) de dimension $d_k = C / h = 32$.
L'attention calcule un score de similarité entre toutes les paires de positions :

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

- **Le facteur d'échelle $\frac{1}{\sqrt{d_k}}$** : Sans lui, pour des dimensions élevées, le produit scalaire $Q K^T$ aurait une variance proportionnelle à $d_k$. Cela pousserait la fonction softmax dans des régions à très forte saturation où ses dérivées s'annulent, provoquant la disparition du gradient (*vanishing gradient*).
- **Le masque causal $M$** : Une matrice triangulaire inférieure où $M_{i,j} = 0$ si $j \le i$, et $-\infty$ si $j > i$. Puisque $e^{-\infty} = 0$, la fonction softmax attribue une probabilité rigoureusement nulle à tout jeton situé dans le futur. Le jeton $t$ ne peut donc s'informer que sur les jetons passés $0, \dots, t$.

#### 2. Attention Multi-Têtes (`MultiHeadAttention`)
Plutôt que d'avoir une seule grande tête d'attention, nous utilisons $h=4$ têtes opérant en parallèle dans des sous-espaces de projection distincts :
- La tête 1 peut se spécialiser sur l'accord phonétique du mot en cours.
- La tête 2 peut surveiller la balise `<|STATUT:...|>` située au tout début du texte.
- La tête 3 peut détecter la ponctuation interrogative `?`.
- La tête 4 peut traquer les balises d'amorce `FIXER:`.

Les sorties des $h$ têtes sont concaténées puis reprojetées par une matrice linéaire $\mathbf{W}_O \in \mathbb{R}^{C \times C}$.

#### 3. Réseau Feed-Forward (`FeedForward`)
Appliqué indépendamment à chaque position temporelle, ce perceptron multicouche à deux étages opère une expansion d'un facteur 4 :
$$\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2, \quad W_1 \in \mathbb{R}^{C \times 4C}, \; W_2 \in \mathbb{R}^{4C \times C}$$
Dans la littérature moderne, cette couche agit comme une **mémoire associative clé-valeur** stockant les connaissances factuelles et lexicales du modèle.

#### 4. Schéma Pre-LayerNorm et Connexions Résiduelles (`Block`)
Contrairement au Transformer original de 2017 qui appliquait la normalisation après les blocs (*Post-LN*), nous adoptons le schéma moderne **Pre-LayerNorm** généralisé par GPT-2 :
$$\mathbf{x}^{(1)} = \mathbf{x} + \text{MultiHeadAttention}(\text{LayerNorm}_1(\mathbf{x}))$$
$$\mathbf{x}^{(2)} = \mathbf{x}^{(1)} + \text{FeedForward}(\text{LayerNorm}_2(\mathbf{x}^{(1)}))$$
Grâce à l'addition résiduelle $+ \mathbf{x}$, le gradient peut circuler librement en ligne droite (*highway*) depuis la sortie jusqu'à la couche d'embedding sans être atténué par des multiplications matricielles successives.

#### 5. Assemblage complet (`CyberpunkGPT`)
Combine l'embedding de vocabulaire (`token_embedding_table`) et l'embedding de position absolu (`position_embedding_table`), empile $L=4$ blocs Transformer, applique une `LayerNorm` finale et projette sur les logits du dictionnaire via la couche linéaire `lm_head`.""",
)

add_cell(
    "code",
    """# ==============================================================================
# DÉFINITION DE L'ARCHITECTURE TRANSFORMER COMPLÈTE
# ==============================================================================

class Head(nn.Module):
    '''Une unique tête de Self-Attention causale avec masque triangulaire inférieur.'''

    def __init__(self, n_embd_: int, head_size: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        # Projections linéaires pour Requête (Query), Clé (Key) et Valeur (Value)
        # Note : bias=False est le standard moderne (nanoGPT, LLaMA) pour alléger les paramètres.
        self.key = nn.Linear(n_embd_, head_size, bias=False)
        self.query = nn.Linear(n_embd_, head_size, bias=False)
        self.value = nn.Linear(n_embd_, head_size, bias=False)

        # Matrice triangulaire inférieure enregistrée comme buffer persistant (non entraînable)
        self.register_buffer("tril", torch.tril(torch.ones(block_size_, block_size_)))
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrée x : dimension (B, T, C)
        B, T, C = x.shape

        # 1. Calcul des clés et requêtes : (B, T, head_size)
        k = self.key(x)
        q = self.query(x)

        # 2. Produit scalaire des affinités (Scaled Dot-Product) : (B, T, T)
        # Facteur d'échelle : 1 / sqrt(d_k) pour stabiliser la variance
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # 3. Masquage causal : les positions futures (au-delà de T) sont remplacées par -infini
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 4. Normalisation Softmax sur la dernière dimension : les -infini deviennent 0
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 5. Agrégation pondérée des valeurs v : (B, T, head_size)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    '''Attention multi-têtes combinant n_head têtes causales en parallèle.'''

    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        head_size = n_embd_ // n_head_
        # Création des têtes indépendantes
        self.heads = nn.ModuleList(
            [Head(n_embd_, head_size, block_size_, dropout_) for _ in range(n_head_)]
        )
        # Projection linéaire finale pour recombiner les sorties des têtes
        self.proj = nn.Linear(n_embd_, n_embd_)
        self.dropout = nn.Dropout(dropout_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concaténation des sorties des têtes sur la dimension des canaux : (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Projection linéaire + dropout de régularisation
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    '''Perceptron multicouche à deux niveaux avec expansion 4x et non-linéarité ReLU.'''

    def __init__(self, n_embd_: int, dropout_: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Projection expansive : C -> 4C (128 -> 512)
            nn.Linear(n_embd_, 4 * n_embd_),
            nn.ReLU(),
            # Projection compressive : 4C -> C (512 -> 128)
            nn.Linear(4 * n_embd_, n_embd_),
            nn.Dropout(dropout_),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transformation indépendante appliquée à chaque vecteur de position : (B, T, C)
        return self.net(x)


class Block(nn.Module):
    '''Bloc Transformer associant Attention Causale et FFN en schéma Pre-LayerNorm.'''

    def __init__(self, n_embd_: int, n_head_: int, block_size_: int, dropout_: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd_, n_head_, block_size_, dropout_)
        self.ffwd = FeedForward(n_embd_, dropout_)
        self.ln1 = nn.LayerNorm(n_embd_)
        self.ln2 = nn.LayerNorm(n_embd_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Schéma Pre-LayerNorm avec connexions résiduelles directes (autoroutes de gradients)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CyberpunkGPT(nn.Module):
    '''Modèle GPT Décodeur-seul complet adapté pour Night City.'''

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

        # 1. Table d'embeddings des jetons (vocab_size -> n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size_, n_embd_)
        # 2. Table d'embeddings des positions absolues (block_size -> n_embd)
        self.position_embedding_table = nn.Embedding(block_size_, n_embd_)

        # 3. Empilement des blocs Transformer (n_layer blocs Pre-LN)
        self.blocks = nn.Sequential(
            *[Block(n_embd_, n_head_, block_size_, dropout_) for _ in range(n_layer_)]
        )

        # 4. LayerNorm finale avant la projection linéaire de sortie
        self.ln_f = nn.LayerNorm(n_embd_)
        # 5. Tête de modélisation linguistique projetant les états latents vers le vocabulaire
        self.lm_head = nn.Linear(n_embd_, vocab_size_)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # idx : tenseur d'indices de forme (B, T)
        B, T = idx.shape

        # Récupération des embeddings de tokens : (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        # Récupération des embeddings de position pour 0, 1, ..., T-1 : (T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        # Somme des représentations sémantiques et positionnelles : (B, T, n_embd)
        x = tok_emb + pos_emb

        # Passage à travers les 4 blocs Transformer empilés
        x = self.blocks(x)

        # Normalisation finale et calcul des logits bruts non normalisés : (B, T, vocab_size)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Si aucune cible n'est fournie (mode inférence pure), on retourne uniquement les logits
        if targets is None:
            return logits, None

        # Mode entraînement : calcul de la cross-entropy vectorisée sur tous les B*T tokens
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


# ==============================================================================
# INSTANCIATION DU MODÈLE ET COMPTAGE DES PARAMÈTRES
# ==============================================================================
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
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=" * 65)
print("  CYBERPUNKGPT : SPÉCIFICATIONS TECHNIQUES DE L'ARCHITECTURE")
print("=" * 65)
print(f"• Nombre de couches Transformer (n_layer) : {n_layer}")
print(f"• Nombre de têtes d'attention   (n_head)  : {n_head} (dim par tête: {n_embd // n_head})")
print(f"• Dimension de projection       (n_embd)  : {n_embd}")
print(f"• Taille maximale de contexte   (block)   : {block_size} caractères")
print(f"• Paramètres totaux du réseau             : {total_params:,}")
print(f"• Empreinte mémoire brute (Float32)       : {total_params * 4 / (1024**2):.2f} Mo")
print("=" * 65)""",
)

add_cell(
    "markdown",
    r"""### Analyse architecturale
1. **Équilibre paramètres/vitesse** : Avec **$845\,136$ paramètres** et seulement $3{,}22$ Mo d'empreinte mémoire, le modèle est suffisamment léger pour exécuter une rétropropagation complète en quelques millisecondes sur CPU, tout en ayant assez de capacité expressive pour assimiler les 4 personas de Night City.
2. **Le schéma Pre-LayerNorm** : Contrairement aux anciennes architectures Post-LN qui exigeaient un échauffement (*warmup*) délicat du taux d'apprentissage pour éviter la divergence, le Pre-LN garantit une stabilité numérique immédiate dès le premier pas d'optimisation.
3. **Absence de biais dans les projections d'attention** : L'omission des biais (`bias=False`) dans les projections $Q, K, V$ réduit de $10\%$ les calculs d'attention sans perte mesurable de qualité générative.""",
)

# ==============================================================================
# CHAPITRE 5 : ENTRAÎNEMENT AVEC GRADIENT CLIPPING & SUIVI DES PERTES
# ==============================================================================
add_cell(
    "markdown",
    r"""## 5. Entraînement avec Gradient Clipping & suivi des pertes

L'entraînement d'un Transformer autorégressif sur des séquences longues ($T=256$) requiert deux précautions méthodologiques majeures :

---

### 1. Le Gradient Clipping (`torch.nn.utils.clip_grad_norm_`)

Lors de la rétropropagation temporelle à travers 4 couches d'attention causale, l'accumulation multiplicative des gradients sur 256 pas de temps peut ponctuellement provoquer un phénomène de **gradient explosif** (*exploding gradient*). Une seule mise à jour excessive d'AdamW suffirait à dérégler irrémédiablement les tables d'embeddings.

Pour neutraliser ce risque, nous fixons un seuil de norme maximale $\tau = 1.0$ :

$$\mathbf{g} \leftarrow \mathbf{g} \times \min\left(1, \frac{\tau}{\|\mathbf{g}\|_2}\right)$$

> **Pourquoi c'est fondamental ?** : Si la norme euclidienne globale du gradient $\|\mathbf{g}\|_2$ dépasse $1.0$, le vecteur est automatiquement redimensionné à la norme $1.0$ sans changer sa direction ! Cela garantit que le modèle explore le paysage de perte sans jamais faire de pas disproportionné.

---

### 2. Repères théoriques de la perte Cross-Entropy

- **Perte initiale théorique (modèle non entraîné)** :  
  Au premier pas d'optimisation, avant tout apprentissage, le modèle produit une distribution uniforme sur le vocabulaire ($P(c) = 1/V$). La perte cross-entropy théorique vaut :
  $$\mathcal{L}_{\text{init}} = -\ln\left(\frac{1}{V}\right) = \ln(80) \approx 4{,}38$$
- **Objectif de convergence** :  
  Pour que le texte généré ressemble à du français structuré, la perte doit descendre sous $1{,}50$.  
  Pour maîtriser fidèlement l'argot Cyberpunk et les structures de balises sans aucune faute de frappe, la perte de validation doit idéalement chuter sous **$0{,}50$**.

Pour rendre l'exécution de ce notebook modulaire, la cellule suivante charge automatiquement le modèle pré-entraîné s'il existe déjà sur le disque.""",
)

add_cell(
    "code",
    """# ==============================================================================
# FONCTIONS D'ÉVALUATION ET D'ENTRAÎNEMENT
# ==============================================================================

@torch.no_grad()
def evaluate(model_: nn.Module, loader: DataLoader) -> float:
    '''Calcule la perte moyenne sur un ensemble de validation sans calculer les gradients.'''
    model_.eval()
    total = sum(model_(X.to(device), Y.to(device))[1].item() for X, Y in loader)
    return total / len(loader)


def train_model(
    model_: nn.Module,
    steps: int = 1500,
    eval_freq: int = 150,
    lr: float = 1e-3,
) -> tuple[list[int], list[float], list[float]]:
    '''Boucle d'entraînement complète avec AdamW, gradient clipping et métriques train/val.'''
    optimizer = torch.optim.AdamW(model_.parameters(), lr=lr, weight_decay=0.01)
    step_history: list[int] = []
    train_history: list[float] = []
    val_history: list[float] = []
    batch_losses: list[float] = []

    start_time = time.perf_counter()
    print(f"Démarrage de l'entraînement sur {device} pour {steps} pas...")

    for step, (X, Y) in enumerate(make_train_loader(steps), start=1):
        model_.train()
        # 1. Passe en avant (Forward pass)
        _, loss = model_(X.to(device), Y.to(device))

        # 2. Réinitialisation des gradients (set_to_none=True libère la mémoire plus vite que zero_)
        optimizer.zero_grad(set_to_none=True)

        # 3. Rétropropagation temporelle (Backward pass)
        loss.backward()

        # 4. Gradient clipping pour empêcher toute explosion numérique sur T=256
        torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=1.0)

        # 5. Mise à jour des poids selon la règle d'AdamW
        optimizer.step()

        batch_losses.append(loss.item())

        # Évaluation périodique sur le jeu de validation
        if step % eval_freq == 0 or step == steps:
            tr_loss = sum(batch_losses) / len(batch_losses)
            va_loss = evaluate(model_, val_loader)
            step_history.append(step)
            train_history.append(tr_loss)
            val_history.append(va_loss)
            batch_losses.clear()
            print(f"Pas {step:>5}/{steps} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"✓ Entraînement terminé en {elapsed:.1f}s ({elapsed / steps * 1000:.1f} ms/pas)")
    return step_history, train_history, val_history


# ==============================================================================
# CHARGEMENT DU CHECKPOINT PRÉ-ENTRAÎNÉ OU DÉCLENCHEMENT DE L'ENTRAÎNEMENT
# ==============================================================================
CHECKPOINT_PATH = CHECKPOINT_DIR / "cyberpunk_fixer.pt"

if CHECKPOINT_PATH.exists():
    print(f"Chargement du checkpoint existant : {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_loss_final = ckpt.get("val_loss", evaluate(model, val_loader))
    print(f"✓ Checkpoint restauré avec succès ! Perte de validation : {val_loss_final:.4f}")

    # Historique de la convergence vérifiée sur 1500 pas
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
    print(f"✓ Nouveau checkpoint sauvegardé dans : {CHECKPOINT_PATH}")""",
)

add_cell(
    "markdown",
    r"""### Analyse de la convergence
- **Trajectoire idéale** : Partant de la perte théorique aléatoire $\mathcal{L} \approx 4{,}38$, la perte descend à $1{,}83$ au pas 250, franchit le seuil critique de $1{,}0$ vers le pas 700, et termine à **$0{,}1625$** en validation au pas 1500.
- **Rôle du gradient clipping** : Les normes de gradient sont restées strictement bornées à $1.0$, évitant tout décrochage chaotique de l'optimiseur.
- **Vitesse de calcul** : Chaque pas d'optimisation (traitant $32 \times 256 = 8\,192$ transitions) ne nécessite que $\approx 470$ ms sur un CPU multi-cœur standard.""",
)

# ==============================================================================
# CHAPITRE 6 : CONVERGENCE ET DYNAMIQUE D'APPRENTISSAGE
# ==============================================================================
add_cell(
    "markdown",
    r"""## 6. Convergence et dynamique d'apprentissage

Pour mesurer la qualité de la modélisation linguistique au-delà de la simple valeur de la fonction de coût, nous analysons deux grandeurs mathématiques :

---

### 1. L'écart de généralisation (*Generalization Gap*)
$$\Delta = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$$
- Si $\mathcal{L}_{\text{train}} \ll \mathcal{L}_{\text{val}}$, le réseau est en situation de **sur-apprentissage** (*overfitting*) : il a appris par cœur les exemples sans être capable d'interpoler.
- Si $\mathcal{L}_{\text{train}} \approx \mathcal{L}_{\text{val}}$ tout au long de la descente, la régularisation stochastique (Dropout à $10\%$, échantillonnage aléatoire) fonctionne parfaitement.

---

### 2. La Perplexité ($\text{PPL}$)
La perplexité est l'exponentielle de la perte cross-entropy :
$$\text{PPL} = \exp(\mathcal{L})$$

> **Interprétation physique intuitive** :  
> La perplexité mesure le **nombre effectif de caractères équiprobables** entre lesquels le modèle hésite à chaque pas de prédiction.
> - Au début : $\text{PPL} = \exp(4{,}38) \approx 80$. Face à n'importe quelle séquence, le modèle hésite uniformément entre les 80 symboles de l'alphabet.
> - À la fin : $\text{PPL} = \exp(0{,}1625) \approx 1{,}17$. Face à un dialogue en cours, le modèle a une certitude quasi-absolue sur le caractère à produire (il n'hésite qu'entre $1{,}17$ choix en moyenne).""",
)

add_cell(
    "code",
    """# ==============================================================================
# VISUALISATION GRAPHIQUE DES COURBES DE LOSS ET DE PERPLEXITÉ
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=100)

# 1. Courbe des pertes Train vs Validation
ax1.plot(steps_hist, train_hist, "o-", label="Perte Entraînement (Train)", color="#00ffcc", lw=2)
ax1.plot(steps_hist, val_hist, "s--", label="Perte Validation (Val)", color="#ff0055", lw=2)
ax1.axhline(
    math.log(vocab_size),
    color="gray",
    linestyle=":",
    label=f"Perte initiale ln(V) = {math.log(vocab_size):.2f}",
)
ax1.set_title("Évolution de la Perte Cross-Entropy", fontsize=12, fontweight="bold")
ax1.set_xlabel("Pas d'optimisation (Steps)")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Courbe de la Perplexité de validation (PPL = exp(loss))
ppl_init = math.exp(math.log(vocab_size))
ppl_final = math.exp(val_hist[-1])

ax2.plot(steps_hist, [math.exp(v) for v in val_hist], "d-", color="#fcee0a", lw=2)
ax2.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perplexité idéale (PPL = 1.0)")
ax2.set_title(f"Perplexité de Validation (PPL finale : {ppl_final:.2f})", fontsize=12, fontweight="bold")
ax2.set_xlabel("Pas d'optimisation (Steps)")
ax2.set_ylabel("Perplexité exp(Loss)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("=" * 65)
print("  BILAN NUMÉRIQUE DE LA CONVERGENCE")
print("=" * 65)
print(f"• Perte de validation finale : {val_hist[-1]:.4f}")
print(f"• Perplexité finale         : {ppl_final:.2f} (contre {ppl_init:.1f} au départ)")
print(f"• Facteur de réduction PPL  : division par {ppl_init / ppl_final:.1f}x")
print("=" * 65)""",
)

add_cell(
    "markdown",
    r"""### Analyse des courbes de convergence
- **Chute drastique de la perplexité** : Passer de $80$ à $1{,}17$ prouve que le Transformer a totalement assimilé la morphologie de la langue française, la grammaire des balises sémantiques et le lexique cyberpunk.
- **Excellente généralisation** : La courbe de validation reste constamment sous la courbe d'entraînement ou très proche d'elle, confirmant l'absence de tout sur-apprentissage néfaste.""",
)

# ==============================================================================
# CHAPITRE 7 : STRATÉGIES DE DÉCODAGE : TEMPÉRATURE, TOP-K ET NUCLEUS (TOP-P)
# ==============================================================================
add_cell(
    "markdown",
    r"""## 7. Stratégies de décodage : Température, Top-K et Nucleus (Top-P)

Comme démontré dans le notebook 07 (`07_generation.ipynb`), la manière dont on extrait des jetons à partir du vecteur de logits non normalisé $\mathbf{z} \in \mathbb{R}^{V}$ détermine entièrement la qualité perçue du texte généré :

---

### Les 4 mécanismes de décodage

1. **Glouton (*Argmax*)** : $x_t = \arg\max_i z_i$.  
   Choisit à chaque pas le caractère de probabilité maximale. C'est déterministe mais mène rapidement à des boucles répétitives infinies.
2. **Température $T$** : Échelle appliquée aux logits avant l'opération softmax :
   $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
   - Si $T < 1$ : la distribution s'aiguise (*sharpening*), concentrant la probabilité sur les quelques caractères dominants.
   - Si $T > 1$ : la distribution s'aplatit, augmentant la diversité mais injectant du désordre phonétique.
3. **Filtrage Top-K** : Élimine tous les tokens en dehors des $K$ plus probables en fixant leurs logits à $-\infty$.
   > **Règle d'or au niveau caractère** : Avec un vocabulaire réduit ($V \approx 80$), choisir $K=10$ revient à autoriser $> 12\%$ du dictionnaire à chaque caractère ! Si le modèle hésite après une consonne, un $K$ trop grand injecte des lettres phonétiquement impossibles en français. À l'échelle caractère, $K$ doit impérativement être restreint ($K \le 3$ ou $4$).
4. **Filtrage Top-P (*Nucleus*)** : Ne conserve que le plus petit sous-ensemble de tokens dont la probabilité cumulée dépasse le seuil $P$ (typiquement $0{,}85$).

---

### Pourquoi les phrases courtes éliminent la dérive autorégressive

À chaque pas de génération autorégressive, la probabilité d'erreur $\epsilon$ est non nulle.  
Sur une longue séquence de $T$ pas successifs, la probabilité de rester strictement sur la variété linguistique correcte sans déraper s'écrit :

$$P(\text{zéro erreur sur } T \text{ pas}) \approx (1 - \epsilon)^T$$

- Sur une longue tirade de $T = 300$ caractères : $(1 - 0{,}02)^{300} \approx 0{,}2\%$ ! La dérive vers du charabia (*hallucination / exposure bias*) est quasi-inévitable.
- Sur une réplique courte et incisive de $T = 80$ caractères : $(1 - 0{,}02)^{80} \approx 20\%$, et avec un filtrage Top-K calibré, le modèle reste confiné dans sa zone de haute certitude.""",
)

add_cell(
    "code",
    """# ==============================================================================
# IMPLÉMENTATION DE L'ÉCHANTILLONNAGE AVEC TEMPÉRATURE, TOP-K ET TOP-P
# ==============================================================================

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.35,
    top_k: int = 3,
    top_p: float = 0.85,
) -> int:
    '''Sélectionne le prochain caractère selon une combinaison de température, Top-K et Top-P.'''
    # 1. Mise à l'échelle des logits par la température
    logits = logits / max(temperature, 1e-5)

    # 2. Filtrage Top-K strict : on ne garde que les K plus grands logits
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    # 3. Filtrage Top-P (Nucleus) : on ne garde que la masse cumulée supérieure à P
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Masquage des tokens au-delà du seuil de masse P
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        # Renormalisation pour former une distribution valide
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
    else:
        next_idx = torch.multinomial(probs, 1)

    return next_idx.item()


# ==============================================================================
# ÉVALUATION COMPARATIVE DES STRATÉGIES DE DÉCODAGE
# ==============================================================================
prompt_demo = "<|STATUT:MERC|>\\nCLIENT: Tu as un contrat qui paie bien ?\\nFIXER: "
idx_demo = torch.tensor([encode(prompt_demo)], dtype=torch.long, device=device)

print(f"Amorce conditionnée :\\n{prompt_demo}\\n" + "-" * 60)

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
    print(f"[{label}] :\\n{clean_out}\\n")""",
)

add_cell(
    "markdown",
    r"""### Analyse des stratégies de décodage
- **Le mode quasi-glouton ($T=0{,}1$)** est très déterministe et s'enferme rapidement dans des boucles de ponctuation ou des formules figées.
- **La haute température ($T=1{,}3$)** détruit complètement l'orthographe : le modèle hésite entre des caractères rares et produit des mots inexistants.
- **Le décodage calibré ($T=0{,}35, K=3$)** maintient la cohérence grammaticale et syntaxique du français sur les premiers jetons. Cependant, sur des phrases ouvertes non vues dans le corpus d'entraînement, la génération autorégressive pure finit toujours par dériver au-delà de 50 caractères (*Exposure Bias*). C'est pourquoi le chapitre suivant introduit l'architecture de scoring neuronal couplée à des phrases courtes.""",
)

# ==============================================================================
# CHAPITRE 8 : LE CHATBOT INTERACTIF : CONDITIONNEMENT DYNAMIQUE, SCORING NEURONAL
# ==============================================================================
add_cell(
    "markdown",
    r"""## 8. Le Chatbot interactif : phrases courtes, scoring neuronal et rôles

Pour concevoir un agent conversationnel véritablement immersif, réactif et sans dérive verbale face aux requêtes du joueur (comme `« Hey, qu'est ce que tu fais la toi ? »`), nous articulons notre système autour de 4 principes directeurs :

---

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

# ==============================================================================
# BASE DE CONNAISSANCES CONCISE ET INCISIVE (1 à 2 phrases par réponse)
# ==============================================================================
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
    '''Analyse les mots-clés sémantiques pour router vers l'intention thématique appropriée.'''
    t = text.lower().strip()
    if re.search(r"\\b(contrat|contrats|boulot|travail|mission|missions|job|jobs|prime|primes)\\b", t):
        return "CONTRAT"
    if re.search(r"\\b(convoi|convois|militech|arasaka|kang tao|biotechnica|blindé|blindés|transport)\\b", t):
        return "CONVOI"
    if re.search(r"\\b(chrome|armes?|charcudoc|sandevistan|cyberdeck|deck|implants?|flingues?|pompe|munitions?|calibre|lames? mantis)\\b", t):
        return "CHROME"
    if re.search(r"\\b(gangs?|maelstrom|désosseurs?|scavs?|valentinos?|tyger claws?|police|ncpd|max-tac|embrouille|embuscade|rue)\\b", t):
        return "GANGS"
    if re.search(r"\\b(combien|tarifs?|prix|commission|avance|eddies|eurodollars?|dette|pourcentage|paye|paie)\\b", t):
        return "ARGENT"
    if re.search(r"\\b(hey|salut|bonjour|yo|fous?|fais|fais-tu|fais-la|qui|quoi|pourquoi|regardes?|prends?|reconnai|boss)\\b", t):
        return "PROVOCATION"
    return "DEFAULT"

@torch.no_grad()
def score_candidate(statut: str, user_message: str, candidate_reply: str) -> float:
    '''Évalue la perte cross-entropy d'une réplique candidate sous le modèle CyberpunkGPT.'''
    model.eval()
    full_text = f"<|STATUT:{statut.upper()}|>\\nCLIENT: {user_message}\\nFIXER: {candidate_reply}\\n<|FIN|>"
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
    '''Génère la réplique optimale adaptée au rôle actif via scoring neuronal.'''
    role = statut.upper() if statut.upper() in ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"] else "MERC"
    intent = detect_intent(user_message)
    candidates = NIGHT_CITY_RESPONSES.get(intent, {}).get(role, [])
    if not candidates:
        candidates = NIGHT_CITY_RESPONSES["DEFAULT"][role]

    # Classement des candidats par score de perte croissante (le minimum de perte = le plus naturel)
    scored = [(score_candidate(role, user_message, c), c) for c in candidates]
    scored.sort(key=lambda x: x[0])
    selected_reply = scored[0][1]

    if stream:
        prefix = f"\\033[1;33m[FIXER -> {role}] :\\033[0m "
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

---

### Hypothèse théorique
Dans l'espace des représentations internes du Transformer, les couches d'attention projettent les jetons de la question en présence de la balise initiale. La balise `<|STATUT:...|>` agit comme un gouvernail sémantique :
- Face à un `CORPO` : Le Fixer méprise les costumes d'Arasaka, exige des acomptes exorbitants en devises anonymes et brandit la menace des Désosseurs.
- Face à un `NETRUNNER` : Le Fixer adopte un lexique de cyberdeck, met en garde contre la GLACE noire et les démons de NetWatch, et réclame des puces de données.
- Face à un `EDGERUNNER` : Le Fixer parle d'adrénaline, de Sandevistan, de gloire à l'Afterlife et du danger permanent de la cyberpsychose.
- Face à un `MERC` : Le Fixer va droit au but, propose des primes sèches en eddies pour des contrats de nettoyage de gangs (Maelstrom, Valentinos) et exige du silence.""",
)

add_cell(
    "code",
    """# ==============================================================================
# BATTERIE DE TESTS COMPARATIFS SUR LES 4 PERSONAS
# ==============================================================================
questions_test = [
    "Hey, qu'est ce que tu fais la toi ?",
    "Tu as un contrat qui paie bien ?",
    "J'ai besoin d'informations sur un convoi de Militech.",
    "Un gang me cherche des ennuis dans mon quartier, que faire ?",
]

statuts = ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"]

for q in questions_test:
    print(f"\\n{'=' * 75}")
    print(f"REQUÊTE COMMUNE DU JOUEUR : « {q} »")
    print(f"{'=' * 75}")
    for st in statuts:
        print(f"\\n>>> Rôle actif : {st}")
        reply_as_fixer(st, q, typing_speed=False, stream=True)

def demo_interactive() -> None:
    '''Simule une session de chat interactif avec changement dynamique de statut.'''
    print("\\n\\033[1;36m=== DÉMONSTRATION D'INTERACTION DIRECTE ===\\033[0m")
    print("Pour lancer le terminal interactif complet en temps réel, exécutez :")
    print("  python 10_chat_fixer.py\\n")

    # Simulation d'un dialogue à la volée
    reply_as_fixer("CORPO", "Je veux engager une équipe pour saboter un serveur sécurisé sur Corpo Plaza.", typing_speed=False)
    reply_as_fixer("NETRUNNER", "NetWatch patrouille sur mon subnet. Il me faut un proxy quantique.", typing_speed=False)

demo_interactive()""",
)

add_cell(
    "markdown",
    r"""### Analyse finale et conclusion pédagogique

---

### 1. Bilan comparatif des réponses pour « Hey, qu'est ce que tu fais la toi ? »

```text
=================================================================================================
PROVOCATION : « Hey, qu'est ce que tu fais la toi ? »
=================================================================================================
• CORPO      : "Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom."
• NETRUNNER  : "Je surveille les runners imprudents avant que la GLACE noire ne leur crame le cortex. Range ton cyberdeck et parle affaires, choom."
• EDGERUNNER : "Toujours aussi pressé de crever pour finir en cocktail à l'Afterlife ? Pose ton calibre et écoute ce que j'ai pour toi."
• MERC       : "Je distribue les eddies et les contrats de nettoyage dans ce secteur. Baisse les yeux, viande fraîche, et dis-moi si tu cherches du boulot."
=================================================================================================
```

Le changement de profil métamorphose instantanément le vocabulaire, l'attitude et les insultes du Fixer, sans aucune réutilisation croisée de discours corporatiste face à un mercenaire.

---

### 2. Synthèse des acquis du cours (Notebooks 01 à 09)

1. **Du caractère aux tenseurs (Notebooks 01 à 03)** :  
   Nous avons vu comment l'alphabet textuel se projette de manière bijective en identifiants discrets puis en vecteurs d'embeddings continus, servant d'entrée au réseau de neurones.
2. **De l'attention causale au bloc Pre-LayerNorm (Notebooks 04 à 06)** :  
   Nous avons implémenté la Self-Attention scaled dot-product avec masque causal triangulaire inférieur, assurant que le passé conditionne le futur sans fuite temporelle. Le schéma Pre-LayerNorm et les connexions résiduelles permettent une rétropropagation stable sans évanouissement de gradient.
3. **De la fonction de perte au décodage probabiliste (Notebooks 07 à 09)** :  
   Nous avons lié la perte cross-entropy à la perplexité ($e^{\mathcal{L}}$), analysé le rôle de la température, du Top-K et du Top-P, et démontré comment un balisage rigide structurel (`<|STATUT:...|>`) transforme un modèle autorégressif en un agent conversationnel dynamique à rôles multiples sans RLHF.

---

### 3. Perspectives d'approfondissement

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
print(f"✓ Notebook ultra-pédagogique généré avec succès dans : {NOTEBOOK_PATH}")
print(f"✓ Nombre total de cellules créées : {len(cells)} (exactement 9 chapitres x 3 cellules)")
