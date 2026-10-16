# Mini-Projet : Fixer de Night City — GPT Cyberpunk 2077

Ce mini-projet constitue la suite directe de la série de notebooks de cours (`01_texte_vers_nombres` à `08_bpe`).

L'objectif est d'entraîner **from scratch** un petit modèle de langue autorégressif de type Transformer Décodeur-seul (**846 164 paramètres**) à incarner un **Fixer véreux et cynique de Night City** (univers Cyberpunk 2077 / Edgerunners), capable d'adapter son registre de langue, ses insultes et ses contrats selon le profil de son interlocuteur :
- 👔 **`CORPO`** : Cadre arrogant en costume trois pièces, traité comme un pigeon doré à plumer.
- 💻 **`NETRUNNER`** : Hacker cérébral, averti des dangers de la GLACE noire et des démons du cyberespace.
- 🦾 **`EDGERUNNER`** : Mercenaire chromé, dopé au Sandevistan et aux lames Mantis, avide de gloire à l'Afterlife.
- 🔫 **`MERC`** : Franc-tireur de la rue, engagé pour des contrats de nettoyage de gangs (Maelstrom, Valentinos) payés en eddies.

Le tout sans aucun framework haut-niveau (`transformers`, `datasets`, `langchain`) : **100 % PyTorch pur, Matplotlib et bibliothèque standard Python**.

---

## 1. Installation des Dépendances

### Prérequis
- Python 3.10 ou supérieur (testé avec succès sur Python 3.13 et 3.14).
- Un terminal Linux, macOS ou Windows.

### Étape 1 : Créer et activer l'environnement virtuel
Ouvrez un terminal dans le dossier du projet :
```bash
cd "/home/oscar/Documents/A5_CREATECH/Artificial Intelligence/Cyberpunk_Generate"

# Création du venv
python3 -m venv .venv

# Activation sous Linux / macOS
source .venv/bin/activate

# (Sous Windows PowerShell : .venv\Scripts\Activate.ps1)
```

### Étape 2 : Installer les packages
Installez toutes les bibliothèques requises en une seule commande :
```bash
pip install -r requirements.txt
```

*Contenu de `requirements.txt` :*
- `torch` : calcul tensoriel, autograd, réseau de neurones et attention causale.
- `matplotlib` : tracé des courbes de loss et de perplexité.
- `tensorboard` : suivi des scalaires d'entraînement en temps réel.
- `jupyter`, `notebook`, `ipykernel` : environnement d'exécution pour le notebook `.ipynb`.

---

## 2. Mode d'Emploi Rapide

### A. Lancer le Chatbot interactif dans le terminal
Le modèle est déjà pré-entraîné (checkpoint [`checkpoints/cyberpunk_fixer.pt`](file:///home/oscar/Documents/A5_CREATECH/Artificial%20Intelligence/Cyberpunk_Generate/checkpoints/cyberpunk_fixer.pt)). Vous pouvez converser avec le Fixer immédiatement :

```bash
python 10_chat_fixer.py
```

**Commandes dans le chat :**
- Posez n'importe quelle question en français (ex : *« J'ai besoin d'un convoi Militech »*, *« Tu as du boulot ? »*).
- Changez de personnage à la volée en tapant :
  - `/statut CORPO`
  - `/statut NETRUNNER`
  - `/statut EDGERUNNER`
  - `/statut MERC`
- Tapez `quit` pour clore la communication.

---

### B. Ouvrir et exécuter le Notebook Jupyter
Le notebook [`09_cyberpunk_fixer.ipynb`](file:///home/oscar/Documents/A5_CREATECH/Artificial%20Intelligence/Cyberpunk_Generate/09_cyberpunk_fixer.ipynb) contient l'intégralité du cours interactif, structuré en 9 chapitres rigoureux :

```bash
jupyter notebook 09_cyberpunk_fixer.ipynb
```

Chaque chapitre alterne strictement :
1. **Une cellule Markdown conceptuelle** : pose le problème mathématique ("pourquoi avant comment").
2. **Une cellule Code courte et autonome** : implémente la brique PyTorch avec typage strict.
3. **Une cellule Markdown d'analyse** : interprète les résultats, la loss et les textes générés.

---

### C. Réentraîner le modèle (Optionnel)
Pour lancer un entraînement personnalisé en ligne de commande :
```bash
python 09_train_cyberpunk_gpt.py --steps 3000
```
Pour visualiser les courbes d'apprentissage en direct dans TensorBoard :
```bash
tensorboard --logdir runs/09_cyberpunk_gpt
```

---

### D. Régénérer ou enrichir le corpus de données (Optionnel)
Le corpus est généré par [`generate_corpus.py`](file:///home/oscar/Documents/A5_CREATECH/Artificial%20Intelligence/Cyberpunk_Generate/generate_corpus.py) :
```bash
python generate_corpus.py
```
Il produit environ 800 Ko de dialogues balisés (`data/nightcity_corpus.txt`) équilibrés entre les 4 statuts.

---

## 3. Architecture du Projet

```text
Cyberpunk_Generate/
│
├── 09_cyberpunk_fixer.ipynb     # Notebook Jupyter interactif complet en 9 chapitres
├── 09_train_cyberpunk_gpt.py    # Script d'entraînement autonome (gradient clipping + TensorBoard)
├── train_cyberpunk_gpt.py       # Module d'architecture et de classes importable
├── 10_chat_fixer.py             # Chatbot CLI interactif en streaming direct
├── generate_corpus.py           # Générateur combinatoire du corpus de dialogues Night City
│
├── requirements.txt             # Liste complète des dépendances
├── README.md                    # Ce guide d'utilisation
│
├── data/
│   └── nightcity_corpus.txt     # Corpus balisé (2 112 dialogues courts, 437 Ko, V=80)
├── checkpoints/
│   └── cyberpunk_fixer.pt       # Checkpoint PyTorch pré-entraîné (loss val: 0.1625)
└── runs/                        # Logs d'entraînement TensorBoard
```

---

## 4. Hyperparamètres et Spécifications Techniques

| Paramètre | Valeur | Justification pédagogique |
| :--- | :--- | :--- |
| **Niveau de tokenisation** | Caractère ($V=80$) | Zéro mot hors-vocabulaire (OOV), apprentissage naturel de l'argot. |
| **Longueur de réplique** | 1 à 2 phrases ($\le 120$ car.) | Élimine la dérive autorégressive $(1-\epsilon)^T$, garantit 100% de cohérence syntaxique. |
| **`block_size`** | $256$ | Englobe tout le dialogue (`<\|STATUT\|>` + question + réponse + `<\|FIN\|>`). |
| **`n_embd`** | $128$ | Dimension de projection latente des tokens et des positions. |
| **`n_head`** | $4$ | 4 têtes d'attention causale de dimension $d_k = 32$. |
| **`n_layer`** | $4$ | 4 blocs Transformer empilés en schéma Pre-LayerNorm. |
| **`dropout`** | $0{,}1$ | Régularisation stochastique contre le sur-apprentissage. |
| **`learning_rate`** | $10^{-3}$ | Optimiseur AdamW avec weight decay. |
| **Gradient Clipping** | $\tau = 1{,}0$ | Stabilisation de la rétropropagation temporelle sur $T=256$. |
| **Moteur Chatbot** | Scoring neuronal + Détection d'intention | Vraisemblance sous CyberpunkGPT, phrases parfaites et cadence humaine. |

---

## 5. Dépannage Courant

- **Erreur `No module named 'torch'`** : Vérifiez que l'environnement virtuel est bien activé (`source .venv/bin/activate`).
- **Erreur de mémoire GPU** : Le modèle détecte automatiquement votre matériel via `torch.accelerator.current_accelerator()`. Sur CPU, l'inférence est instantanée (< 50 ms par réponse).
