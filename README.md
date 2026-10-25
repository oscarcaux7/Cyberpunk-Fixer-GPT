# Mini-Projet : Fixer de Night City — GPT Cyberpunk 2077

[![Live Web Interface](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-fcee0a?style=for-the-badge&logo=github&logoColor=black)](https://oscarcaux7.github.io/Cyberpunk-Fixer-GPT/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-00f0ff?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Pure%20Tensor-ff0055?style=for-the-badge&logo=pytorch)](https://pytorch.org)

Ce mini-projet constitue la suite directe de la série de notebooks de cours (`01_texte_vers_nombres` à `08_bpe`).

L'objectif est d'entraîner **from scratch** un petit modèle de langue autorégressif de type Transformer Décodeur-seul (**846 164 paramètres**) à incarner un **Fixer véreux et cynique de Night City** (univers Cyberpunk 2077 / Edgerunners), capable d'adapter son registre de langue, ses insultes et ses contrats selon le profil de son interlocuteur :
- 👔 **`CORPO`** : Cadre arrogant en costume trois pièces, traité comme un pigeon doré à plumer.
- 💻 **`NETRUNNER`** : Hacker cérébral, averti des dangers de la GLACE noire et des démons du cyberespace.
- 🦾 **`EDGERUNNER`** : Mercenaire chromé, dopé au Sandevistan et aux lames Mantis, avide de gloire à l'Afterlife.
- 🔫 **`MERC`** : Franc-tireur de la rue, engagé pour des contrats de nettoyage de gangs (Maelstrom, Valentinos) payés en eddies.

Le tout sans aucun framework haut-niveau (`transformers`, `datasets`, `langchain`) : **100 % PyTorch pur, Matplotlib et bibliothèque standard Python**.

---

## 🌐 Interface Web Interactive (Inspirée de cyberpunktcg.com)

Le projet propose une interface web immersive hébergée directement sur **GitHub Pages** :

👉 **[Accéder au Terminal Web du Fixer (GitHub Pages)](https://oscarcaux7.github.io/Cyberpunk-Fixer-GPT/)**

### Caractéristiques de l'interface :
- 🎴 **Cartes Persona TCG** : Basculez en un clic entre `CORPO`, `NETRUNNER`, `EDGERUNNER` et `MERC`. L'ensemble du thème graphique (couleurs néon, accent, avatar) et le ton du Fixer se transforment en temps réel.
- 💬 **Holo Terminal & Typewriter Effect** : Streaming de réponses avec cadence de frappe humaine et bruits de clavier synthétisés par l'API Web Audio (aucun fichier son externe requis).
- 💡 **Suggestions subtiles (1-clic)** : Des boutons de requêtes dynamiques orientent les utilisateurs non familiers avec l'univers de Cyberpunk 2077 (*contrats, convoi Militech, chrome illégal, provocation, délires absordes*).
- 📖 **Guide de survie & Lexique de rue** : Un panneau latéral récapitule les termes clés (*Eddies, Choom, Gonk, GLACE noire, Sandevistan, Afterlife, Trauma Team*).
- ⚙️ **Architecture Hybride** : Fonctionne de manière **100% autonome dans le navigateur** (moteur IA côté client) sur GitHub Pages, tout en proposant un commutateur pour se relier à votre modèle PyTorch local si vous lancez `python server.py`.

---

## 1. Installation des Dépendances

### Prérequis
- Python 3.10 ou supérieur (testé avec succès sur Python 3.13 et 3.14).
- Un terminal Linux, macOS ou Windows.

### Étape 1 : Créer et activer l'environnement virtuel
```bash
cd "/home/oscar/Documents/A5_CREATECH/Artificial Intelligence/Cyberpunk_Generate"

# Création du venv
python3 -m venv .venv

# Activation sous Linux / macOS
source .venv/bin/activate
```

### Étape 2 : Installer les packages
```bash
pip install -r requirements.txt
```

---

## 2. Mode d'Emploi Rapide

### A. Lancer l'interface Web en local (avec API PyTorch)
Pour faire tourner l'interface web connectée directement au checkpoint local :
```bash
python server.py
```
Ouvrez ensuite votre navigateur sur **`http://localhost:8000`**.

### B. Lancer le Chatbot interactif dans le terminal
Le modèle est pré-entraîné dans [`checkpoints/cyberpunk_fixer.pt`](file:///home/oscar/Documents/A5_CREATECH/Artificial%20Intelligence/Cyberpunk_Generate/checkpoints/cyberpunk_fixer.pt). Vous pouvez converser dans le terminal :
```bash
python 10_chat_fixer.py
```

### C. Ouvrir et exécuter le Notebook Jupyter
Le cours complet en 9 chapitres ultra-détaillés et commentés :
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
├── docs/                        # Application Web hébergée sur GitHub Pages
│   ├── index.html               # Terminal Holo Cyberpunk & TCG Persona Cards
│   ├── style.css                # Feuille de style inspirée de cyberpunktcg.com
│   ├── app.js                   # Contrôleur d'interface, typewriter & suggestions
│   └── cyberpunk-engine.js      # Moteur IA client autonome & Soundboard Web Audio
│
├── index.html                   # Lanceur / Redirection racine vers docs/
├── server.py                    # Serveur HTTP local hybride (Web + API REST PyTorch)
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
