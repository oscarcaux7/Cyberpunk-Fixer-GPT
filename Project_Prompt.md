# Instructions : Générer le notebook `09_cyberpunk_fixer.ipynb` (et le script de corpus)

Tu es un enseignant-chercheur en Deep Learning. Ton objectif est de créer un Jupyter Notebook complet et exécutable nommé `09_cyberpunk_fixer.ipynb` qui constitue la suite directe d'un cours construit autour de 8 notebooks (`01_texte_vers_nombres` à `08_bpe`).

Le but de ce mini-projet est d'entraîner "from scratch" un petit modèle GPT (niveau caractère ou BPE) à incarner un **Fixer véreux et caricatural de Night City (univers Cyberpunk 2077 / Edgerunners)**, puis d'interagir avec lui en mode Chatbot en adaptant son registre de langue et ses insultes selon le statut de l'interlocuteur (`CORPO`, `NETRUNNER`, `EDGERUNNER`, `MERC`).

---

## 1. Règles strictes de style et conventions de code (À RESPECTER ABSOLUMENT)

1. **Aucune librairie haut-niveau externe** : Interdiction d'utiliser `transformers` (HuggingFace), `datasets`, ou `langchain`. Tout doit être écrit en **PyTorch pur (`torch`, `torch.nn`, `torch.nn.functional`)**, `matplotlib` et la bibliothèque standard Python.
2. **Conventions techniques du cours** :
   - Utiliser des annotations de type Python modernes partout (`list[int]`, `tuple[torch.Tensor, torch.Tensor]`, `torch.Tensor | None = None`).
   - Détection du matériel en début de notebook :
     `device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"`
   - Gestion des chemins :
     ```python
     try:
         ROOT = Path(__file__).resolve().parent
     except NameError:
         ROOT = Path.cwd()
     DATA_DIR = ROOT / "data"
     CHECKPOINT_DIR = ROOT / "checkpoints"
     ```
   - Fixer la graine aléatoire (`SEED = 1337` ou `2077`, `torch.manual_seed(SEED)`).
3. **Ton pédagogique (Cellules Markdown)** :
   - Les cellules Markdown doivent être écrites en français, avec un style sobre, direct, mathématiquement rigoureux et intuitif ("pourquoi on fait ça avant de montrer comment").
   - Alterner systématiquement : **1 cellule Markdown d'explication conceptuelle** -> **1 cellule Code courte et autonome** -> **1 cellule Markdown d'analyse du résultat**.

---

## 2. Plan détaillé du Notebook `09_cyberpunk_fixer.ipynb` (Cellule par Cellule)

Génère le notebook en suivant exactement ces 9 chapitres :

### Introduction (Markdown + Code d'imports)
- **Markdown** : Expliquer comment transformer un petit modèle de langue génératif (prédiction du caractère suivant) en un **chatbot conditionné par un rôle** sans RLHF ni fine-tuning lourd, simplement grâce au formatage du corpus avec des balises structurelles :
  ```text
  <|STATUT:CORPO|>
  CLIENT: [Question]
  FIXER: [Réponse caricaturale]
  <|FIN|>