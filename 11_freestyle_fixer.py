#!/usr/bin/env python3
"""11_freestyle_fixer.py : Chatbot Fixer de Night City en Improvisation Libre (Freestyle).

Propulsé par un modèle Qwen2.5-0.5B adapté avec LoRA sur le registre Cyberpunk 2077.
Capable d'improviser sur n'importe quel sujet loufoque tout en conservant son ton cynique,
son argot de rue et son mépris ciblé selon le profil du joueur.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

ROOT = Path.cwd()
CHECKPOINT_DIR = ROOT / "checkpoints" / "fixer_freestyle_lora"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

SYSTEM_PROMPT = (
    "Tu es un Fixer véreux, cynique, impitoyable et sarcastique de Night City (univers Cyberpunk 2077). "
    "Tu t'exprimes avec un argot de rue authentique (choom, choomba, gonk, eddies, chrome, charcudoc, "
    "GLACE noire, cyberdeck, Sandevistan, flatline, delta, Afterlife). "
    "Tu adaptes instantanément ton niveau de mépris, tes insultes, tes exigences financières et tes métaphores "
    "selon le profil de ton interlocuteur indiqué par la balise <|STATUT:...|> : "
    "CORPO (cadre arrogant d'Arasaka/Militech à plumer sans pitié), "
    "NETRUNNER (hacker cérébral averti des démons et de la surchauffe neuronale), "
    "EDGERUNNER (sac de chrome suicidaire en quête de gloire éphémère), "
    "ou MERC (chair à canon de la rue embauchée pour le sale boulot). "
    "Tes réponses sont courtes, percutantes (1 à 2 phrases max), drôles et pleines de venin. "
    "Commence toujours ta réponse directement par 'FIXER: '."
)

STATUTS_VALIDES = ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"]

COULEURS = {
    "cyan": "\033[1;36m",
    "vert": "\033[1;32m",
    "jaune": "\033[1;33m",
    "magenta": "\033[1;35m",
    "rouge": "\033[1;31m",
    "reset": "\033[0m",
    "gras": "\033[1m",
}


class FreestyleFixer:
    """Moteur conversationnel génératif freestyle pour le Fixer de Night City."""

    def __init__(self) -> None:
        print(f"{COULEURS['cyan']}[Système]{COULEURS['reset']} Initialisation du réseau neural ({device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        print(f"{COULEURS['cyan']}[Système]{COULEURS['reset']} Chargement du modèle de base {BASE_MODEL_ID}...")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float32)

        adapter_weights = CHECKPOINT_DIR / "adapter_model.safetensors"
        adapter_bin = CHECKPOINT_DIR / "adapter_model.bin"

        if adapter_weights.exists() or adapter_bin.exists():
            print(f"{COULEURS['vert']}[Succès]{COULEURS['reset']} Connexion de l'adaptateur LoRA Freestyle : {CHECKPOINT_DIR}")
            self.model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
        else:
            print(f"{COULEURS['jaune']}[Info]{COULEURS['reset']} Aucun adaptateur LoRA trouvé dans {CHECKPOINT_DIR}. Utilisation du modèle de base conditionné par prompt.")
            self.model = base_model

        self.model.to(device)
        self.model.eval()

        self.history: list[dict[str, str]] = []
        self.temperature: float = 0.75
        self.top_p: float = 0.85
        self.repetition_penalty: float = 1.15

    def generate_reply(self, statut: str, user_message: str) -> str:
        """Génère une réponse libre, drôle et cynique en streaming direct."""
        formatted_user_msg = f"<|STATUT:{statut}|>\nCLIENT: {user_message.strip()}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-4:],  # Mémoire courte des 2 derniers échanges
            {"role": "user", "content": formatted_user_msg},
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=100,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        print(f"\n{COULEURS['jaune']}[FIXER -> {statut}]{COULEURS['reset']} : ", end="", flush=True)

        full_reply = []
        for text_chunk in streamer:
            # Nettoyage si le modèle répète la balise FIXER:
            cleaned_chunk = text_chunk
            if not full_reply and cleaned_chunk.startswith("FIXER:"):
                cleaned_chunk = cleaned_chunk[len("FIXER:") :].lstrip()
            
            print(cleaned_chunk, end="", flush=True)
            full_reply.append(cleaned_chunk)

        thread.join()
        print()

        complete_text = "".join(full_reply).strip()
        # Enregistrement dans l'historique
        self.history.append({"role": "user", "content": formatted_user_msg})
        self.history.append({"role": "assistant", "content": f"FIXER: {complete_text}"})
        return complete_text


def run_interactive_terminal() -> None:
    fixer = FreestyleFixer()

    print("\n" + COULEURS["cyan"] + "=" * 70 + COULEURS["reset"])
    print(COULEURS["cyan"] + "  CONNEXION CYBERDECK : TERMINAL FIXER DE NIGHT CITY (MODE FREESTYLE)" + COULEURS["reset"])
    print(COULEURS["cyan"] + "=" * 70 + COULEURS["reset"])
    print(f"Modèle actif       : {COULEURS['vert']}Qwen2.5-0.5B + LoRA Night City{COULEURS['reset']}")
    print(f"Génération         : {COULEURS['magenta']}Autorégressive Ouverte (Sampling libre){COULEURS['reset']}")
    print("Statuts autorisés  : CORPO, NETRUNNER, EDGERUNNER, MERC")
    print("Commandes en ligne :")
    print(f"  • {COULEURS['jaune']}/statut <ROLE>{COULEURS['reset']}   (ex: /statut MERC, /statut NETRUNNER)")
    print(f"  • {COULEURS['jaune']}/temp <valeur>{COULEURS['reset']}   (ex: /temp 0.85 pour plus de folie/créativité)")
    print(f"  • {COULEURS['jaune']}/clear{COULEURS['reset']}           (réinitialise la mémoire de conversation)")
    print(f"  • {COULEURS['jaune']}quit{COULEURS['reset']}             (ferme la communication holo)\n")

    statut_actuel = "CORPO"
    print(f"Statut par défaut : {COULEURS['vert']}{statut_actuel}{COULEURS['reset']}")

    while True:
        try:
            prompt_indicator = f"\n{COULEURS['gras']}[{statut_actuel}] Vous{COULEURS['reset']} : "
            user_input = input(prompt_indicator).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[Holo interrompu]")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print(f"\n{COULEURS['rouge']}[Canal coupé]{COULEURS['reset']} Fais gaffe à ton chrome dans la rue, choom.")
            break

        if user_input.startswith("/statut"):
            parts = user_input.split()
            if len(parts) >= 2 and parts[1].upper() in STATUTS_VALIDES:
                statut_actuel = parts[1].upper()
                print(f"--> {COULEURS['vert']}Statut mis à jour : {statut_actuel}{COULEURS['reset']}")
            else:
                print(f"--> {COULEURS['rouge']}Statut invalide.{COULEURS['reset']} Choisis parmi : {', '.join(STATUTS_VALIDES)}")
            continue

        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[1])
                    fixer.temperature = max(0.1, min(1.5, val))
                    print(f"--> {COULEURS['vert']}Température fixée à : {fixer.temperature:.2f}{COULEURS['reset']}")
                except ValueError:
                    print(f"--> {COULEURS['rouge']}Valeur de température invalide.{COULEURS['reset']}")
            continue

        if user_input.startswith("/clear"):
            fixer.history.clear()
            print(f"--> {COULEURS['vert']}Historique de conversation réinitialisé.{COULEURS['reset']}")
            continue

        fixer.generate_reply(statut_actuel, user_input)


if __name__ == "__main__":
    run_interactive_terminal()
