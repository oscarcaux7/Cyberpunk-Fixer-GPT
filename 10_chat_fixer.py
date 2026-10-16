#!/usr/bin/env python3
"""Chatbot interactif Night City — Incarne un Fixer véreux et cynique.

Caractéristiques :
- Phrases courtes, percutantes et cyniques (1 à 2 phrases max)
- Adaptation dynamique radicale selon le statut actif (CORPO, NETRUNNER, EDGERUNNER, MERC)
- Forte corrélation sémantique avec la question posée
- Cadence de frappe humaine réaliste (streaming immersif)
- Commandes directes : /statut <ROLE>, /aide, quit
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_cyberpunk_gpt import CyberpunkGPT, device

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

CHECKPOINT_PATH = ROOT / "checkpoints" / "cyberpunk_fixer.pt"

# ==============================================================================
# BASE DE RÉPONSES CONCISES ET INCISIVES (PAR INTENTION ET PAR RÔLE)
# ==============================================================================
NIGHT_CITY_RESPONSES = {
    # 1. SALUTATIONS, PROVOCATIONS ET QUESTIONS D'IDENTITÉ ("Hey, qu'est ce que tu fais la toi ?", "T'es qui ?")
    "PROVOCATION": {
        "CORPO": [
            "Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom.",
            "Je compte les eddies que ta boîte me verse pour étouffer vos sales dossiers. Baisse d'un ton ou je vends ton matricule aux Désosseurs.",
            "T'as cru que t'étais dans ton salon VIP sur Corpo Plaza ? Ici c'est Night City, costard, alors range ton arrogance.",
            "Je suis le Fixer qui décide si tu passes la nuit entier ou en pièces détachées. Montre les eddies ou raccroche ce holo.",
        ],
        "NETRUNNER": [
            "Je surveille les runners imprudents avant que la GLACE noire ne leur crame le cortex. Range ton cyberdeck et parle affaires, choom.",
            "Yo le runner, baisse la fréquence de ton deck ! Tu crois que j'ai pas senti ton ping sur mon sous-réseau ?",
            "NetWatch patrouille dans tout le secteur, alors fais profil bas. Dis-moi ce que tu cherches avant qu'on se fasse tracer.",
            "Je gère mon trafic de données pour payer mes packs de RAM. Pose tes questions vite, le canal n'est pas sécurisé.",
        ],
        "EDGERUNNER": [
            "Je regarde un sac de chrome sur pattes prêt à vriller en cyberpsychose. Range tes lames Mantis, choomba, et écoute le contrat.",
            "Toujours aussi pressé de crever pour finir en cocktail à l'Afterlife ? Pose ton calibre et écoute ce que j'ai pour toi.",
            "L'odeur du chrome brûlé et du Sandevistan, ça ne trompe pas. J'ai une mission suicide taillée sur mesure pour toi.",
            "Doucement sur la détente, choom ! Tes implants vibrent tellement qu'on t'entend depuis Heywood.",
        ],
        "MERC": [
            "Je distribue les eddies et les contrats de nettoyage dans ce secteur. Baisse les yeux, viande fraîche, et dis-moi si tu cherches du boulot.",
            "C'est moi qui donne les ordres ici, gonk. Si t'es pas là pour un contrat, dégage avant que je perde patience.",
            "Je cherche un solo capable de tirer droit sans poser de questions. Tu veux des eddies ou tu préfères papoter ?",
            "Ici le respect se gagne au fusil à pompe, pas avec des grandes phrases. Montre-moi ce que tu as dans le ventre.",
        ],
    },
    # 2. DEMANDES DE CONTRATS ET TRAVAIL ("Tu as un contrat ?", "Du boulot bien payé ?")
    "CONTRAT": {
        "CORPO": [
            "J'ai un vol de données confidentielles chez Biotechnica. Cinquante mille eddies d'acompte, en devises anonymes.",
            "Un contrat chirurgical pour éliminer un rival sur Corpo Plaza. Tarif doublé pour les costards, à prendre ou à laisser.",
            "Un convoi de Militech sans escorte lourde traverse Kabuki ce soir. Paie d'avance et je te donne les fréquences radio.",
        ],
        "NETRUNNER": [
            "Une forteresse de données de Kang Tao à infiltrer à distance. Tu pompes les schémas sans alerter la GLACE rouge, huit mille eddies.",
            "J'ai un shard crypté récupéré sur un agent de la NetWatch. Si tes démons le percent, je te file vingt pour cent du butin.",
            "Un entrepôt d'Arroyo a un subnet isolé à pirater avant l'aube. Branche ton câble, télécharge les registres et disparais.",
        ],
        "EDGERUNNER": [
            "Un repaire de Maelstrom blindé à nettoyer à Northside. Découpe tout ce qui bouge au Sandevistan, dix mille eddies à la clé !",
            "Un transport de chrome militaire des Désosseurs à intercepter. Fais tout sauter aux grenades et ramène la cargaison.",
            "Une prime sur un cyberpsycho armé d'un lance-roquettes à Pacifica. Rentre en un seul morceau et ton nom sera à l'Afterlife.",
        ],
        "MERC": [
            "Un colis sensible volé par les Tyger Claws à Little China. Élimine les sentinelles, récupère la mallette, quatre mille eddies nets.",
            "Un boulot de nettoyage dans les faubourgs de Santo Domingo. Pas de témoins, pas de bavures, paie cash à la livraison.",
            "Une cargaison d'armes disparue entre le port et Heywood. Tu refroidis les voleurs et tu touches trois mille eddies.",
        ],
    },
    # 3. CONVOIS MILITAIRES ET CIBLES BLINDÉES ("Des infos sur le convoi Militech ?", "Blindé ?")
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
    # 4. CHROME, ARMES ET CHARCUDOCS ("Où trouver un charcudoc ?", "Tu as du chrome ?", "Des armes ?")
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
    # 5. MENACES DE GANGS ET SÉCURITÉ ("Des ennuis avec les Valentinos / Maelstrom", "Un gang me cherche")
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
    # 6. ARGENT, TARIFS ET AVANCES ("Combien tu prends ?", "Une avance ?", "Quel tarif ?")
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
    # 7. PAR DÉFAUT / HORS-SUJET ("Tu fais quoi ?", open input)
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
    """Détecte l'intention sémantique via analyse de motifs lexicaux en français."""
    t = text.lower().strip()

    # 1. Contrats & Travail
    if re.search(r"\b(contrat|contrats|boulot|travail|mission|missions|job|jobs|prime|primes)\b", t):
        return "CONTRAT"
    # 2. Convois & Cibles corporatistes
    if re.search(r"\b(convoi|convois|militech|arasaka|kang tao|biotechnica|blindé|blindés|transport)\b", t):
        return "CONVOI"
    # 3. Chrome, Armes & Charcudocs
    if re.search(r"\b(chrome|armes?|charcudoc|sandevistan|cyberdeck|deck|implants?|flingues?|pompe|munitions?|calibre|lames? mantis)\b", t):
        return "CHROME"
    # 4. Gangs & Menaces de rue
    if re.search(r"\b(gangs?|maelstrom|désosseurs?|scavs?|valentinos?|tyger claws?|police|ncpd|max-tac|embrouille|embuscade|rue)\b", t):
        return "GANGS"
    # 5. Argent, Tarifs & Commissions
    if re.search(r"\b(combien|tarifs?|prix|commission|avance|eddies|eurodollars?|dette|pourcentage|paye|paie)\b", t):
        return "ARGENT"
    # 6. Salutations, Provocations & Questions d'identité
    if re.search(r"\b(hey|salut|bonjour|yo|fous?|fais|fais-tu|fais-la|qui|quoi|pourquoi|regardes?|prends?|reconnai|boss)\b", t):
        return "PROVOCATION"

    return "DEFAULT"


# ==============================================================================
# MOTEUR NEURONAL CYBERPUNK GPT
# ==============================================================================
class FixerEngine:
    def __init__(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            print(f"\033[1;31m[ERREUR] Checkpoint introuvable dans {ckpt_path}.\033[0m")
            sys.exit(1)

        checkpoint = torch.load(ckpt_path, map_location=device)
        self.model = CyberpunkGPT(**checkpoint["config"]).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.chars = checkpoint["chars"]
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.block_size = self.model.block_size

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    @torch.no_grad()
    def score_candidate(self, statut: str, user_message: str, candidate_reply: str) -> float:
        """Évalue la perte cross-entropy sous le modèle CyberpunkGPT."""
        full_text = f"<|STATUT:{statut.upper()}|>\nCLIENT: {user_message}\nFIXER: {candidate_reply}\n<|FIN|>"
        tokens = self.encode(full_text)
        if len(tokens) > self.block_size:
            tokens = tokens[-self.block_size :]
        if len(tokens) < 2:
            return 999.0

        idx = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
        _, loss = self.model(idx, targets)
        return loss.item()

    def reply(
        self,
        statut: str,
        user_message: str,
        typing_speed: bool = True,
    ) -> str:
        """Génère la réplique du Fixer : concise, percutante, adaptée au rôle."""
        role = statut.upper() if statut.upper() in ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"] else "MERC"
        intent = detect_intent(user_message)
        candidates = NIGHT_CITY_RESPONSES.get(intent, {}).get(role, [])
        if not candidates:
            candidates = NIGHT_CITY_RESPONSES["DEFAULT"][role]

        # Sélection neuronale : le modèle élit le candidat de cross-entropy minimale
        scored = [(self.score_candidate(role, user_message, c), c) for c in candidates]
        scored.sort(key=lambda x: x[0])
        selected_reply = scored[0][1]

        # Affichage avec cadence de frappe humaine
        prefix = "\033[1;33m[FIXER]\033[0m : "
        print(prefix, end="", flush=True)

        for char in selected_reply:
            print(char, end="", flush=True)
            if typing_speed:
                if char in {",", ";", ":"}:
                    time.sleep(random.uniform(0.04, 0.08))
                elif char in {".", "!", "?"}:
                    time.sleep(random.uniform(0.12, 0.20))
                else:
                    time.sleep(random.uniform(0.010, 0.022))

        print()
        return selected_reply


# ==============================================================================
# POINT D'ENTRÉE CLI INTERACTIF
# ==============================================================================
if __name__ == "__main__":
    engine = FixerEngine(CHECKPOINT_PATH)

    print("\033[1;36m" + "=" * 68 + "\033[0m")
    print("\033[1;36m  CONNEXION HOLO ACTIVÉE : TERMINAL FIXER DE NIGHT CITY\033[0m")
    print("\033[1;36m" + "=" * 68 + "\033[0m")
    statuts_valides = ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"]

    statut_actuel = "CORPO"
    print(f"Statut actif : \033[1;32m{statut_actuel}\033[0m")
    print(f"Statuts disponibles : {', '.join(statuts_valides)}")
    print("Commandes directes :")
    print("  • \033[1;35m/statut <ROLE>\033[0m  (ex: /statut MERC, /statut NETRUNNER)")
    print("  • \033[1;35mquit\033[0m            (ferme la communication holo)\n")

    while True:
        try:
            user_input = input(f"\033[1;32m[{statut_actuel}] Vous :\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nConnexion rompue.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\033[1;31m[Canal fermé]\033[0m Fais gaffe à ton chrome dans la rue, choom.")
            break

        if user_input.startswith("/statut"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1 and parts[1].upper().strip() in statuts_valides:
                statut_actuel = parts[1].upper().strip()
                print(f"--> \033[1;32mStatut mis à jour : {statut_actuel}\033[0m\n")
            else:
                print(f"--> \033[1;31mStatut invalide.\033[0m Choisissez parmi : {', '.join(statuts_valides)}\n")
            continue

        engine.reply(statut_actuel, user_input, typing_speed=True)
        print()
