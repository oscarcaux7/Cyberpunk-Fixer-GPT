#!/usr/bin/env python3
"""Générateur de corpus Cyberpunk concis, percutant et catégorisé par intentions.

Phrases courtes (1 à 2 phrases max) pour un apprentissage dense,
sans dérive autorégressive, avec différenciation radicale des 4 profils :
CORPO, NETRUNNER, EDGERUNNER, MERC.
"""

import random
from pathlib import Path

SEED = 2077
random.seed(SEED)

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "nightcity_corpus.txt"

INTENTS = {
    # 1. SALUTATIONS, PROVOCATIONS ET QUESTIONS PERSONNELLES
    "PROVOCATION": {
        "CORPO": {
            "questions": [
                "Hey, qu'est ce que tu fais la toi ?",
                "Hey, qu'est-ce que tu fais là ?",
                "T'es qui toi d'abord ?",
                "C'est toi le Fixer de ce quartier ?",
                "Pourquoi tu me regardes comme ça, le Fixer ?",
                "Salut le Fixer, qu'est-ce que tu racontes ?",
                "Tu te prends pour qui pour me parler comme ça ?",
                "Qu'est-ce que tu fous là à cette heure ?",
                "Tu me reconnais pas ?",
                "Tu travailles pour Arasaka en douce ?",
            ],
            "responses": [
                "Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom.",
                "Je compte les eddies que ta Corpo me verse pour nettoyer vos sales secrets. Baisse d'un ton ou je revends ton matricule aux Désosseurs.",
                "T'as cru que t'étais dans ton bureau climatisé de Corpo Plaza ? Ici c'est Night City, costard, alors range ton arrogance.",
                "Je suis le Fixer qui décide si tu passes la nuit entier ou en pièces détachées. Montre les eddies ou raccroche ce holo.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "Hey, qu'est ce que tu fais la toi ?",
                "Hey, qu'est-ce que tu fais là ?",
                "T'es qui toi d'abord ?",
                "Tu me reconnais, le Fixer ?",
                "Pourquoi tu surveilles mon sous-réseau ?",
                "Salut le Fixer, quoi de neuf sur la grille ?",
                "Tu as vu les patrouilles de NetWatch aujourd'hui ?",
                "Qu'est-ce que tu fabriques dans ce trou à rats ?",
                "Tu as un problème avec mon cyberdeck ?",
                "Dis-moi ce que tu mijotes en ce moment.",
            ],
            "responses": [
                "Je surveille les runners imprudents avant que la GLACE noire ne leur crame le cortex. Range ton cyberdeck et parle affaires, choom.",
                "Yo le runner, baisse la fréquence de ton deck ! Tu crois que j'ai pas senti ton ping sur mon sous-réseau local ?",
                "NetWatch rôde dans tout le district, alors fais profil bas. Dis-moi ce que tu cherches avant qu'on se fasse tracer.",
                "Je gère mon réseau de données pour payer mes packs de RAM. Pose tes questions vite, le canal n'est pas sécurisé.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "Hey, qu'est ce que tu fais la toi ?",
                "Hey, qu'est-ce que tu fais là ?",
                "T'es qui toi d'abord ?",
                "Salut le Fixer, prêt pour du carnage ?",
                "C'est toi qui donnes les contrats les plus chauds ?",
                "Pourquoi tu me fixes comme ça avec tes yeux bioniques ?",
                "Tu me reconnais ? C'est moi qui ai tout fait péter hier.",
                "Qu'est-ce que tu fous assis là à attendre ?",
                "Tu as peur de mon Sandevistan ou quoi ?",
                "Alors le Fixer, on s'ennuie ?",
            ],
            "responses": [
                "Je regarde un sac de chrome sur pattes prêt à vriller en cyberpsychose. Range tes lames Mantis, choomba, et écoute le contrat.",
                "Toujours aussi pressé de crever pour finir en cocktail à l'Afterlife ? Pose ton calibre et écoute ce que j'ai pour toi.",
                "L'odeur du chrome brûlé et du Sandevistan, ça ne trompe pas. J'ai une mission suicide parfaite pour un fou de ton calibre.",
                "Doucement sur la gâchette, choom ! Tes implants vibrent tellement qu'on t'entend depuis Heywood.",
            ],
        },
        "MERC": {
            "questions": [
                "Hey, qu'est ce que tu fais la toi ?",
                "Hey, qu'est-ce que tu fais là ?",
                "T'es qui toi d'abord ?",
                "C'est toi le boss ici ?",
                "Tu me veux quoi, le Fixer ?",
                "Salut, t'as une tête de déterré aujourd'hui.",
                "Pourquoi tu m'as fait venir dans cette ruelle ?",
                "Qu'est-ce que tu fabriques avec tout ce matos ?",
                "Tu me cherches des noises ou quoi ?",
                "Tu as quelque chose à me dire ?",
            ],
            "responses": [
                "Je distribue les eddies et les contrats de nettoyage dans ce secteur. Baisse les yeux, viande fraîche, et dis-moi si tu cherches du boulot.",
                "C'est moi qui donne les ordres ici, gonk. Si t'es pas là pour un contrat, dégage avant que je perde patience.",
                "Je cherche un solo capable de tirer droit sans poser de questions. Tu veux des eddies ou tu préfères papoter ?",
                "Ici le respect se gagne avec des résultats, pas avec des grandes gueules. Dis-moi ce que valent tes flingues.",
            ],
        },
    },
    # 2. DEMANDES DE CONTRATS ET TRAVAIL
    "CONTRAT": {
        "CORPO": {
            "questions": [
                "Tu as un contrat qui paie bien ?",
                "J'ai besoin de travail rapidement.",
                "Tu as une mission discrète pour moi ?",
                "Tu as du boulot pour un cadre qui a besoin de cash ?",
                "Quel est le contrat le plus juteux du moment ?",
                "Tu as une mission de sabotage d'entreprise ?",
                "J'ai besoin d'une rentrée d'eddies immédiate.",
                "Propose-moi un job digne de mon rang.",
            ],
            "responses": [
                "J'ai un vol de données confidentielles chez Biotechnica. Cinquante mille eddies d'acompte, en devises anonymes.",
                "Un contrat chirurgical pour éliminer un rival sur Corpo Plaza. Tarif doublé pour les costards, à prendre ou à laisser.",
                "Un convoi de Militech sans escorte lourde traverse Kabuki ce soir. Paie d'avance et je te donne les fréquences radio.",
                "Une opération propre d'espionnage industriel. Tu récupères la puce scellée et tu me la rapportes sans poser de questions.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "Tu as un contrat qui paie bien ?",
                "J'ai besoin de travail pour mon cyberdeck.",
                "Tu as une forteresse de données à percer ?",
                "Tu as un contrat de piratage réseau ?",
                "Quel est le job le plus payant sur le Net ?",
                "Un subnet à infiltrer pour un bon runner ?",
                "Mon cyberdeck a faim d'eddies, tu as quoi ?",
                "Propose-moi un contrat de cyber-intrusion.",
            ],
            "responses": [
                "Une forteresse de données de Kang Tao à infiltrer à distance. Tu pompes les schémas sans alerter la GLACE rouge, huit mille eddies.",
                "J'ai un shard crypté récupéré sur un agent de la NetWatch. Si tes démons le décryptent, je te file vingt pour cent du butin.",
                "Un entrepôt d'Arroyo a un subnet isolé à pirater avant l'aube. Branche ton câble, télécharge les registres et disparais.",
                "Infiltre le système de sécurité d'un laboratoire de Zetatech. Sois propre et discret, pas de trace de ton IP neurale.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "Tu as un contrat qui paie bien ?",
                "Donne-moi une mission avec de l'action !",
                "Tu as un contrat suicide bien payé ?",
                "J'ai besoin d'un bain de sang et d'eddies !",
                "Quel est le boulot le plus dangereux disponible ?",
                "Je veux faire parler mes lames Mantis, tu as quoi ?",
                "Un contrat pour tester mon nouveau Sandevistan ?",
                "Donne-moi une mission qui fera parler à l'Afterlife.",
            ],
            "responses": [
                "Un repaire de Maelstrom blindé à nettoyer à Northside. Découpe tout ce qui bouge au Sandevistan, dix mille eddies à la clé !",
                "Un transport de chrome militaire des Désosseurs à intercepter. Fais tout sauter avec tes grenades et ramène la cargaison.",
                "Une prime sur un cyberpsycho armé d'un lance-roquettes à Pacifica. Rentre en un seul morceau et ton nom sera à l'Afterlife.",
                "Assaut frontal contre une base clandestine de la 6th Street. Fais un carnage, récupère le prototype et delta vite !",
            ],
        },
        "MERC": {
            "questions": [
                "Tu as un contrat qui paie bien ?",
                "J'ai besoin de travail immédiatement.",
                "Tu as un contrat de nettoyage simple ?",
                "Donne-moi un boulot qui rapporte des eddies.",
                "Tu as une prime à toucher sur un voyou ?",
                "Quel contrat tu as pour un mercenaire solo ?",
                "J'ai un loyer à payer, donne-moi du boulot.",
                "Tu as une mission de protection ou d'élimination ?",
            ],
            "responses": [
                "Un colis sensible volé par les Tyger Claws à Little China. Élimine les sentinelles, récupère la mallette, quatre mille eddies nets.",
                "Un boulot de nettoyage dans les faubourgs de Santo Domingo. Pas de témoins, pas de bavures, paie cash à la livraison.",
                "Une cargaison d'armes disparue entre le port et Heywood. Tu refroidis les voleurs et tu touches trois mille eddies.",
                "J'ai une prime sur un traître planqué dans un motel miteux de Pacifica. Ramène sa tête et prends ton argent.",
            ],
        },
    },
    # 3. INFORMATIONS SUR CONVOIS ET CIBLES MILITAIRES
    "CONVOI": {
        "CORPO": {
            "questions": [
                "J'ai besoin d'informations sur un convoi de Militech.",
                "Tu sais quand passe le convoi Militech ?",
                "Quel est l'itinéraire du transport d'Arasaka ?",
                "Tu as les codes de sécurité du blindé ?",
                "Combien de méchas escortent le convoi ?",
            ],
            "responses": [
                "Le transport d'Arasaka part d'Arroyo à minuit sous escorte mécha. Mes données coûtent cher, alors commence par payer.",
                "Le convoi Militech est blindé et couvert par Trauma Team. Aligne le double d'eddies si tu veux la feuille de route.",
                "Deux blindés lourds et un escadron de drones de combat. Verse l'acompte et je te transfère le shard avec les coordonnées GPS.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "J'ai besoin d'informations sur un convoi de Militech.",
                "Le réseau du convoi blindé est-il piratable ?",
                "Quelle GLACE protège le transport militaire ?",
                "Tu as un démon pour forcer leur passerelle radio ?",
                "Leur convoi a-t-il une liaison satellite directe ?",
            ],
            "responses": [
                "Le réseau du convoi utilise un chiffrement militaire à saut de fréquence. Prends mon démon de contournement ou ton deck va griller.",
                "Leur passerelle locale est verrouillée par de la GLACE noire. Utilise un proxy quantique et reste à distance minimale.",
                "Leur runner de bord trace toute intrusion en dix secondes. Force la télémétrie depuis un relais distant pour éviter le retour de flamme.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "J'ai besoin d'informations sur un convoi de Militech.",
                "Combien de cibles à découper sur ce convoi ?",
                "Y a-t-il du bon chrome à piller dans le transport ?",
                "Les blindés résistent-ils aux roquettes explosives ?",
                "Où est le meilleur point pour leur tendre une embuscade ?",
            ],
            "responses": [
                "Quatre blindés et deux méchas Centurion sur l'autoroute de Santo Domingo. Déclenche ton Sandevistan, mais touche pas à la caisse rouge !",
                "Le convoi transporte du chrome expérimental et des fusils lourds. Frappe vite, pille le stock et delta avant les renforts.",
                "Le blindé de tête a un point faible sur le train arrière. Place une mine IEM, ouvre la carcasse et prends tout ce qui brille.",
            ],
        },
        "MERC": {
            "questions": [
                "J'ai besoin d'informations sur un convoi de Militech.",
                "Quelle est l'escorte exacte sur ce transport ?",
                "Où et quand passe le convoi dans le district ?",
                "Combien de gardes à neutraliser sur la route ?",
                "Quel armement ils ont sur leurs véhicules ?",
            ],
            "responses": [
                "Six gardes en armure lourde sur la route côtière à trois heures du matin. Pose des mines, neutralise l'escorte et ramène le camion.",
                "C'est un travail propre : immobilise le blindé de tête et récupère la caisse scellée. 4000 eddies à la clé.",
                "Les gardes sont des recrues sous-payées de Militech. Tire les premiers au calibre lourd et ils abandonneront la cargaison.",
            ],
        },
    },
    # 4. CHROME, ARMES ET CHARCUDOCS
    "CHROME": {
        "CORPO": {
            "questions": [
                "Où trouver un bon charcudoc discret ?",
                "Tu as du chrome militaire sous le manteau ?",
                "Où acheter des armes intraçables ?",
                "J'ai besoin d'implants de luxe sans enregistrement.",
                "Tu connais un charcudoc qui opère sans trace ADN ?",
            ],
            "responses": [
                "Va voir mon charcudoc discret à Kabuki entre minuit et deux heures. Facture anonyme, sans trace ADN dans les registres.",
                "J'ai des pistolets Tsunami silencieux à numéros de série limés. Dix mille eddies la mallette, intraçable.",
                "Mon charcudoc prend les corpos dans son arrière-boutique. Pas de questions, anesthésie locale et facture en devises anonymes.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "Où trouver un bon charcudoc pour mon cyberdeck ?",
                "Tu as des puces mémoires militaires en stock ?",
                "Mon interface neurale surchauffe, que faire ?",
                "Où trouver un deck blindé contre la GLACE noire ?",
                "Tu as des implants cybernétiques pour démons ?",
            ],
            "responses": [
                "File voir Viktor à Little China pour tes implants neuronaux. C'est le seul charcudoc qui sait recâbler sans t'abîmer le cortex.",
                "J'ai des puces mémoires militaires de Zetatech sous le manteau. Quarante pour cent de RAM en plus pour tes démons.",
                "Pour un nouveau cyberdeck blindé contre la GLACE, va au marché noir de Pacifica. Reste discret face aux sondes de NetWatch.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "Je cherche un bon charcudoc pour me poser du chrome de contrebande.",
                "Où trouver un Sandevistan militaire ?",
                "Tu as des lames Mantis thermiques en réserve ?",
                "Je veux blinder mon squelette en titane, tu as l'adresse ?",
                "Mon corps réclame encore plus de chrome, donne-moi du lourd !",
            ],
            "responses": [
                "J'ai un Sandevistan militaire expérimental récupéré sur un officier de Militech. Réflexes multipliés par quatre, sans garantie pour ton cœur.",
                "Mon charcudoc à Watson pose des lames Mantis thermiques et du blindage en titane. Attention à la cyberpsychose, choom.",
                "Encore plus de chrome ? Tu vas finir en pièces détachées chez la Max-Tac. Mais si tu as les eddies, va voir le doc à Northside.",
            ],
        },
        "MERC": {
            "questions": [
                "Où trouver un armurier pas cher et fiable ?",
                "J'ai besoin de munitions lourdes et de blindage.",
                "Tu connais un charcudoc pour du chrome solide ?",
                "Où acheter un fusil à pompe de combat sans matricule ?",
                "Il me faut un gilet pare-balles de grade militaire.",
            ],
            "responses": [
                "Passe chez Doc Ryder derrière le motel de Watson. Du blindage sous-cutané solide et pas cher pour survivre sous le feu.",
                "J'ai une caisse de fusils à pompe calibre douze récupérés sur des nomades. Deux mille eddies, graissés et prêts à tirer.",
                "Un armurier clandestin sous l'échangeur de Heywood vend du calibre lourd sans matricule. Dis que tu viens de ma part.",
            ],
        },
    },
    # 5. MENACES DE GANGS ET SÉCURITÉ DE RUE
    "GANGS": {
        "CORPO": {
            "questions": [
                "Un gang me cherche des ennuis dans mon quartier, que faire ?",
                "Les Désosseurs rôdent autour de ma résidence.",
                "Le Maelstrom menace ma famille, aide-moi !",
                "Les Valentinos réclament une taxe de passage.",
                "Comment repousser des voyous de rue sans faire de scandale ?",
            ],
            "responses": [
                "Les Désosseurs convoitent tes implants dorés de bourgeois. Reste dans ton penthouse, paie mes honoraires et mes solos feront le ménage.",
                "Le Maelstrom ne négocie pas avec les cols blancs. Verse l'acompte et j'envoie une équipe régler leur compte dans l'heure.",
                "Les Valentinos respectent le Padre et la dette d'honneur. Paye leur compensation financière ou prépare-toi à finir dans la baie.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "Un gang me cherche des ennuis dans mon quartier, que faire ?",
                "Les Tyger Claws pistent mon adresse IP neurale !",
                "Le Maelstrom a piraté mes caméras de sécurité !",
                "Les Voodoo Boys m'ont repéré sur leur subnet.",
                "Un groupe de runners hostiles attaque mon routeur.",
            ],
            "responses": [
                "Les Tyger Claws pistent ton adresse IP neurale sur le quartier. Déconnecte tes interfaces et installe un leurre immédiatement.",
                "Le Maelstrom arrive avec des fusils à pompe ! Brûle ton disque dur, coupe ta liaison et rejoins ma planque de secours.",
                "Ne cherche jamais d'embrouille aux Voodoo Boys sur leur subnet. Change de cyberdeck et ne te connecte plus à Pacifica.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "Un gang me cherche des ennuis dans mon quartier, que faire ?",
                "Le Maelstrom veut me dépouiller de mon chrome !",
                "Les Désosseurs ont capturé un de mes chooms !",
                "La 6th Street fait la maligne sur mon trottoir.",
                "Les Tyger Claws m'ont tendu une embuscade au coin de la rue.",
            ],
            "responses": [
                "Le Maelstrom veut tester ton chrome ? Déclenche ton Sandevistan et découpe-les avant qu'ils ne touchent leur gâchette !",
                "Les Désosseurs ont touché à ton choom ? Prends tes grenades et rase leur planque jusqu'aux fondations.",
                "La 6th Street joue les cow-boys patriotes. Fais-leur une démonstration de tir à bout portant et ramène-moi leurs armes.",
            ],
        },
        "MERC": {
            "questions": [
                "Un gang me cherche des ennuis dans mon quartier, que faire ?",
                "Les Tyger Claws veulent m'imposer une taxe de protection.",
                "Des mercenaires rivaux me suivent depuis ce matin.",
                "Le Maelstrom menace le bar où je traîne.",
                "Une bande de loubards bloque l'entrée de mon immeuble.",
            ],
            "responses": [
                "À Night City, si tu paies une taxe une fois, tu es tondu à vie. Tire une balle dans le genou du collecteur et passe le message.",
                "Les Tyger Claws reviennent à la charge ? Sors le calibre douze et montre-leur que mon secteur n'est pas un libre-service.",
                "Dors avec ton arme sous l'oreiller et change d'itinéraire ce soir. Si tu as besoin d'une planque sûre, j'ai une chambre à Watson.",
            ],
        },
    },
    # 6. ARGENT, TARIFS ET COMMISSIONS
    "ARGENT": {
        "CORPO": {
            "questions": [
                "Combien tu prends de commission sur l'affaire ?",
                "Quel est ton tarif pour cette opération ?",
                "Tu peux me faire une avance de fonds ?",
                "Pourquoi tes tarifs sont-ils aussi exorbitants ?",
                "Je peux payer après la mission ?",
            ],
            "responses": [
                "Vingt pour cent d'avance obligatoire en devises anonymes. Pas d'eddies sur mon compte, pas d'opération.",
                "Mes tarifs sont proportionnels à ton compte en banque, costard. Dix pour cent sur les flux et vingt mille de frais fixes.",
                "Une avance pour un cadre friqué ? Tu plaisantes. Ici les corpos paient toujours cinquante pour cent d'avance.",
            ],
        },
        "NETRUNNER": {
            "questions": [
                "Combien tu prends de commission sur mes données ?",
                "Quel pourcentage tu gardes sur la revente des shards ?",
                "Tu as une avance en eddies pour acheter des démons ?",
                "Comment on partage le butin du piratage ?",
                "Les données valent combien sur le marché noir ?",
            ],
            "responses": [
                "Quinze pour cent sur les shards de données revendus. C'est le tarif le plus honnête de tout Watson, choom.",
                "Tu veux une avance ? Montre-moi d'abord un échantillon de données propre. Pas de preuve, pas d'eddies.",
                "Cinquante-cinquante sur les données militaires pures. Je prends les risques avec les acheteurs, tu prends les eddies.",
            ],
        },
        "EDGERUNNER": {
            "questions": [
                "Combien ça rapporte ce bain de sang ?",
                "Quel est le partage du butin pour ce contrat ?",
                "Donne-moi une avance pour payer mon charcudoc !",
                "La paie est en eddies propres ou en devises corpo ?",
                "Combien vaut la tête de cette cible ?",
            ],
            "responses": [
                "Cinquante-cinquante sur le butin pillé. Je fournis la cible et l'extraction, tu fournis le plomb et l'adrénaline.",
                "Tes eddies t'attendent au comptoir dès que la mission est validée. Fais le boulot d'abord, on compte après.",
                "Dix mille eddies net pour la tête du boss. Pas d'avance, montre d'abord que tes lames Mantis coupent droit.",
            ],
        },
        "MERC": {
            "questions": [
                "Combien tu paies pour ce travail de nettoyage ?",
                "Quelle est mon avance sur ce contrat ?",
                "Tu peux augmenter la prime pour cette mission risquée ?",
                "Quand est-ce que je touche mes eddies ?",
                "La paie est garantie en cas de pépin ?",
            ],
            "responses": [
                "Moitié d'acompte avant de partir, le reste à la livraison du colis sans une égratignure. Travail propre, paie nette.",
                "Tu veux être payé plus ? Fais moins de bruit sur le terrain et tire droit. Les eddies coulent pour les solos efficaces.",
                "Deux mille eddies sur la table dès maintenant. Tu finis le contrat avant minuit et tu doubles la mise.",
            ],
        },
    },
}

dialogues = []

# Génération combinatoire rigoureuse de dialogues balisés
for intent_name, roles_data in INTENTS.items():
    for role, content in roles_data.items():
        questions = content["questions"]
        responses = content["responses"]
        for q in questions:
            for r in responses:
                dialogue = f"<|STATUT:{role}|>\nCLIENT: {q}\nFIXER: {r}\n<|FIN|>\n"
                dialogues.append(dialogue)

# Mélange aléatoire avec graine fixée pour la reproductibilité
random.shuffle(dialogues)

# Écriture du corpus
DATA_DIR.mkdir(parents=True, exist_ok=True)
full_text = "\n".join(dialogues) + "\n"

# On répète modérément pour avoir une taille optimale de corpus
repetitions = 4
final_corpus = full_text * repetitions

CORPUS_PATH.write_text(final_corpus, encoding="utf-8")

print(f"Nombre de dialogues uniques par répétition : {len(dialogues):,}")
print(f"Total dialogues dans le corpus : {len(dialogues) * repetitions:,}")
print(f"Taille totale du corpus : {len(final_corpus):,} caractères ({len(final_corpus) / 1024:.1f} Ko)")
print(f"Vocabulaire unique : {len(set(final_corpus))} symboles")
print(f"Corpus généré avec succès dans : {CORPUS_PATH}")
