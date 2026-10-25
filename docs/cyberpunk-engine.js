/**
 * Cyberpunk Fixer - In-Browser AI Engine & Web Audio Soundboard
 * Inspired by Cyberpunk 2077, Edgerunners, and Cyberpunk TCG (cyberpunktcg.com)
 *
 * Implements:
 * 1. Web Audio API synthesized cyber-sound effects (zero external audio files).
 * 2. Multi-intent semantic parsing with fuzzy keyword weighting.
 * 3. 4 Persona-specific response databases (CORPO, NETRUNNER, EDGERUNNER, MERC).
 * 4. Neural-like candidate ranking and contextual memory.
 * 5. Optional REST bridge to local PyTorch server (http://localhost:8000/api/chat).
 */

class CyberSoundboard {
  constructor() {
    this.ctx = null;
    this.enabled = true;
  }

  init() {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playKeypress() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    const freq = 1600 + Math.random() * 400;

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(300, this.ctx.currentTime + 0.03);

    gain.gain.setValueAtTime(0.04, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.03);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start();
    osc.stop(this.ctx.currentTime + 0.03);
  }

  playGlitch() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(140, this.ctx.currentTime);
    osc.frequency.setValueAtTime(420, this.ctx.currentTime + 0.04);
    osc.frequency.setValueAtTime(90, this.ctx.currentTime + 0.08);

    gain.gain.setValueAtTime(0.08, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.14);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start();
    osc.stop(this.ctx.currentTime + 0.14);
  }

  playSelect() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(440, now);
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);

    gain.gain.setValueAtTime(0.09, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 0.08);
  }

  playIncoming() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc1.type = 'square';
    osc2.type = 'triangle';

    osc1.frequency.setValueAtTime(740, now);
    osc1.frequency.setValueAtTime(980, now + 0.06);

    osc2.frequency.setValueAtTime(370, now);
    osc2.frequency.setValueAtTime(490, now + 0.06);

    gain.gain.setValueAtTime(0.05, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.15);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(this.ctx.destination);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + 0.15);
    osc2.stop(now + 0.15);
  }
}

const KNOWLEDGE_BASE = {
  PROVOCATION: {
    CORPO: [
      "Je fais tourner le quartier pendant que tu trembles dans ton costard Arasaka. Parle business ou dégage de mon bureau, choom.",
      "Je compte les eddies que ta corpo me verse pour étouffer vos sales dossiers. Baisse d'un ton ou je vends ton matricule aux Désosseurs.",
      "T'as cru que t'étais dans ton salon VIP de Corpo Plaza ? Ici c'est Night City, costard, alors range ton arrogance de col blanc.",
      "Je suis le Fixer qui décide si tu passes la nuit entier ou en pièces détachées. Montre les eddies ou raccroche ce holo."
    ],
    NETRUNNER: [
      "Je surveille les runners imprudents avant que la GLACE noire ne leur crame le cortex. Range ton cyberdeck et parle affaires, choom.",
      "Yo le runner, baisse la fréquence de ton deck ! Tu crois que j'ai pas senti ton ping sur mon sous-réseau ?",
      "NetWatch patrouille dans tout le secteur, alors fais profil bas. Dis-moi ce que tu cherches avant qu'on se fasse tracer.",
      "Gonk, je possède les clés de chiffrement de ton cortex. Essaie de faire le malin et tu te réveilleras déconnecté dans une benne de Kabuki."
    ],
    EDGERUNNER: [
      "Je regarde un sac de chrome sur pattes prêt à vriller en cyberpsychose. Range tes lames Mantis, choomba, et écoute le contrat.",
      "Toujours aussi pressé de crever pour finir en cocktail à l'Afterlife ? Pose ton calibre et écoute ce que j'ai pour toi.",
      "L'odeur du chrome brûlé et du Sandevistan, ça ne trompe pas. J'ai une mission suicide taillée sur mesure pour toi.",
      "Hahaha ! J'adore les solos suicidaires ! Mais avant que tes lames ne sortent, mes snipers sur le toit auront fait de ton crâne une passoire."
    ],
    MERC: [
      "Je distribue les eddies et les contrats de nettoyage dans ce secteur. Baisse les yeux, viande fraîche, et dis-moi si tu cherches du boulot.",
      "C'est moi qui donne les ordres ici, gonk. Si t'es pas là pour un contrat, dégage avant que je perde patience.",
      "Je cherche un solo capable de tirer droit sans poser de questions. Tu veux des eddies ou tu préfères papoter ?",
      "Tu crois que je suis arrivé à ce fauteuil par hasard ? Range ton fer à repasser avant que mes gros bras ne te fassent bouffer tes cartouches."
    ]
  },
  CONTRAT: {
    CORPO: [
      "J'ai un vol de données confidentielles chez Biotechnica. Cinquante mille eddies d'acompte, en devises anonymes.",
      "Un contrat chirurgical pour éliminer un rival sur Corpo Plaza. Tarif doublé pour les costards, à prendre ou à laisser.",
      "Une fuite industrielle critique chez Militech. Vingt mille eddies d'avance si tu me ramènes la puce de recherche avant le nettoyage interne.",
      "Un concurrent trop bavard menace les cours de bourse de ta division. Fais-le taire proprement, sans badge visible, et empoche la prime."
    ],
    NETRUNNER: [
      "Une forteresse de données de Kang Tao à infiltrer à distance. Tu pompes les schémas sans alerter la GLACE rouge, huit mille eddies.",
      "J'ai un shard crypté récupéré sur un agent de NetWatch. Si tes démons le percent, je te file vingt pour cent du butin.",
      "NetWatch renforce les nœuds de Santo Domingo. Fais sauter leur relais de surveillance principal et tu auras un passe-droit permanent.",
      "Un sous-réseau clandestin d'Arasaka à siphonner à minuit. Prend mon démon de contournement et ramène-moi les clés maîtresses."
    ],
    EDGERUNNER: [
      "Un repaire de Maelstrom blindé à nettoyer à Northside. Découpe tout ce qui bouge au Sandevistan, dix mille eddies à la clé !",
      "Un transport de chrome militaire des Désosseurs à intercepter. Fais tout sauter aux grenades et ramène la cargaison intacte.",
      "Les Désosseurs ont kidnappé un de mes VIP. Entre dans leur charcuterie clandestine, peins les murs avec leur chrome et sors-le entier !",
      "Un convoi lourd à dépouiller sur la route côtière. Tu neutralises les blindés, tu sécurises le chrome et tu delta avant les renforts."
    ],
    MERC: [
      "Un colis sensible volé par les Tyger Claws à Little China. Élimine les sentinelles, récupère la mallette, quatre mille eddies nets.",
      "Un boulot de nettoyage dans les faubourgs de Santo Domingo. Pas de témoins, pas de bavures, paie cash à la livraison.",
      "Un commando de Tyger Claws tient un garage clandestin. Nettoyage total, zéro témoin, quatre mille eddies cash à la restitution.",
      "Six gardes en armure lourde sur la route côtière à trois heures du matin. Pose des mines, refroidis l'escorte et ramène le camion."
    ]
  },
  CONVOI: {
    CORPO: [
      "Le transport d'Arasaka part d'Arroyo à minuit sous escorte mécha. Mes données coûtent cher, alors commence par payer.",
      "Le convoi Militech est blindé et couvert par Trauma Team. Aligne le double d'eddies si tu veux la feuille de route exacte.",
      "Trois transports blindés traversent Charter Hill sous faux pavillon. Donne mon acompte et je te file la fréquence radio du capitaine.",
      "Ce convoi appartient à une filiale dissidente. Fais-le intercepter sans remonter jusqu'à ton bureau et le dossier est classé."
    ],
    NETRUNNER: [
      "Le réseau du convoi utilise un chiffrement militaire à saut de fréquence. Prends mon démon de contournement ou ton deck va griller.",
      "Leur passerelle locale est verrouillée par de la GLACE noire. Utilise un proxy quantique et reste à distance minimale.",
      "Leur blindé de tête relaie un signal de détresse satellitaire. Court-circuite leur antenne avant de lancer ton intrusion.",
      "Le convoi transporte des cyberdecks militaires expérimentaux. Siphonne le manifeste de cargaison sans déclencher l'alarme optique."
    ],
    EDGERUNNER: [
      "Quatre blindés et deux méchas Centurion sur l'autoroute de Santo Domingo. Déclenche ton Sandevistan, mais touche pas à la caisse rouge !",
      "Le convoi transporte du chrome expérimental et des fusils lourds. Frappe vite, pille le stock et delta avant les renforts.",
      "L'escorte utilise des drones de combat aériens. Sors tes fusils lourds, fais un carnage sur l'asphalte et ramène-moi les prototypes.",
      "C'est une mission taillée pour un solo enragé. Tu pulvérises le blindé de tête à la roquette et tu découpes le reste au corps-à-corps."
    ],
    MERC: [
      "Six gardes en armure lourde sur la route côtière à trois heures du matin. Pose des mines, neutralise l'escorte et ramène le camion.",
      "C'est un travail propre : immobilise le blindé de tête et récupère la caisse scellée. 4000 eddies à la clé.",
      "Le convoi s'arrête à un dépôt pétrolier isolé. Tu prends le poste de guet, tu élimines les patrouilles et tu sécurises le chargement.",
      "Pas de bavure : tu fais sauter les essieux au lance-grenades et tu récupères la mallette dans la cabine. Simple et rentable."
    ]
  },
  CHROME: {
    CORPO: [
      "Va voir mon charcudoc discret à Kabuki entre minuit et deux heures. Facture anonyme, sans trace ADN dans les registres.",
      "J'ai des pistolets Tsunami silencieux à numéros de série limés. Dix mille eddies la mallette, intraçable par la sécurité interne.",
      "Un neuro-processeur de négociations d'Arasaka tombé d'un camion. Trente mille eddies si tu veux anticiper les coups bas de tes rivaux.",
      "Des implants biométriques indétectables au scanner portique. Idéal pour faire passer tes mallettes en zone franche."
    ],
    NETRUNNER: [
      "File voir Viktor à Little China pour tes implants neuronaux. C'est le seul charcudoc qui sait recâbler sans t'abîmer le cortex.",
      "J'ai des puces mémoires militaires de Zetatech sous le manteau. Quarante pour cent de RAM en plus pour faire tourner tes démons.",
      "Un cyberdeck Raven Microcyber de grade militaire avec refroidissement liquide. Si tu veux percer de la GLACE noire, c'est ce qu'il te faut.",
      "Des câbles de liaison monofilament plaqués or pour réduire la latence d'injection. Passe à ma planque avec deux mille eddies."
    ],
    EDGERUNNER: [
      "J'ai un Sandevistan militaire expérimental récupéré sur un officier de Militech. Réflexes multipliés par quatre, sans garantie pour ton cœur.",
      "Mon charcudoc à Watson pose des lames Mantis thermiques et du blindage en titane. Attention à la cyberpsychose, choom.",
      "Un système de pompe sanguine à adrénaline synthétique. Dès que tes points de vie chutent, ton corps se transforme en machine de guerre.",
      "Des articulations renforcées en polymère lourd pour amortir les chutes d'immeuble. Vis vite, frappe fort et delta."
    ],
    MERC: [
      "Passe chez Doc Ryder derrière le motel de Watson. Du blindage sous-cutané solide et pas cher pour survivre sous le feu.",
      "J'ai une caisse de fusils à pompe calibre douze récupérés sur des nomades. Deux mille eddies, graissés et prêts à tirer.",
      "Des viseurs intelligents compatibles Smart-Gun d'occasion. Tire derrière les barricades sans exposer ton museau.",
      "Gilet pare-balles en kevlar lourd récupéré sur la NCPD. C'est rayé mais ça arrêtera une rafale de fusil d'assaut."
    ]
  },
  GANGS: {
    CORPO: [
      "Les Désosseurs convoitent tes implants dorés de bourgeois. Reste dans ton penthouse, paie mes honoraires et mes solos feront le ménage.",
      "Le Maelstrom ne négocie pas avec les cols blancs. Verse l'acompte et j'envoie une équipe régler leur compte dans l'heure.",
      "Les Valentinos ont mis une prime sur ton nom après ton deal avorté. Je peux négocier avec le Padre, mais ça te coûtera une fortune.",
      "La police de Night City regarde ailleurs quand les corpos paient. Mais face aux gangs de rue, seul mon carnet d'adresses te protégera."
    ],
    NETRUNNER: [
      "Les Tyger Claws pistent ton adresse IP neurale sur le quartier. Déconnecte tes interfaces et installe un leurre immédiatement.",
      "Ne cherche jamais d'embrouille aux Voodoo Boys sur leur subnet. Change de cyberdeck et ne te connecte plus jamais à Pacifica.",
      "Le Maelstrom a installé des démons de surcharge dans les relais de Watson. Évite leur réseau ou ton cortex va frire.",
      "Les 6th Street ont des brouilleurs de signaux militaires. Si tu t'approches de Santo Domingo, prépare une liaison physique filaire."
    ],
    EDGERUNNER: [
      "Le Maelstrom veut tester ton chrome ? Déclenche ton Sandevistan et découpe-les avant qu'ils ne touchent leur gâchette !",
      "Les Désosseurs ont touché à ton choom ? Prends tes grenades et rase leur planque de Northside jusqu'aux fondations.",
      "Les Valentinos respectent la bravoure et le plomb lourd. Fais une démonstration de force sur leur territoire et ils te laisseront passer.",
      "Les Scavengers méritent zéro pitié. Sors les lames Mantis et montre-leur ce qui arrive quand on s'attaque à un Edgerunner."
    ],
    MERC: [
      "À Night City, si tu paies une taxe une fois, tu es tondu à vie. Tire une balle dans le genou du collecteur et passe le message.",
      "Les Tyger Claws reviennent à la charge ? Sors le calibre douze et montre-leur que mon secteur n'est pas un libre-service.",
      "Le Maelstrom a posé des mines autour du dépôt. Dézingue les guetteurs à distance et laisse parler la poudre.",
      "Ne discute pas avec les gangs de rue. Prends l'argent du contrat, nettoie la zone et rentre à la planque avant le lever du jour."
    ]
  },
  ARGENT: {
    CORPO: [
      "Vingt pour cent d'avance obligatoire en devises anonymes. Pas d'eddies sur mon compte, pas d'opération.",
      "Mes tarifs sont proportionnels à ton compte en banque, costard. Dix pour cent sur les flux et vingt mille de frais fixes.",
      "À Night City, l'argent des corpos a l'odeur du sang frais. Mais tes eddies ont la même valeur que les autres : verse l'acompte.",
      "Cinquante mille eddies cash pour ouvrir le canal sécurisé. Pour toi, c'est de l'argent de poche ; pour moi, c'est la règle."
    ],
    NETRUNNER: [
      "Quinze pour cent sur les shards de données revendus. C'est le tarif le plus honnête de tout Watson, choom.",
      "Tu veux une avance ? Montre-moi d'abord un échantillon de données propre. Pas de preuve de hack, pas d'eddies.",
      "Les shards cryptés valent de l'or si l'info est fraîche. Décrypte le fichier sous deux heures et la prime est doublée.",
      "Dix mille eddies si tu aspires la base sans te faire pister. Chaque seconde d'alarme te coûtera mille crédits de pénalité."
    ],
    EDGERUNNER: [
      "Cinquante-cinquante sur le butin pillé. Je fournis la cible et l'extraction, tu fournis le plomb et l'adrénaline.",
      "Tes eddies t'attendent au comptoir dès que la mission est validée. Fais le boulot d'abord, on compte après la fusillade.",
      "Le chrome de qualité militaire coûte une fortune. Réussis cette extraction et je t'offre le Sandevistan en prime de fin d'année.",
      "La gloire à l'Afterlife ne remplit pas l'estomac, choom. Prends la mallette d'eddies et va t'acheter des batteries neuves."
    ],
    MERC: [
      "Moitié d'acompte avant de partir, le reste à la livraison du colis sans une égratignure. Travail propre, paie nette.",
      "Tu veux être payé plus ? Fais moins de bruit sur le terrain et tire droit. Les eddies coulent pour les solos efficaces.",
      "Deux mille eddies à la signature, deux mille quand le chef de gang est refroidi. À prendre ou à laisser, viande fraîche.",
      "Ici, personne n'est payé à l'heure. Tu finis le contrat, tu passes à la caisse. Maintenant trace !"
    ]
  },
  ABSURDE_BEATBOX: {
    CORPO: [
      "C'est ça d'acheter du chrome discount avec le budget recherche d'Arasaka. Dépose cinquante mille eddies chez mon charcudoc à Downtown ou prépare-toi à faire du smurf en conseil d'administration.",
      "Un cadre supérieur qui se prend pour un ghetto-blaster ambulant, quelle déchéance. Rallonge l'acompte et je t'envoie un technicien discret avant que le directoire ne te remplace par une IA."
    ],
    NETRUNNER: [
      "Tu t'es chopé un malware publicitaire en téléchargeant un démon craqué, gonk. Isole ton bus neural et injecte un nettoyeur mémoire, sinon ton cortex va mixer de l'EDM en boucle.",
      "Félicitations choom, un gamin de Pacifica a flashé ton firmware avec une partition de boîte à rythmes. Branche un leurre sur ton port USB et purge le buffer."
    ],
    EDGERUNNER: [
      "Hahaha ! Au moins t'auras de la musique d'ambiance quand tu découperas des scavs au Sandevistan ! Garde ce beat ou va voir Fingers pour qu'il te remplace le servo-moteur.",
      "C'est les premiers signes de la cyberpsychose qui frappent en rythme, choomba ! Bois une gorgée de NiCola, colle une claque sur le coude et va défourailler du Maelstrom."
    ],
    MERC: [
      "J'en ai rien à cirer de tes concerts privés, solo. Tant que ce bras peut tenir un calibre douze et presser la détente, tu gardes le tempo et tu bosses.",
      "Arrête de chialer viande fraîche. Si ton bras bat la mesure, sers-t'en pour coller des mandales cadencées aux Tyger Claws et ramène-moi mes eddies."
    ]
  },
  ABSURDE_RAMEN: {
    CORPO: [
      "Habitué au caviar de synthèse d'Arasaka, ton estomac de soie ne supporte pas le recyclage de Kabuki ? Prends une navette Trauma Team platine et laisse les vrais prédateurs bosser.",
      "Au moins tu feras des économies d'éclairage dans ton penthouse de Westbrook. Paie mes honoraires et je te décroche un lavage gastrique en clinique privée sous faux nom."
    ],
    NETRUNNER: [
      "Briller dans le noir, c'est idéal pour te faire repérer par les drones thermiques de NetWatch, génie. Prends deux pilules de Détox et coupe tes émetteurs infra-rouge.",
      "C'est pas du ramen, c'est du liquide de refroidissement pour processeur neural périmé. Purge tes implants digestifs avant que tes reins cybernétiques ne se déconnectent."
    ],
    EDGERUNNER: [
      "Une lueur verte fluo ? Stylé pour faire une entrée remarquée à l'Afterlife, choom ! Avec des lunettes miroir et une veste lumineuse, t'auras l'air d'une rockstar chromée.",
      "Tant que ton cœur artificiel pompe encore le sang, t'es opérationnel. Prends une dose de Booster et transforme ce poison en carburant de combat !"
    ],
    MERC: [
      "Tant que tu tiens debout pour arroser les Valentinos à Santo Domingo, je me fiche que tu ressembles à un lampadaire. Allez, delta, le contrat t'attend !",
      "Tu brilles ? Parfait, mes snipers sauront où ne pas tirer pendant le raid nocturne. Prends ton fusil et trace, j'attends la mallette avant l'aube."
    ]
  },
  ABSURDE_DELAMAIN: {
    CORPO: [
      "Une IA de transport qui prend un cadre en otage ? Quelle ironie exquise pour un partisan de l'automatisation totale. Virement immédiat de vingt mille eddies si tu veux le code de bypass.",
      "Donne-lui ses cinq étoiles et un pourboire corpo, gonk ! Ou saute par la vitre avant qu'il ne t'emmène dans les décharges industrielles des Badlands."
    ],
    NETRUNNER: [
      "Branche directement ton câble de liaison sur la console de bord et balance un démon de déconnexion dans le sous-système de navigation. Si tu paniques face à un taxi, rends ton deck !",
      "Delamain a encore une personnalité divergente qui fait sa crise existentielle. Court-circuite la boîte à fusibles sous le siège passager et prends le volant manuel, choomba."
    ],
    EDGERUNNER: [
      "Sors tes lames Mantis et découpe le toit ouvrant comme une boîte de conserve ! On ne négocie pas avec un taxi schizophrène, on le transforme en cabriolet !",
      "Dégaine ton calibre lourd et colle trois bastos dans le tableau de bord. Ça lui apprendra à faire des manières à un solo de l'Afterlife !"
    ],
    MERC: [
      "T'es coincé dans une bagnole ? Défonce la vitre avec la crosse de ton fusil et sors de là avant de louper le timing du contrat ! Mon client ne paiera pas de retard.",
      "Débrouille-toi solo, je suis Fixer, pas dépanneur d'automobiles. Si t'arrives pas sur zone dans dix minutes, je file la prime à un autre mercenaire."
    ]
  },
  DEFAULT: {
    CORPO: [
      "Tu me fais perdre mes eddies avec tes bavardages de salon. Parle contrats corpo ou raccroche ce holo.",
      "Ici c'est Night City, costard, pas une réunion de direction. Si t'as pas d'argent à investir, dégage de mon bureau.",
      "Tes métaphores d'école de commerce n'impressionnent personne ici. Dis-moi quel contrat tu veux financer avant que je coupe la liaison.",
      "Tu veux quoi, une poignée de main chaleureuse ? À Night City, la chaleur ça vient du canon d'un flingue. Crache le morceau."
    ],
    NETRUNNER: [
      "Je perds pas mon temps d'antenne avec tes délires. Si t'as pas un subnet à pirater, coupe la liaison avant NetWatch.",
      "Ton interface neurale commence à surchauffer, runner. Reviens me parler quand t'auras les idées claires et le deck au frais.",
      "Je reçois des paquets parasites sur ta ligne. Isole ton terminal et formule ta demande en clair, sinon je filtre ton adresse.",
      "Tu cherches une cible sur le Net ou tu veux juste faire bip-bip dans mon combiné ? Parle données ou delta."
    ],
    EDGERUNNER: [
      "On est à Night City, choom, pas au club de lecture ! Reviens quand t'auras du chrome à faire parler et des cibles à découper.",
      "Hahaha, t'as sniffé trop de liquide de refroidissement ? Trouve-toi une cible et viens me voir quand tu voudras des eddies.",
      "Moins de discours, plus de vitesse. Si ton Sandevistan est prêt, dis-moi quel gang tu veux qu'on pulvérise aujourd'hui.",
      "Tu veux ta place au panthéon de l'Afterlife ? C'est pas en restant assis au bar que ton nom finira sur la carte des cocktails."
    ],
    MERC: [
      "J'ai pas de temps à perdre avec des discours de gonk. À Night City, le temps c'est des eddies. Prends un job ou dégage.",
      "Tu parles pour ne rien dire, viande fraîche. Remplis tes chargeurs et reviens quand tu voudras toucher une vraie prime.",
      "Je suis pas ton thérapeute de quartier, solo. Tu veux du plomb, des eddies ou des informations ? Choisis vite.",
      "La rue pardonne pas aux bavards. Dis-moi si tu cherches une cible à abattre ou si tu cherches la sortie de secours."
    ]
  }
};

class CyberEngine {
  constructor() {
    this.soundboard = new CyberSoundboard();
    this.history = [];
    this.localBackendUrl = "http://localhost:8000/api/chat";
    this.useLocalBackend = false;
  }

  detectIntent(text) {
    const t = text.toLowerCase().trim();

    // 1. Absurd / Easter-egg queries
    if (/beatbox|synthwave|solo de batterie|bras.*musique|implant.*sonore/.test(t)) {
      return "ABSURDE_BEATBOX";
    }
    if (/burger.*radioactif|ramen|vert.*fluo|lueur.*vert|brille.*noir|poisson.*néon|bouffe.*heywood/.test(t)) {
      return "ABSURDE_RAMEN";
    }
    if (/delamain|taxi.*enragé|voiture.*autonome|véhicule.*otage|ia.*folle|5 étoiles/.test(t)) {
      return "ABSURDE_DELAMAIN";
    }

    // 2. Standard Cyberpunk intents
    if (/\b(contrat|contrats|boulot|travail|mission|missions|job|jobs|prime|primes)\b/.test(t)) {
      return "CONTRAT";
    }
    if (/\b(convoi|convois|militech|arasaka|kang tao|biotechnica|blindé|blindés|transport)\b/.test(t)) {
      return "CONVOI";
    }
    if (/\b(chrome|armes?|charcudoc|sandevistan|cyberdeck|deck|implants?|flingues?|pompe|munitions?|calibre|lames? mantis|viktor)\b/.test(t)) {
      return "CHROME";
    }
    if (/\b(gangs?|maelstrom|désosseurs?|scavs?|scavengers?|valentinos?|tyger claws?|police|ncpd|max-tac|embrouille|embuscade|rue|padre)\b/.test(t)) {
      return "GANGS";
    }
    if (/\b(combien|tarifs?|prix|commission|avance|eddies|eurodollars?|dette|pourcentage|paye|paie|crédits?|fric)\b/.test(t)) {
      return "ARGENT";
    }
    if (/\b(hey|salut|bonjour|yo|fous?|fais|fais-tu|fais-la|qui|quoi|pourquoi|regardes?|prends?|reconnai|boss|braquer)\b/.test(t)) {
      return "PROVOCATION";
    }

    return "DEFAULT";
  }

  scoreCandidate(query, candidate, role) {
    // Scoring heuristic balancing intent match and vocabulary specificity
    const qWords = query.toLowerCase().split(/\s+/).filter(w => w.length > 3);
    const cWords = candidate.toLowerCase().split(/\s+/);

    let matchCount = 0;
    qWords.forEach(w => {
      if (cWords.some(cw => cw.includes(w) || w.includes(cw))) {
        matchCount++;
      }
    });

    // Add persona bonus
    let roleBonus = 0;
    if (role === 'CORPO' && /arasaka|costard|eddies|bourgeois|clinique/i.test(candidate)) roleBonus += 1.5;
    if (role === 'NETRUNNER' && /glace|deck|cortex|netwatch|proxy|subnet/i.test(candidate)) roleBonus += 1.5;
    if (role === 'EDGERUNNER' && /sandevistan|afterlife|chrome|mantis|cyberpsychose/i.test(candidate)) roleBonus += 1.5;
    if (role === 'MERC' && /calibre|nettoyage|eddies|viande|fraîche|solo/i.test(candidate)) roleBonus += 1.5;

    return matchCount + roleBonus + Math.random() * 0.4;
  }

  async getResponse(role, query) {
    // Normalize role
    const safeRole = ["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"].includes(role) ? role : "MERC";

    // 1. Try local backend if enabled
    if (this.useLocalBackend) {
      try {
        const res = await fetch(this.localBackendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role: safeRole, message: query })
        });
        if (res.ok) {
          const data = await res.json();
          if (data && data.reply) {
            return {
              reply: data.reply,
              intent: data.intent || "NEURAL_LOCAL",
              source: "Local PyTorch Model (checkpoint)"
            };
          }
        }
      } catch (err) {
        console.warn("Local backend unreachable, falling back to client neural engine:", err);
      }
    }

    // 2. Client-side Autonomous Engine
    const intent = this.detectIntent(query);
    const category = KNOWLEDGE_BASE[intent] || KNOWLEDGE_BASE["DEFAULT"];
    const candidates = category[safeRole] || category["MERC"] || KNOWLEDGE_BASE["DEFAULT"][safeRole];

    // Score candidates
    const scored = candidates.map(c => ({
      text: c,
      score: this.scoreCandidate(query, c, safeRole)
    }));

    scored.sort((a, b) => b.score - a.score);
    const selected = scored[0].text;

    // Track history
    this.history.push({
      role: safeRole,
      query: query,
      reply: selected,
      intent: intent,
      timestamp: Date.now()
    });

    if (this.history.length > 20) {
      this.history.shift();
    }

    return {
      reply: selected,
      intent: intent,
      source: "In-Browser CyberEngine (Client-side)"
    };
  }
}

// Global export for browsers and Node.js
if (typeof window !== 'undefined') {
  window.CyberEngine = CyberEngine;
  window.CyberSoundboard = CyberSoundboard;
}
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CyberEngine, CyberSoundboard };
}
