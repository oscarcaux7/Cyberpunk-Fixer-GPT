/**
 * Cyberpunk Fixer - Main Application Controller
 * Handles UI events, Persona switching, Real-time Typewriter stream,
 * Sound effects, and Prompt Suggestion Chips.
 */

document.addEventListener("DOMContentLoaded", () => {
  // 1. Initialize Engine & Soundboard
  const engine = new CyberEngine();
  let currentRole = "CORPO";
  let isTyping = false;
  let sfxEnabled = true;

  // 2. DOM Elements
  const chatLog = document.getElementById("chat-log");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const roleCards = document.querySelectorAll(".tcg-card");
  const inputPrefix = document.getElementById("input-prefix");
  const chipsContainer = document.getElementById("chips-container");
  const sfxToggleBtn = document.getElementById("sfx-toggle");
  const scanlinesToggleBtn = document.getElementById("scanlines-toggle");
  const localBackendToggle = document.getElementById("local-backend-toggle");
  const backendLabel = document.getElementById("backend-label");
  const fixerRoleBadge = document.getElementById("fixer-role-badge");

  // 3. Dynamic Suggestion Prompts by Persona
  const PROMPT_SUGGESTIONS = {
    CORPO: [
      { text: "J'ai besoin d'un convoi Militech", icon: "💼" },
      { text: "Combien pour éliminer un rival sur Corpo Plaza ?", icon: "🎯" },
      { text: "Où trouver un charcudoc discret sans trace ADN ?", icon: "🩸" },
      { text: "Hey, qu'est-ce que tu fais là toi ?", icon: "⚡" },
      { text: "Mon cyber-bras fait du beatbox tout seul", icon: "🧪" }
    ],
    NETRUNNER: [
      { text: "J'ai besoin de démons pour percer de la GLACE noire", icon: "💻" },
      { text: "Tu as un shard crypté de la NetWatch ?", icon: "💾" },
      { text: "File-moi l'adresse de Viktor à Little China", icon: "🔌" },
      { text: "Hey, qu'est-ce que tu fais là toi ?", icon: "⚡" },
      { text: "J'ai mangé un ramen et mes veines brillent en vert néon", icon: "🧪" }
    ],
    EDGERUNNER: [
      { text: "Je veux un Sandevistan militaire expérimental", icon: "🦾" },
      { text: "Un repaire de Maelstrom à raser au Sandevistan ?", icon: "💥" },
      { text: "Comment éviter de vriller en cyberpsychose ?", icon: "🧠" },
      { text: "Hey, qu'est-ce que tu fais là toi ?", icon: "⚡" },
      { text: "Un taxi Delamain enragé refuse de me laisser descendre !", icon: "🚕" }
    ],
    MERC: [
      { text: "Tu as un contrat de nettoyage qui paie cash ?", icon: "💰" },
      { text: "Six gardes lourdement armés sur la route côtière", icon: "🔫" },
      { text: "Les Tyger Claws me cherchent des embrouilles", icon: "💀" },
      { text: "Hey, qu'est-ce que tu fais là toi ?", icon: "⚡" },
      { text: "Moitié d'acompte avant de partir, réglo ?", icon: "💵" }
    ]
  };

  // 4. Update Suggestions Bar
  function updateSuggestions(role) {
    if (!chipsContainer) return;
    chipsContainer.innerHTML = "";
    const list = PROMPT_SUGGESTIONS[role] || PROMPT_SUGGESTIONS["MERC"];

    list.forEach(item => {
      const chip = document.createElement("button");
      chip.className = "prompt-chip";
      chip.innerHTML = `<span class="chip-icon">${item.icon}</span> <span>${item.text}</span>`;
      chip.addEventListener("click", () => {
        userInput.value = item.text;
        userInput.focus();
        engine.soundboard.playSelect();
      });
      chipsContainer.appendChild(chip);
    });
  }

  // 5. Persona Switch Handler
  function setRole(newRole, triggerSound = true) {
    if (["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"].includes(newRole)) {
      currentRole = newRole;
      document.body.setAttribute("data-role", currentRole);

      // Update Card Active State
      roleCards.forEach(c => {
        if (c.getAttribute("data-role-card") === currentRole) {
          c.classList.add("selected");
        } else {
          c.classList.remove("selected");
        }
      });

      // Update HUD elements
      if (inputPrefix) inputPrefix.textContent = `[${currentRole}] :`;
      if (fixerRoleBadge) fixerRoleBadge.textContent = currentRole;

      updateSuggestions(currentRole);

      if (triggerSound) {
        engine.soundboard.playSelect();
      }

      // Append System Event in Chat
      appendSystemMessage(`Liaison Holo reconfigurée : Profil actif [${currentRole}]`);
    }
  }

  // Bind Card Click Events
  roleCards.forEach(card => {
    card.addEventListener("click", () => {
      const role = card.getAttribute("data-role-card");
      if (role && role !== currentRole) {
        setRole(role);
      }
    });
  });

  // 6. Chat Message Rendering
  function appendSystemMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row system";
    row.style.alignSelf = "center";
    row.style.maxWidth = "100%";
    row.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--theme-accent); background: rgba(0,0,0,0.4); padding: 4px 12px; border: 1px dashed var(--theme-accent); border-radius: 2px;">
        ◆ ${text} ◆
      </div>
    `;
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    row.innerHTML = `
      <div class="message-meta">
        <span class="sender-tag">[${currentRole}] VOUS</span>
        <span class="time">${time}</span>
      </div>
      <div class="message-body">${escapeHTML(text)}</div>
    `;
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendFixerStreamingMessage(fullText, intentName, callback) {
    isTyping = true;
    engine.soundboard.playIncoming();

    const row = document.createElement("div");
    row.className = "message-row fixer";
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    row.innerHTML = `
      <div class="message-meta">
        <span class="sender-tag">[FIXER ➔ ${currentRole}]</span>
        <span class="time">${time}</span>
        <span style="font-size: 0.7rem; color: var(--text-muted); opacity: 0.7;">// ${intentName}</span>
      </div>
      <div class="message-body">
        <span class="text-content"></span><span class="cursor"></span>
      </div>
    `;
    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;

    const textSpan = row.querySelector(".text-content");
    const cursor = row.querySelector(".cursor");

    let charIdx = 0;
    const speed = 14; // ms per char

    function typeNextChar() {
      if (charIdx < fullText.length) {
        const char = fullText[charIdx];
        textSpan.textContent += char;
        charIdx++;

        // Subtle audio blip every 3 characters
        if (charIdx % 3 === 0 && char !== ' ') {
          engine.soundboard.playKeypress();
        }

        chatLog.scrollTop = chatLog.scrollHeight;

        let delay = speed;
        if (char === ',' || char === ';') delay = 60;
        if (char === '.' || char === '!' || char === '?') delay = 120;

        setTimeout(typeNextChar, delay);
      } else {
        if (cursor) cursor.remove();
        isTyping = false;
        if (callback) callback();
      }
    }

    typeNextChar();
  }

  function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag));
  }

  // 7. Send Message Handler
  async function handleSend() {
    if (isTyping) return;
    const rawText = userInput.value.trim();
    if (!rawText) return;

    userInput.value = "";
    userInput.focus();

    // Check for Terminal Slash Commands
    if (rawText.startsWith("/")) {
      handleSlashCommand(rawText);
      return;
    }

    appendUserMessage(rawText);

    // Get response from Engine
    const result = await engine.getResponse(currentRole, rawText);
    appendFixerStreamingMessage(result.reply, result.intent);
  }

  // 8. Terminal Slash Commands Handler
  function handleSlashCommand(cmd) {
    const parts = cmd.split(" ");
    const command = parts[0].toLowerCase();
    const arg = parts[1] ? parts[1].toUpperCase() : "";

    if (command === "/statut" || command === "/role") {
      if (["CORPO", "NETRUNNER", "EDGERUNNER", "MERC"].includes(arg)) {
        setRole(arg);
      } else {
        appendSystemMessage("Statuts valides : /statut CORPO, /statut NETRUNNER, /statut EDGERUNNER, /statut MERC");
      }
      return;
    }

    if (command === "/clear") {
      chatLog.innerHTML = "";
      appendSystemMessage("Terminal purgé. Canal sécurisé avec le Fixer réouvert.");
      engine.soundboard.playGlitch();
      return;
    }

    if (command === "/help") {
      appendSystemMessage("Commandes : /statut <ROLE>, /clear, /sfx, /eddies, /scanlines, /help");
      return;
    }

    if (command === "/sfx") {
      sfxEnabled = !sfxEnabled;
      engine.soundboard.enabled = sfxEnabled;
      sfxToggleBtn.textContent = sfxEnabled ? "SFX: ON" : "SFX: OFF";
      appendSystemMessage(`Retour audio synthétique : ${sfxEnabled ? "ACTIVÉ" : "DÉSACTIVÉ"}`);
      return;
    }

    if (command === "/scanlines") {
      document.body.classList.toggle("no-scanlines");
      appendSystemMessage("Filtre CRT Scanlines basculé.");
      return;
    }

    if (command === "/eddies") {
      appendSystemMessage("Solde de votre compte fiduciaire clandestin : 14,850 E$ (Eurodollars).");
      return;
    }

    appendSystemMessage(`Commande inconnue '${cmd}'. Tapez /help pour la liste.`);
  }

  // 9. Input & Button Listeners
  sendBtn.addEventListener("click", handleSend);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    } else {
      // Audio feedback on typing
      engine.soundboard.playKeypress();
    }
  });

  // 10. Sound & Display Toggles
  if (sfxToggleBtn) {
    sfxToggleBtn.addEventListener("click", () => {
      sfxEnabled = !sfxEnabled;
      engine.soundboard.enabled = sfxEnabled;
      sfxToggleBtn.textContent = sfxEnabled ? "SFX: ON" : "SFX: OFF";
      sfxToggleBtn.classList.toggle("active", sfxEnabled);
      if (sfxEnabled) engine.soundboard.playSelect();
    });
  }

  if (scanlinesToggleBtn) {
    scanlinesToggleBtn.addEventListener("click", () => {
      document.body.classList.toggle("no-scanlines");
      const active = !document.body.classList.contains("no-scanlines");
      scanlinesToggleBtn.classList.toggle("active", active);
      engine.soundboard.playSelect();
    });
  }

  if (localBackendToggle) {
    localBackendToggle.addEventListener("change", (e) => {
      engine.useLocalBackend = e.target.checked;
      backendLabel.textContent = engine.useLocalBackend ? "PyTorch Local (8000)" : "In-Browser Engine (Client)";
      engine.soundboard.playSelect();
      appendSystemMessage(`Mode moteur : ${backendLabel.textContent}`);
    });
  }

  // 11. Initial Setup
  setRole("CORPO", false);
  appendFixerStreamingMessage(
    "Regardez-moi ce costume d'Arasaka qui vient s'encanailler à Kabuki... Tu crois que mes infos sont gratuites, gonk ? Crache les eddies ou raccroche ce holo.",
    "INITIAL_GREETING"
  );
});
