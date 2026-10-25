#!/usr/bin/env python3
"""server.py : Serveur HTTP local hybride pour le terminal Cyberpunk Fixer.

Fournit à la fois :
1. Le service des fichiers statiques de l'interface web (dossier docs/) sur http://localhost:8000
2. Une API REST (POST /api/chat) connectée au modèle PyTorch pré-entraîné (checkpoints/cyberpunk_fixer.pt)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHECKPOINT_PATH = ROOT / "checkpoints" / "cyberpunk_fixer.pt"

# Tentative de chargement du moteur PyTorch local
engine = None
try:
    from train_cyberpunk_gpt import CyberpunkGPT
    from importlib import import_module
    chat_mod = import_module("10_chat_fixer")
    if CHECKPOINT_PATH.exists():
        print(f"[PyTorch] Chargement du checkpoint : {CHECKPOINT_PATH}...")
        engine = chat_mod.FixerEngine(CHECKPOINT_PATH)
        print("[PyTorch] ✓ Moteur FixerEngine PyTorch opérationnel !")
    else:
        print(f"[Avertissement] Checkpoint introuvable dans {CHECKPOINT_PATH}. Mode fallback actif.")
except Exception as e:
    print(f"[Info] PyTorch non chargé ({e}). Le serveur servira l'interface avec son moteur embarqué.")


class CyberpunkHandler(SimpleHTTPRequestHandler):
    """Gestionnaire HTTP avec support de CORS et route API /api/chat."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_DIR), **kwargs)

    def end_headers(self) -> None:
        # Autorise les requêtes cross-origin pour tester facilement
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
                role = data.get("role", "MERC").upper()
                message = data.get("message", "").strip()

                if not message:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Message vide")
                    return

                # Si le moteur PyTorch est disponible, on l'utilise
                if engine is not None:
                    # stream=False, typing_speed=False pour réponse JSON instantanée
                    reply = engine.reply(role, message, typing_speed=False, stream=False)
                    intent = "PYTORCH_NEURAL_SCORING"
                else:
                    reply = f"[Fixer -> {role}] : Tu veux des eddies ou tu me fais perdre mon temps ? Viens me voir à Kabuki."
                    intent = "STANDALONE_FALLBACK"

                resp_payload = {
                    "role": role,
                    "message": message,
                    "reply": reply,
                    "intent": intent,
                    "engine": "PyTorch CyberpunkGPT (846k params)",
                }

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp_payload, ensure_ascii=False).encode("utf-8"))

            except Exception as e:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Erreur serveur : {e}")
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Route API inconnue")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serveur web et API pour le Fixer Cyberpunk")
    parser.add_argument("--port", type=int, default=8000, help="Port d'écoute (défaut: 8000)")
    args = parser.parse_args()

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, CyberpunkHandler)

    print("\n" + "=" * 68)
    print(f"  TERMINAL FIXER NIGHT CITY EN LIGNE : http://localhost:{args.port}")
    print("=" * 68)
    print("• Interface Web     : http://localhost:8000/")
    print("• API REST          : http://localhost:8000/api/chat")
    print("• Moteur PyTorch    :", "Actif (GPU/CPU)" if engine else "Inactif (Mode Client Web)")
    print("Appuyez sur Ctrl+C pour arrêter le serveur holo.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Serveur arrêté]")
        httpd.server_close()


if __name__ == "__main__":
    main()
