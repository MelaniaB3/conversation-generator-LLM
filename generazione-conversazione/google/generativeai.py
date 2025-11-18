# File: google/generativeai.py
# Questo è un modulo "shim" o "stub" locale.
# Simula le parti essenziali della libreria 'google-generativeai'
# per permettere al programma di avviarsi anche se non è installata
# e per facilitare la modalità dry-run.

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Any, List, Dict

# Configura un logger per questo modulo stub
logger = logging.getLogger(__name__)

# --- Variabili Globali Simulate ---
_api_key_is_set = False

# --- Funzioni Simulate ---

def configure(api_key: str | None = None):
    """Simula la funzione di configurazione dell'SDK."""
    global _api_key_is_set
    if api_key:
        _api_key_is_set = True
        logger.debug("Shim google.generativeai: configure() chiamata con una chiave API.")
    else:
        _api_key_is_set = False
        logger.debug("Shim google.generativeai: configure() chiamata senza chiave API.")


# --- Classi Simulate ---

@dataclass
class GenerationConfig:
    """Simula la classe GenerationConfig."""
    temperature: float = 0.8
    max_output_tokens: int = 2048
    response_mime_type: str = "text/plain"


class GenerativeModel:
    """Simula la classe GenerativeModel."""
    def __init__(self, model_name: str, system_instruction: str = ""):
        self.model_name = model_name
        self.system_instruction = system_instruction
        logger.debug(f"Shim GenerativeModel: istanziato per il modello '{model_name}'.")

    def start_chat(self, history: List[Dict[str, Any]]):
        """Simula l'avvio di una sessione di chat."""
        # Restituisce un'istanza di una classe di chat simulata e passa il nome del modello
        return _SimulatedChatSession(history=history, model_name=self.model_name)

    def generate_content(self, prompt: str, generation_config: GenerationConfig | None = None):
        """Simula la generazione di contenuto basata su un prompt."""
        if generation_config is None:
            generation_config = GenerationConfig()

        class MockResponse:
            def __init__(self, text: str):
                # Simula l'oggetto risposta che ha un attributo .text
                self.text = text
                self.prompt_feedback = "SIMULATED_OK"

        logger.info("Shim GenerativeModel: generate_content() chiamato. Restituzione di una risposta simulata.")
        simulated_text = f"[RISPOSTA SIMULATA DAL MODULO 'google/generativeai.py' per il modello '{self.model_name}']"
        return MockResponse(text=simulated_text)


class _SimulatedChatSession:
    """Classe interna che simula una sessione di chat attiva."""
    def __init__(self, history: List[Dict[str, Any]], model_name: str = "<unknown>"):
        self._history = history
        self.model_name = model_name

    def send_message(self, content: str, generation_config: GenerationConfig):
        """Simula l'invio di un messaggio e la ricezione di una risposta."""
        class MockResponse:
            def __init__(self, text: str):
                # Simula l'oggetto risposta che ha un attributo .text
                self.text = text
                self.prompt_feedback = "SIMULATED_OK"

        logger.info("Shim _SimulatedChatSession: send_message() chiamato. Restituzione di una risposta simulata.")
        
        simulated_text = f"[RISPOSTA SIMULATA DAL MODULO 'google/generativeai.py' per il modello '{self.model_name}']"
        return MockResponse(text=simulated_text)


# --- Sezione Tipi (per compatibilità) ---
# Alcuni SDK hanno un sottomodulo 'types', lo simuliamo.
class TypesModule:
    GenerationConfig = GenerationConfig

types = TypesModule()