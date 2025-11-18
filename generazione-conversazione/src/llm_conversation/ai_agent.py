# File: src/llm_conversation/ai_agent.py
import os
import concurrent.futures
import time
from typing import List, Dict, Any, Iterator
import google.generativeai as genai
from .config import AgentConfig
from .logging_config import get_logger

logger = get_logger(__name__)

class AIAgent:
    def __init__(self, config: AgentConfig):
        self.name = config.name
        self.model_name = config.model
        self.system_prompt = config.system_prompt
        self.temperature = config.temperature
        self.ctx_size = config.ctx_size
        self.genai_model = None
        self._messages: List[Dict[str, Any]] = []

        # Aggiungi il prompt di sistema come primo messaggio
        if self.system_prompt:
            self._messages.append({"role": "system", "content": self.system_prompt})

        if os.environ.get("GOOGLE_API_KEY"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        self._initialize_model()

    def _initialize_model(self):
        if os.getenv("LLM_CONVERSATION_DRY_RUN", "0").lower() in ("1", "true"): return
        try:
            self.genai_model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt
            )
            logger.info(f"Agente '{self.name}': Modello '{self.model_name}' inizializzato.")
        except Exception as e:
            logger.error(f"Agente '{self.name}': Fallita inizializzazione del modello. Errore: {e}")
            self.genai_model = None

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_response(self) -> Iterator[str]:
        if not self._messages:
            logger.warning(f"Agente '{self.name}': Nessun messaggio nella lista _messages. Impossibile generare una risposta.")
            yield f"[ERRORE: Nessun messaggio disponibile per l'agente {self.name}]"; return

        if os.getenv("LLM_CONVERSATION_DRY_RUN", "0").lower() in ("1", "true") or not self.genai_model:
            yield f"[RISPOSTA SIMULATA per {self.name}]"; return

        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature, max_output_tokens=self.ctx_size,
        )
        try:
            timeout_seconds = int(os.environ.get("GEMINI_API_TIMEOUT", "60").split("#")[0].strip())
        except ValueError:
            raise ValueError("GEMINI_API_TIMEOUT deve essere un numero intero valido.")

        def api_call_wrapper():
            gemini_history = []
            for message in self._messages:
                role = "model" if message["role"] == "model" else "user"
                gemini_history.append({"role": role, "parts": [{"text": str(message["content"])}]})
            
            if not gemini_history:
                raise ValueError("La lista gemini_history è vuota. Impossibile eseguire pop().")
            last_message = gemini_history.pop()["parts"][0]["text"]
            if self.genai_model is None:
                raise ValueError("Il modello genai_model non è stato inizializzato.")
            chat = self.genai_model.start_chat(history=gemini_history)
            return chat.send_message(last_message, generation_config=generation_config)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(api_call_wrapper)
                response = future.result(timeout=timeout_seconds)

            if hasattr(response, "text") and response.text:
                yield response.text
            else:
                feedback = getattr(response, 'prompt_feedback', 'N/A')
                yield f"[RISPOSTA VUOTA O BLOCCATA: {feedback}]"
        except concurrent.futures.TimeoutError:
            yield f"[ERRORE: TIMEOUT per l'agente {self.name}]"
        except Exception as e:
            logger.error(f"Agente '{self.name}': Errore API: {e}")
            yield f"[ERRORE API per l'agente {self.name}: {e}]"