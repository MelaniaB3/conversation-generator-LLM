"""
Derived from: https://github.com/famiu/llm_conversation (original repository)
Modified by: Melania Balestri, 2025  adaptations for APP/IE matrix configuration and hosted usage
License: GNU AGPL v3.0 (see LICENSE in project root)
"""

import json
from pathlib import Path
from typing import List, Generator, Tuple, Dict

# Importa la classe AIAgent, l'unica dipendenza di cui ha bisogno
from .ai_agent import AIAgent


class ConversationManager:
    """
    Manager semplice che orchestra un dialogo ping-pong tra due agenti.
    È progettato per essere compatibile con la versione finale di AIAgent.
    """
    def __init__(self, agents: List[AIAgent], initial_message: str | None, **kwargs):
        self.agents = agents
        self.initial_message = initial_message or ""
        self.history: List[Dict[str, str]] = []

        # Salviamo i prompt originali per il file di output.
        self._original_system_prompts = {agent.name: agent.system_prompt for agent in agents}

    def run_conversation(self) -> Generator[Tuple[str, List[str]], None, None]:
        """Esegue il dialogo e gestisce la cronologia per ogni agente."""
        
        if not self.agents or len(self.agents) < 2:
            # Aggiunto un controllo di sicurezza per evitare errori se non ci sono abbastanza agenti
            return

        agent1 = self.agents[0]
        agent2 = self.agents[1]

        # --- Logica Corretta per il Primo Turno ---
        # L'Agente 1 riceve l'istruzione iniziale e genera il suo saluto.
        agent1.add_message("user", self.initial_message)
        response1 = "".join(list(agent1.get_response()))

        # Registriamo la prima frase di dialogo
        self.history.append({"speaker": agent1.name, "message": response1})
        yield (agent1.name, [response1])
        
        # Aggiorniamo le cronologie di entrambi gli agenti
        agent1.add_message("model", response1)
        agent2.add_message("user", response1)

        # Ciclo di conversazione per i turni successivi
        for _ in range(15): # Limite di turni
            # Turno Agente 2 (risponde al saluto dell'Agente 1)
            response2 = "".join(list(agent2.get_response()))
            self.history.append({"speaker": agent2.name, "message": response2})
            yield (agent2.name, [response2])
            agent2.add_message("model", response2)
            agent1.add_message("user", response2)

            # Turno Agente 1 (risponde alla risposta dell'Agente 2)
            response1 = "".join(list(agent1.get_response()))
            self.history.append({"speaker": agent1.name, "message": response1})
            yield (agent1.name, [response1])
            agent1.add_message("model", response1)
            agent2.add_message("user", response1)

            # Condizione di uscita
            if "goodbye" in response1.lower() or "concludes my report" in response1.lower():
                break

    def save_conversation(self, output_path: Path):
        """Salva la conversazione nel formato JSON richiesto."""
        agent_configs = []
        for agent in self.agents:
            # --- CORREZIONE CHIAVE ---
            # Accediamo agli attributi con i nomi corretti definiti nella classe AIAgent:
            # agent.model_name invece di agent.model
            agent_configs.append({
                "name": agent.name,
                "model": agent.model_name,  # << CORREZIONE QUI
                "temperature": agent.temperature,
                "ctx_size": agent.ctx_size,
                "system_prompt": self._original_system_prompts[agent.name]
            })
        
        # Formatta la conversazione per il salvataggio
        formatted_conv = []
        for i, msg in enumerate(self.history):
            turn_number = (i // 2) + 1
            formatted_conv.append({
                "turn": turn_number,
                "speaker": msg["speaker"],
                "message": msg["message"]
            })

        output_data = {"agents": agent_configs, "conversation": formatted_conv}
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
