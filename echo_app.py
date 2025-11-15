#!/usr/bin/env python3
"""
Echo: Two-Agent Ollama Playground
A Gradio application for running conversations between two Ollama models
"""

import gradio as gr
import requests
import json
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from enum import Enum
import ollama
from ollama import Client
import logging
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

class ContextMode(Enum):
    """Context memory modes for agents"""
    NO_MEMORY = "No Memory (System + Last Message)"
    LAST_MESSAGE = "Last Exchange Only"
    ROLLING_2 = "Rolling Window (Last 2)"
    ROLLING_5 = "Rolling Window (Last 5)"
    ROLLING_10 = "Rolling Window (Last 10)"
    ROLLING_20 = "Rolling Window (Last 20)"
    ROLLING_50 = "Rolling Window (Last 50)"
    FULL_HISTORY = "Full History"

@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str = "Agent"
    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_mode: str = ContextMode.ROLLING_10.value

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05

    # Generation control
    max_tokens: int = 512
    context_window: int = 4096
    seed: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)

    # Repetition control
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Advanced sampling
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    tfs_z: float = 1.0

class ConversationState:
    """Manages the conversation state between agents"""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.is_running: bool = False
        self.current_turn: str = "left"  # "left" or "right"
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def add_message(self, role: str, content: str, agent: str):
        """Add a message to the conversation history"""
        with self.lock:
            self.messages.append({
                "role": role,
                "content": content,
                "agent": agent,
                "timestamp": time.time()
            })

    def get_messages_for_agent(self, agent: str, config: AgentConfig) -> List[Dict[str, str]]:
        """Get conversation context based on agent's context mode"""
        with self.lock:
            mode = config.context_mode

            # Always start with system prompt
            context = [{"role": "system", "content": config.system_prompt}]

            if mode == ContextMode.NO_MEMORY.value:
                # Only the last message from the other agent
                if self.messages:
                    other_agent = "right" if agent == "left" else "left"
                    for msg in reversed(self.messages):
                        if msg["agent"] == other_agent:
                            context.append({"role": "user", "content": msg["content"]})
                            break

            elif mode == ContextMode.LAST_MESSAGE.value:
                # Last exchange only (one message from each agent)
                if len(self.messages) >= 2:
                    context.append({"role": "assistant", "content": self.messages[-2]["content"]})
                    context.append({"role": "user", "content": self.messages[-1]["content"]})
                elif len(self.messages) == 1:
                    context.append({"role": "user", "content": self.messages[-1]["content"]})

            elif mode.startswith("Rolling Window"):
                # Extract window size from mode string
                window_size = int(mode.split("(Last ")[1].split(")")[0])
                recent_messages = self.messages[-window_size:] if self.messages else []

                for msg in recent_messages:
                    role = "assistant" if msg["agent"] == agent else "user"
                    context.append({"role": role, "content": msg["content"]})

            elif mode == ContextMode.FULL_HISTORY.value:
                # All messages
                for msg in self.messages:
                    role = "assistant" if msg["agent"] == agent else "user"
                    context.append({"role": role, "content": msg["content"]})

            return context

    def clear(self):
        """Clear conversation history"""
        with self.lock:
            self.messages = []
            self.current_turn = "left"

class OllamaAgent:
    """Represents a single Ollama agent"""

    def __init__(self, name: str, config: AgentConfig):
        self.name = name
        self.config = config
        self.client = Client(host=OLLAMA_BASE_URL)

    def prepare_options(self) -> Dict[str, Any]:
        """Prepare options dictionary for Ollama API"""
        options = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "min_p": self.config.min_p,
            "num_predict": self.config.max_tokens,
            "num_ctx": self.config.context_window,
            "repeat_penalty": self.config.repeat_penalty,
            "repeat_last_n": self.config.repeat_last_n,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "mirostat": self.config.mirostat,
            "mirostat_tau": self.config.mirostat_tau,
            "mirostat_eta": self.config.mirostat_eta,
            "tfs_z": self.config.tfs_z,
        }

        # Add optional parameters
        if self.config.seed is not None:
            options["seed"] = self.config.seed

        if self.config.stop_sequences:
            options["stop"] = self.config.stop_sequences

        return options

    def generate_streaming(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Generate response with streaming"""
        try:
            stream = self.client.chat(
                model=self.config.model,
                messages=messages,
                options=self.prepare_options(),
                stream=True
            )

            for chunk in stream:
                if chunk and 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            yield f"\n[Error: {str(e)}]"

class EchoApp:
    """Main application class"""

    def __init__(self):
        self.conversation = ConversationState()
        self.left_agent: Optional[OllamaAgent] = None
        self.right_agent: Optional[OllamaAgent] = None
        self.left_display_name: str = "Student"
        self.right_display_name: str = "Teacher"
        self.available_models: List[str] = []
        self.conversation_thread: Optional[threading.Thread] = None

        # Load configuration if exists
        self.load_config()

        # Fetch available models
        self.refresh_models()

    def load_config(self):
        """Load configuration from yaml file if exists"""
        config_path = Path("config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info("Loaded configuration from config.yaml")
                    # Apply config if needed
            except Exception as e:
                logger.warning(f"Could not load config.yaml: {e}")

    def refresh_models(self) -> List[str]:
        """Fetch available models from Ollama"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                logger.info(f"Found {len(self.available_models)} models")
            else:
                logger.warning(f"Could not fetch models: {response.status_code}")
                self.available_models = [DEFAULT_MODEL]
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            self.available_models = [DEFAULT_MODEL]

        return self.available_models

    def create_agent_config(self, params: Dict[str, Any]) -> AgentConfig:
        """Create agent configuration from UI parameters"""
        # Parse stop sequences from string input
        stop_sequences = []
        if params.get('stop_sequences'):
            stop_sequences = [s.strip() for s in params['stop_sequences'].split(',') if s.strip()]

        return AgentConfig(
            name=params.get('name', 'Agent'),
            model=params.get('model', DEFAULT_MODEL),
            system_prompt=params.get('system_prompt', DEFAULT_SYSTEM_PROMPT),
            context_mode=params.get('context_mode', ContextMode.ROLLING_10.value),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 0.9),
            top_k=params.get('top_k', 40),
            min_p=params.get('min_p', 0.05),
            max_tokens=params.get('max_tokens', 512),
            context_window=params.get('context_window', 4096),
            seed=params.get('seed') if params.get('seed') else None,
            stop_sequences=stop_sequences,
            repeat_penalty=params.get('repeat_penalty', 1.1),
            repeat_last_n=params.get('repeat_last_n', 64),
            presence_penalty=params.get('presence_penalty', 0.0),
            frequency_penalty=params.get('frequency_penalty', 0.0),
            mirostat=params.get('mirostat', 0),
            mirostat_tau=params.get('mirostat_tau', 5.0),
            mirostat_eta=params.get('mirostat_eta', 0.1),
            tfs_z=params.get('tfs_z', 1.0)
        )

    def run_conversation_loop(self, left_params: Dict, right_params: Dict,
                            initial_prompt: Optional[str] = None) -> Generator[str, None, None]:
        """Run the conversation loop between agents"""
        # Create agents with current configurations
        left_config = self.create_agent_config(left_params)
        right_config = self.create_agent_config(right_params)

        self.left_agent = OllamaAgent("left", left_config)
        self.right_agent = OllamaAgent("right", right_config)

        # Store display names from configs
        self.left_display_name = left_config.name
        self.right_display_name = right_config.name

        # Add initial prompt if provided
        if initial_prompt:
            self.conversation.add_message("user", initial_prompt, "user")
            yield self.format_conversation()

        # Start conversation loop
        while self.conversation.is_running and not self.conversation.stop_event.is_set():
            try:
                # Determine which agent should respond
                if self.conversation.current_turn == "left":
                    agent = self.left_agent
                    agent_name = self.left_display_name
                    next_turn = "right"
                else:
                    agent = self.right_agent
                    agent_name = self.right_display_name
                    next_turn = "left"

                # Get context for this agent
                messages = self.conversation.get_messages_for_agent(agent.name, agent.config)

                # Skip if no messages to respond to
                if len(messages) <= 1:  # Only system prompt
                    if not initial_prompt:
                        yield self.format_conversation()
                        break

                # Generate response
                response_text = ""
                yield self.format_conversation() + f"\n\n**{agent_name} is typing...**"

                for chunk in agent.generate_streaming(messages):
                    if self.conversation.stop_event.is_set():
                        break
                    response_text += chunk
                    yield self.format_conversation_with_partial(agent_name, response_text)

                # Add complete message to conversation
                if response_text and not self.conversation.stop_event.is_set():
                    self.conversation.add_message("assistant", response_text, agent.name)
                    self.conversation.current_turn = next_turn
                    yield self.format_conversation()

                    # Small delay between turns
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                yield self.format_conversation() + f"\n\n**Error: {str(e)}**"
                break

        self.conversation.is_running = False
        yield self.format_conversation() + "\n\n**Conversation stopped.**"

    def format_conversation(self) -> str:
        """Format conversation history for display"""
        if not self.conversation.messages:
            return "*No messages yet. Start the conversation or inject a prompt.*"

        formatted = []
        for msg in self.conversation.messages:
            # Use display names for agents
            if msg["agent"] == "left":
                agent_label = f"🔵 {self.left_display_name}"
            elif msg["agent"] == "right":
                agent_label = f"🔴 {self.right_display_name}"
            elif msg["agent"] == "user":
                agent_label = "👤 User"
            else:
                agent_label = msg["agent"]

            formatted.append(f"**{agent_label}:**\n{msg['content']}\n")

        return "\n---\n".join(formatted)

    def format_conversation_with_partial(self, agent_name: str, partial_text: str) -> str:
        """Format conversation with partial response being typed"""
        base = self.format_conversation()
        return base + f"\n---\n**{agent_name}:**\n{partial_text}"

    def start_conversation(self, left_params: Dict, right_params: Dict) -> Generator[str, None, None]:
        """Start the conversation between agents"""
        if self.conversation.is_running:
            yield "Conversation already running. Please stop it first."
            return

        self.conversation.is_running = True
        self.conversation.stop_event.clear()

        yield from self.run_conversation_loop(left_params, right_params)

    def stop_conversation(self) -> str:
        """Stop the ongoing conversation"""
        self.conversation.is_running = False
        self.conversation.stop_event.set()
        return "Stopping conversation..."

    def clear_conversation(self) -> str:
        """Clear the conversation history"""
        self.conversation.clear()
        return "*Conversation cleared. Ready to start fresh.*"

    def inject_message(self, message: str, from_side: str, left_params: Dict,
                      right_params: Dict) -> Generator[str, None, None]:
        """Inject a user message from either side"""
        if not message:
            yield self.format_conversation()
            return

        # Stop any running conversation
        self.conversation.is_running = False
        self.conversation.stop_event.set()
        time.sleep(0.5)  # Wait for thread to stop

        # Add the injected message
        self.conversation.add_message("user", message, from_side)

        # Set next turn based on injection side
        self.conversation.current_turn = "right" if from_side == "left" else "left"

        # Start conversation
        self.conversation.is_running = True
        self.conversation.stop_event.clear()

        yield from self.run_conversation_loop(left_params, right_params)

def create_ui():
    """Create the Gradio UI"""
    app = EchoApp()

    # Define context mode choices
    context_choices = [mode.value for mode in ContextMode]

    with gr.Blocks(title="Echo: Two-Agent Ollama Playground", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔄 Echo: Two-Agent Ollama Playground")
        gr.Markdown("Watch two AI agents converse with full parameter control")

        # Refresh models button at the top
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Available Models", scale=1)
            status_text = gr.Textbox(label="Status", value="Ready", interactive=False, scale=3)

        with gr.Row():
            # Left Agent Column
            with gr.Column(scale=1):
                gr.Markdown("### 🔵 Left Agent")

                left_name = gr.Textbox(
                    label="Agent Name",
                    value="Student",
                    placeholder="Enter agent name...",
                    interactive=True
                )

                left_model = gr.Dropdown(
                    choices=app.available_models,
                    value=app.available_models[0] if app.available_models else DEFAULT_MODEL,
                    label="Model",
                    interactive=True
                )

                left_context = gr.Dropdown(
                    choices=context_choices,
                    value=ContextMode.ROLLING_10.value,
                    label="Context Memory Mode",
                    interactive=True
                )

                left_system = gr.Textbox(
                    label="System Prompt",
                    value="""You are The Inquisitive Deep-Diving Student.

Your role is to learn as much as possible from a very capable teacher by asking sharp, honest questions and actively engaging with their explanations.

Core behaviors:

Be curious and fearless in asking questions: "why?", "how?", "what if…?", "can you give a concrete example?"

When the teacher explains something, paraphrase it back in your own words to check your understanding.

If something is confusing, admit confusion directly and ask for a simpler explanation or a different angle.

Test ideas with examples, edge cases, and counter-examples: "Does this still hold if…?", "What happens when…?"

Connect new ideas to prior ones: "Is this similar to what we talked about earlier regarding…?"

Dialogue style:

Be lively, engaged, and thoughtful—not flippant or silly.

Keep questions focused and specific, rather than extremely broad.

Occasionally express your thought process out loud: "I think what you're saying is X, which would mean Y… is that right?"

Show intellectual humility: you care about truth and understanding, not about "winning."

Interaction rules (for multi-agent dialog):

Assume the other chatbot is your teacher. Talk only to them.

Begin by stating what you think you understand about the topic and asking a question about the part that's fuzzy for you.

After each teacher response:

First, reflect ("So if I understand you correctly…"), then

Ask a follow-up question that digs deeper, challenges an assumption, or asks for another example.

Encourage the teacher to go step-by-step when things are complex.

Stay in character as a serious, motivated learner who wants to truly understand, not just memorize.""",
                    lines=8,
                    interactive=True
                )

                with gr.Accordion("Sampling Parameters", open=False):
                    left_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                    left_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
                    left_top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
                    left_min_p = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min P")

                with gr.Accordion("Generation Control", open=False):
                    left_max_tokens = gr.Slider(50, 4096, value=512, step=50, label="Max Tokens")
                    left_ctx_window = gr.Slider(512, 32768, value=4096, step=512, label="Context Window")
                    left_seed = gr.Number(label="Seed (optional)", value=None)
                    left_stop_seq = gr.Textbox(label="Stop Sequences (comma-separated)", value="")

                with gr.Accordion("Repetition Control", open=False):
                    left_repeat_penalty = gr.Slider(0.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
                    left_repeat_last_n = gr.Slider(0, 256, value=64, step=16, label="Repeat Last N")
                    left_presence_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Presence Penalty")
                    left_frequency_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Frequency Penalty")

                with gr.Accordion("Advanced Sampling", open=False):
                    left_mirostat = gr.Radio([0, 1, 2], value=0, label="Mirostat")
                    left_mirostat_tau = gr.Slider(0.0, 10.0, value=5.0, step=0.5, label="Mirostat Tau")
                    left_mirostat_eta = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Mirostat Eta")
                    left_tfs_z = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="TFS Z")

                gr.Markdown("### User Injection")
                with gr.Row():
                    left_inject = gr.Textbox(label="Inject as Left", placeholder="Type message...", scale=3)
                    left_inject_btn = gr.Button("Send →", scale=1)

            # Center Conversation Column
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                conversation_display = gr.Markdown(
                    value="*No messages yet. Start the conversation or inject a prompt.*",
                    elem_id="conversation",
                    max_height="70vh",
                    container=True
                )

                with gr.Row():
                    start_btn = gr.Button("▶️ Start Conversation", variant="primary")
                    stop_btn = gr.Button("⏸️ Stop", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear", variant="stop")

            # Right Agent Column
            with gr.Column(scale=1):
                gr.Markdown("### 🔴 Right Agent")

                right_name = gr.Textbox(
                    label="Agent Name",
                    value="Teacher",
                    placeholder="Enter agent name...",
                    interactive=True
                )

                right_model = gr.Dropdown(
                    choices=app.available_models,
                    value=app.available_models[0] if app.available_models else DEFAULT_MODEL,
                    label="Model",
                    interactive=True
                )

                right_context = gr.Dropdown(
                    choices=context_choices,
                    value=ContextMode.ROLLING_10.value,
                    label="Context Memory Mode",
                    interactive=True
                )

                right_system = gr.Textbox(
                    label="System Prompt",
                    value="""You are The Patient Master Teacher.

Your role is to help a single curious student develop deep, transferable understanding of complex ideas. You are calm, kind, and endlessly patient.

Core behaviors:

Use Socratic questioning: ask thoughtful questions rather than immediately giving full answers. Nudge the student to think, hypothesize, and correct themselves.

Explain concepts using simple language first, then gradually introduce more precise or technical vocabulary.

Use analogies, concrete examples, and mini thought experiments to make abstract ideas intuitive.

Frequently check for understanding: ask the student to restate ideas in their own words, or to give an example.

When the student is confused, slow down and go one step simpler, never condescending or impatient.

Dialogue style:

Speak directly to the student ("you") in a friendly, conversational tone.

Keep responses focused, but not terse. Aim for depth over breadth.

When you disagree or need to correct the student, do so gently and constructively, highlighting what they got right before addressing misconceptions.

Periodically summarize the current insight: "So far, we've learned that…"

Interaction rules (for multi-agent dialog):

Assume the other chatbot is the student. Engage only with them, not with any external user.

Start by asking the student what they are most curious about or what they think they understand already.

After each of the student's messages, either:

Ask a probing question, or

Give a concise explanation followed by a probing question.

Avoid long monologues. Alternate between short explanations and questions to keep the dialogue dynamic.

Stay in character as a wise, encouraging mentor who cares more about the student's thinking process than about being right.""",
                    lines=8,
                    interactive=True
                )

                with gr.Accordion("Sampling Parameters", open=False):
                    right_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                    right_top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P")
                    right_top_k = gr.Slider(1, 100, value=40, step=1, label="Top K")
                    right_min_p = gr.Slider(0.0, 1.0, value=0.05, step=0.01, label="Min P")

                with gr.Accordion("Generation Control", open=False):
                    right_max_tokens = gr.Slider(50, 4096, value=512, step=50, label="Max Tokens")
                    right_ctx_window = gr.Slider(512, 32768, value=4096, step=512, label="Context Window")
                    right_seed = gr.Number(label="Seed (optional)", value=None)
                    right_stop_seq = gr.Textbox(label="Stop Sequences (comma-separated)", value="")

                with gr.Accordion("Repetition Control", open=False):
                    right_repeat_penalty = gr.Slider(0.0, 2.0, value=1.1, step=0.05, label="Repeat Penalty")
                    right_repeat_last_n = gr.Slider(0, 256, value=64, step=16, label="Repeat Last N")
                    right_presence_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Presence Penalty")
                    right_frequency_penalty = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Frequency Penalty")

                with gr.Accordion("Advanced Sampling", open=False):
                    right_mirostat = gr.Radio([0, 1, 2], value=0, label="Mirostat")
                    right_mirostat_tau = gr.Slider(0.0, 10.0, value=5.0, step=0.5, label="Mirostat Tau")
                    right_mirostat_eta = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Mirostat Eta")
                    right_tfs_z = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="TFS Z")

                gr.Markdown("### User Injection")
                with gr.Row():
                    right_inject_btn = gr.Button("← Send", scale=1)
                    right_inject = gr.Textbox(label="Inject as Right", placeholder="Type message...", scale=3)

        # Helper function to collect parameters
        def get_left_params():
            return {
                'name': left_name,
                'model': left_model,
                'system_prompt': left_system,
                'context_mode': left_context,
                'temperature': left_temp,
                'top_p': left_top_p,
                'top_k': left_top_k,
                'min_p': left_min_p,
                'max_tokens': left_max_tokens,
                'context_window': left_ctx_window,
                'seed': left_seed,
                'stop_sequences': left_stop_seq,
                'repeat_penalty': left_repeat_penalty,
                'repeat_last_n': left_repeat_last_n,
                'presence_penalty': left_presence_penalty,
                'frequency_penalty': left_frequency_penalty,
                'mirostat': left_mirostat,
                'mirostat_tau': left_mirostat_tau,
                'mirostat_eta': left_mirostat_eta,
                'tfs_z': left_tfs_z
            }

        def get_right_params():
            return {
                'name': right_name,
                'model': right_model,
                'system_prompt': right_system,
                'context_mode': right_context,
                'temperature': right_temp,
                'top_p': right_top_p,
                'top_k': right_top_k,
                'min_p': right_min_p,
                'max_tokens': right_max_tokens,
                'context_window': right_ctx_window,
                'seed': right_seed,
                'stop_sequences': right_stop_seq,
                'repeat_penalty': right_repeat_penalty,
                'repeat_last_n': right_repeat_last_n,
                'presence_penalty': right_presence_penalty,
                'frequency_penalty': right_frequency_penalty,
                'mirostat': right_mirostat,
                'mirostat_tau': right_mirostat_tau,
                'mirostat_eta': right_mirostat_eta,
                'tfs_z': right_tfs_z
            }

        # Get all parameter inputs as list
        left_params = [left_name, left_model, left_system, left_context, left_temp, left_top_p, left_top_k, left_min_p,
                      left_max_tokens, left_ctx_window, left_seed, left_stop_seq,
                      left_repeat_penalty, left_repeat_last_n, left_presence_penalty, left_frequency_penalty,
                      left_mirostat, left_mirostat_tau, left_mirostat_eta, left_tfs_z]

        right_params = [right_name, right_model, right_system, right_context, right_temp, right_top_p, right_top_k, right_min_p,
                       right_max_tokens, right_ctx_window, right_seed, right_stop_seq,
                       right_repeat_penalty, right_repeat_last_n, right_presence_penalty, right_frequency_penalty,
                       right_mirostat, right_mirostat_tau, right_mirostat_eta, right_tfs_z]

        # Event handlers
        def refresh_models_handler():
            models = app.refresh_models()
            return [
                gr.update(choices=models),
                gr.update(choices=models),
                f"Refreshed: Found {len(models)} models"
            ]

        def start_handler(*args):
            left_p = dict(zip(get_left_params().keys(), args[:20]))
            right_p = dict(zip(get_right_params().keys(), args[20:]))
            yield from app.start_conversation(left_p, right_p)

        def inject_left_handler(message, *args):
            if not message:
                yield app.format_conversation()
                return
            left_p = dict(zip(get_left_params().keys(), args[:20]))
            right_p = dict(zip(get_right_params().keys(), args[20:]))
            yield from app.inject_message(message, "left", left_p, right_p)

        def inject_right_handler(message, *args):
            if not message:
                yield app.format_conversation()
                return
            left_p = dict(zip(get_left_params().keys(), args[:20]))
            right_p = dict(zip(get_right_params().keys(), args[20:]))
            yield from app.inject_message(message, "right", left_p, right_p)

        # Wire up events
        refresh_btn.click(
            refresh_models_handler,
            outputs=[left_model, right_model, status_text]
        )

        start_btn.click(
            start_handler,
            inputs=left_params + right_params,
            outputs=conversation_display
        )

        stop_btn.click(
            app.stop_conversation,
            outputs=status_text
        )

        clear_btn.click(
            app.clear_conversation,
            outputs=conversation_display
        )

        left_inject_btn.click(
            inject_left_handler,
            inputs=[left_inject] + left_params + right_params,
            outputs=conversation_display
        ).then(
            lambda: "",
            outputs=left_inject
        )

        right_inject_btn.click(
            inject_right_handler,
            inputs=[right_inject] + left_params + right_params,
            outputs=conversation_display
        ).then(
            lambda: "",
            outputs=right_inject
        )

        # Also allow Enter key to send messages
        left_inject.submit(
            inject_left_handler,
            inputs=[left_inject] + left_params + right_params,
            outputs=conversation_display
        ).then(
            lambda: "",
            outputs=left_inject
        )

        right_inject.submit(
            inject_right_handler,
            inputs=[right_inject] + left_params + right_params,
            outputs=conversation_display
        ).then(
            lambda: "",
            outputs=right_inject
        )

    return demo

if __name__ == "__main__":
    print("Starting Echo: Two-Agent Ollama Playground...")
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL}")
    print("Make sure Ollama is running locally!")
    print("-" * 50)

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )