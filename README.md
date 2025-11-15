# Echo: Two-Agent Ollama Playground

A sophisticated Gradio application that allows you to orchestrate conversations between two Ollama AI models with fine-grained control over their behavior and parameters.

## Features

- **Dual Agent System**: Run two independent Ollama models simultaneously
- **Streaming Responses**: Watch responses generate in real-time
- **Context Memory Control**: Individual memory management for each agent
  - No Memory (System prompt + last message only)
  - Last Exchange Only
  - Rolling Windows (2, 5, 10, 20, 50 messages)
  - Full History
- **Comprehensive Parameter Control**: Full access to all Ollama generation parameters
- **User Injection**: Inject messages from either side at any time
- **Strict Alternation**: Agents take turns responding in order
- **Live Configuration**: Adjust parameters on the fly without restarting

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running locally
3. At least one Ollama model pulled (e.g., `ollama pull llama3.2`)

### Installing Ollama

```bash
# On Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On macOS
brew install ollama

# Start Ollama server
ollama serve
```

### Pulling Models

```bash
# Pull some models to use
ollama pull llama3.2
ollama pull mistral
ollama pull qwen2.5
ollama pull phi3
```

## Installation

1. Clone or navigate to the Echo directory:
```bash
cd /home/xadmin/Documents/Echo
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure Ollama is running:
```bash
ollama serve  # In a separate terminal
```

2. Run the application:
```bash
python echo_app.py
```

3. Open your browser to `http://localhost:7860`

## User Interface Guide

### Layout

The UI is divided into three main columns:

#### Left Agent Column
- **Model Selection**: Choose the model for the left agent
- **Context Memory Mode**: Select how much conversation history this agent remembers
- **System Prompt**: Define the agent's personality and behavior
- **Parameter Controls** (in accordions):
  - Sampling Parameters
  - Generation Control
  - Repetition Control
  - Advanced Sampling
- **User Injection**: Send messages as the left agent

#### Center Column
- **Conversation Display**: Shows the ongoing conversation
- **Control Buttons**:
  - Start Conversation: Begin the agent dialogue
  - Stop: Pause the conversation
  - Clear: Reset the conversation history
- **Turn Indicator**: Shows which agent should respond next

#### Right Agent Column
- Mirror of the left column for the right agent
- Independent configuration from the left agent

### Starting a Conversation

1. **Configure Both Agents**:
   - Select models for each agent
   - Set their system prompts to define personalities
   - Adjust context memory modes
   - Tweak generation parameters as desired

2. **Initiate Dialogue**:
   - Click "Start Conversation" to begin
   - Or inject an initial message from either side

3. **Watch the Conversation**:
   - Agents will alternate turns automatically
   - Responses stream in real-time
   - The conversation continues until you stop it

### Context Memory Modes

Each agent can have different memory settings:

- **No Memory**: Agent only sees its system prompt and the immediate message from the other agent
- **Last Exchange Only**: Remembers only the most recent back-and-forth
- **Rolling Window**: Keeps the last N messages in context
- **Full History**: Maintains complete conversation (watch token limits!)

### User Injection

You can inject messages from either side at any time:
1. Type your message in the injection text box
2. Click "Send" or press Enter
3. The conversation will continue from that point

### Parameter Guide

#### Sampling Parameters
- **Temperature** (0.0-2.0): Controls randomness. Lower = more focused, Higher = more creative
- **Top P** (0.0-1.0): Nucleus sampling threshold
- **Top K** (1-100): Number of top tokens to consider
- **Min P** (0.0-1.0): Minimum probability threshold

#### Generation Control
- **Max Tokens**: Maximum response length
- **Context Window**: Total context size in tokens
- **Seed**: For reproducible outputs
- **Stop Sequences**: Strings that halt generation

#### Repetition Control
- **Repeat Penalty**: Penalizes repeated tokens
- **Repeat Last N**: Look-back window for repetition checking
- **Presence/Frequency Penalty**: Additional repetition controls

#### Advanced Sampling
- **Mirostat**: Alternative sampling method (0=off, 1=v1, 2=v2)
- **Mirostat Tau/Eta**: Mirostat parameters
- **TFS Z**: Tail-free sampling parameter

## Configuration File

The `config.yaml` file contains default settings. The application will use these as starting values but you can override them in the UI.

## Tips and Best Practices

### Creating Interesting Conversations

1. **Contrasting Personalities**: Give agents different perspectives
   - Creative vs. Analytical
   - Optimistic vs. Cautious
   - Expert vs. Student

2. **Memory Experiments**: Try different context modes
   - No Memory agents create surreal conversations
   - Full History maintains coherent long discussions
   - Rolling Windows balance coherence and surprise

3. **Parameter Tuning**:
   - Higher temperature for creative agents
   - Lower temperature for analytical agents
   - Adjust top_p and top_k together for best results

### Performance Optimization

1. **Model Selection**: Smaller models respond faster
2. **Token Limits**: Keep max_tokens reasonable (512-1024)
3. **Context Windows**: Larger contexts use more memory
4. **Keep Alive**: Models stay loaded for 5 minutes by default

### Troubleshooting

**"Connection refused" error**:
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is available

**"Model not found" error**:
- Pull the model first: `ollama pull model-name`
- Click "Refresh Available Models" button

**Slow responses**:
- Check if model is loading (first request is slower)
- Try a smaller model
- Reduce max_tokens and context_window

**Conversation stops unexpectedly**:
- Check for stop sequences in generated text
- Verify token limits aren't reached
- Look for error messages in the terminal

## Advanced Usage

### Running Multiple Instances

You can run multiple Echo instances on different ports:

```bash
# Terminal 1
python echo_app.py --port 7860

# Terminal 2 (modify the code or use different config)
python echo_app.py --port 7861
```

### Using Different Ollama Endpoints

If Ollama is running on a different machine or port, modify the `OLLAMA_BASE_URL` in `echo_app.py`:

```python
OLLAMA_BASE_URL = "http://192.168.1.100:11434"  # Remote Ollama
```

### Custom Stop Sequences

For model-specific stop tokens:
- Llama models: `<|eot_id|>`
- ChatML format: `<|im_end|>`
- Alpaca format: `### Response:`

## Model Recommendations

### For Creative Writing
- Models: llama3.2, mistral, mixtral
- Temperature: 0.8-1.2
- Top P: 0.95
- Context: Full History or Rolling Window (20+)

### For Technical Discussions
- Models: codellama, deepseek-coder, qwen2.5-coder
- Temperature: 0.3-0.5
- Top P: 0.7
- Repeat Penalty: 1.2

### For Rapid Back-and-Forth
- Models: phi3, tinyllama, qwen2.5:0.5b
- Max Tokens: 200-300
- Context: Rolling Window (5-10)

### For Philosophical Debates
- Models: llama3.2, yi, solar
- Temperature: 0.7
- Context: Full History
- System Prompts: Give opposing philosophical stances

## Contributing

Feel free to enhance Echo with:
- Additional context memory modes
- Preset management UI
- Conversation export/import
- Multi-agent support (3+ agents)
- Custom UI themes
- Token usage tracking

## License

This project is provided as-is for educational and experimental purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure Ollama is properly installed and running
3. Verify you have at least one model pulled
4. Check the terminal output for error messages

Enjoy orchestrating AI conversations with Echo!