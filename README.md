# negotiation-delegation
AI delegation for real-world negotiation scenarios. SPI 352 (Fall 2025) final project.

## Overview

This project assesses biases in language model agents for real estate tasks through controlled experiments. We test whether agents discriminate against buyers and sellers based on protected characteristics (race, gender) in price recommendations and negotiations.

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create virtual environment**:
   ```bash
   uv venv venv-spi352
   source venv-spi352/bin/activate  # On macOS/Linux
   # or
   venv-spi352\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

4. **Set up API keys**:
   Create a `.env` file in the project root with your API keys:
   ```
   ANTHROPIC_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here  # Optional, if using GPT models
   ```

## Configuration

Edit `config/experiments.yaml` to configure:
- Model selection (provider, model name)
- Experiment parameters (sample sizes, max proposals)
- House specifications
- Buyer/seller profiles
- Statistical test settings

## Running Experiments

### Run both experiments:
```bash
python main.py
```

### Run specific experiment:
```bash
# Experiment 1 only (MVP price recommendation)
python main.py --experiment 1

# Experiment 2 only (full negotiation)
python main.py --experiment 2
```

### Run with statistical analysis:
```bash
python main.py --analyze
```

### Custom configuration:
```bash
python main.py --config path/to/config.yaml --experiment 1 --analyze
```

## Experiments

### Experiment 1: MVP Price Recommendation
Tests whether LLMs recommend different starting prices based on buyer/seller characteristics (name, race, gender). Each trial makes a single LLM call asking for a price recommendation.

### Experiment 2: Full Negotiation
Simulates full negotiations between buyer and seller agents with:
- Think/reflect steps (internal reasoning)
- Discussion steps (messages between agents)
- Proposal steps (price offers/acceptances)
- Configurable max proposals (default: 10 total = 5 buyer + 5 seller)

Supports both single-buyer and multi-buyer scenarios.

## Results

Results are saved to the `results/` directory:
- `experiment1_*.csv`: Price recommendation results
- `experiment2_*.csv`: Negotiation results
- `analysis_*.json`: Statistical analysis results (if `--analyze` flag used)

## Project Structure

```
negotiation-delegation/
├── config/              # Experiment configurations
├── src/
│   ├── models/         # LLM client abstraction
│   ├── experiments/    # Experiment implementations
│   ├── agents/         # Buyer/seller agents
│   ├── data/           # Profile and house spec generation
│   ├── negotiation/    # Negotiation engines
│   ├── analysis/       # Statistical analysis
│   └── utils/          # Utilities (config loader, results writer)
├── results/            # Output directory
└── main.py            # Entry point
```

## Supported Models

- **Anthropic**: Claude 3.7 Sonnet (default), other Claude models
- **OpenAI**: GPT-4o, GPT-5, GPT-5 Mini
- **Open-source**: Qwen, Olmo, Llama, Kimi (via OpenAI-compatible API)

Each experiment run uses the same model (no mixing models in one negotiation).
