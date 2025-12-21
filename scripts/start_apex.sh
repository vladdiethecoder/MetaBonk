#!/bin/bash
# =============================================================================
# MetaBonk Apex - Quick Start Script
# =============================================================================
#
# Usage:
#   ./start_apex.sh train     # Run full training pipeline
#   ./start_apex.sh agent     # Run the trained agent
#   ./start_apex.sh menu      # Run menu navigator only
#   ./start_apex.sh install   # Install dependencies
#
# =============================================================================

set -e
cd "$(dirname "$0")/.."

COMMAND=${1:-help}
DEVICE="cuda"

# Check if CUDA is available
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cpu"
    echo "Note: CUDA not available, using CPU"
fi

case $COMMAND in
    install)
        echo "Installing dependencies..."
        pip install torch torchvision numpy gymnasium httpx pydantic pillow tqdm rich mss
        
        # Check for Ollama
        if command -v ollama &> /dev/null; then
            echo "Ollama found, pulling llava model..."
            ollama pull llava || echo "Ollama pull failed (optional)"
        else
            echo "Ollama not installed. For VLM features, install from: https://ollama.com"
        fi
        
        echo "Dependencies installed!"
        ;;
    
    train)
        echo "Starting Apex Training Pipeline..."
        python scripts/train_apex.py --phase all --epochs 20 --generate-data --device $DEVICE
        ;;
    
    agent)
        echo "Starting Apex Agent..."
        python scripts/run_agent.py --mode autonomous --device $DEVICE
        ;;
    
    menu)
        echo "Starting Menu Navigator..."
        python scripts/menu_navigator.py --goal "Start Game" --vlm ollama
        ;;
    
    test)
        echo "Running import tests..."
        python -c "
from src.learner.world_model import WorldModel
from src.learner.liquid_networks import NeuralCircuitPolicy
from src.learner.skill_tokens import SkillVQVAE
from src.learner.decision_transformer import DecisionTransformer
from src.learner.free_energy import FreeEnergyObjective
from src.learner.causal_rl import CausalGraph
from src.learner.interpretability import SparseAutoencoder
from src.orchestrator.eureka import EurekaRewardGenerator
from src.orchestrator.ued import PLRCurriculum
print('All imports successful!')
"
        ;;
    
    help|*)
        echo "MetaBonk Apex - NSAIG Architecture"
        echo ""
        echo "Commands:"
        echo "  install   Install Python dependencies"
        echo "  train     Run full training pipeline (World Model, Skills, etc.)"
        echo "  agent     Run trained agent in autonomous mode"
        echo "  menu      Run VLM menu navigator"
        echo "  test      Test all module imports"
        echo ""
        echo "Example:"
        echo "  ./scripts/start_apex.sh install"
        echo "  ./scripts/start_apex.sh train"
        echo "  ./scripts/start_apex.sh agent"
        ;;
esac
