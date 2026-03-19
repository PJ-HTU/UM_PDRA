# UM_PDRA - Unified Model for Post-Disaster Road Assessment Drone Routing

This repository implements the **Unified Model (UM)** proposed in the paper *"A Unified Model for Multi-Task Drone Routing in Post-Disaster Road Assessment"*, aiming to solve multi-task drone routing optimization for post-disaster road damage assessment.

## Problem Statement

Given a road network affected by disaster, deploy a fleet of drones to:
- Maximize collected damage information across the road network
- Complete assessment within time constraints
- Respect battery flight time limits

## Problem Variants

The problem can be extended to 8 PDRA variants through attribute combinations:

| Variant | Open Route (OR) | Time Window (TW) | Multi-Depot (MD) |
|:-------:|:--------------:|:----------------:|:----------------:|
| PDRA-Basic |   |   |   |
| PDRA-OR | ✓ |   |   |
| PDRA-TW |   | ✓ |   |
| PDRA-OR-TW | ✓ | ✓ |   |
| PDRA-MD |   |   | ✓ |
| PDRA-OR-MD | ✓ |   | ✓ |
| PDRA-TW-MD |   | ✓ | ✓ |
| PDRA-OR-TW-MD | ✓ | ✓ | ✓ |

**Attribute Descriptions:**
- **OR (Open Route)**: Drones do not need to return to depot, can land at any feasible location
- **TW (Time Window)**: Critical road segments must be assessed before specified deadlines
- **MD (Multi-Depot)**: Drones can launch from multiple depot locations

## Model Architecture

![UM Architecture](https://raw.githubusercontent.com/PJ-HTU/UM_PDRA/main/Model%20Architecture.jpg)

The unified model processes road network data, problem parameters, and variant attributes through five main components: 
- (1) **Input** layer embeds node features and problem configurations, distinguishing road network nodes and depot nodes; 
- (2) **Encoder** transforms embeddings into high-level contextual representations using modern transformer layers with RMS normalization, FlashAttention, and SGLUFFN; 
- (3) **Decoder** constructs solutions autoregressively through probability computation, softmax selection, and single-head attention mechanisms; 
- (4) **Update** module maintains solution feasibility by managing infeasible action masks, active drone indices, and current time states; 
- (5) **Output** produces feasible multi-drone routes satisfying variant-specific constraints with appropriate termination conditions.

## Key Features

- **Rapid Performance**: 1-10 seconds inference time vs. 100-2,000 seconds for traditional methods
- **Superior Solution Quality**: Outperforms commercial solvers (Gurobi) and traditional heuristics
- **No Domain Expertise Required**: Eliminates need for hand-crafted algorithms through end-to-end learning
- **Strong Generalization**: Robust performance across varying problem scales, drone numbers, and time constraints
- **Multi-task Learning**: Handles diverse parameter combinations in a unified framework
- **8× Efficiency**: Consolidates 8 PDRA variants in one model, reducing training time and parameters by 8-fold
- **Adaptive to New Attributes**: Lightweight adapter mechanism enables efficient incorporation of unseen attributes (e.g., multi-depot settings) without full retraining

## Technical Highlights

### Key Innovations

1. **Network Transformation**: Converts link-based routing problems (assessing road segments) into node-based formulations to eliminate ambiguity and reduce computational complexity.

2. **Synthetic Data Generation**: Addresses large-scale training dataset scarcity by generating realistic road network instances (grid initialization → link pruning → node perturbation).

3. **Attention-based Encoder-Decoder**: Uses Transformer architecture to learn optimal routing strategies end-to-end via deep reinforcement learning (DRL).

4. **Multi-task Learning**: Handles simultaneous training across varying drone numbers and time constraints, eliminating the need for separate models per parameter combination.

5. **Modern Transformer Architecture**: Incorporates RMS normalization, pre-normalization configuration, FlashAttention, and SGLUFFN for enhanced computational efficiency and solution quality.

6. **Lightweight Finetuning**: Adapter mechanism enables efficient incorporation of new attributes (e.g., multi-depot) with minimal computational cost while preserving pre-trained knowledge.

## Quick Start

### Dependencies
```bash
# Python environment requirements
Python 3.8+
PyTorch 1.10+
NumPy
Pandas
Matplotlib
Scipy
```

### Installation
```bash
git clone https://github.com/PJ-HTU/UM_PDRA.git
cd UM_PDRA
pip install -r requirements.txt
```

### Training
```bash
# Train on 100-node synthetic road network instances
python train_n100.py --epochs 200 --batch_size 64 --embedding_dim 128
```

**Training Parameters:**
- `epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 64)
- `embedding_dim`: Embedding dimension (default: 128)

### Testing
```bash
# Test on custom road network instances (supports up to 1000 nodes)
python test_n100.py --model_path ./checkpoints/um_best.pth --augmentation 8
```

**Testing Parameters:**
- `model_path`: Path to pre-trained model
- `augmentation`: Instance augmentation factor (improves solution diversity through coordinate flipping/swapping)

## Repository Structure
```
UM_PDRA/
├── UM/                          # Core framework directory containing all model and task logic
│   ├── PDRA/                    # Post-Disaster Road Assessment (PDRA) task module
│   │   ├── Unified_model/       # Implementation of the unified Transformer-based model and training pipeline
│   │   │   ├── train_n100.py    # Primary training script for PDRA problems with ~100 nodes
│   │   │   │                    # Configures training parameters (epochs, batch size, model hyperparameters)
│   │   │   │                    # Initializes environment, model, and trainer; executes training loop
│   │   │   ├── test_n100.py     # Testing and evaluation script for ~100-node PDRA instances
│   │   │   │                    # Loads pre-trained models, runs inference, and computes performance metrics
│   │   │   │                    # Supports data augmentation (8-fold geometric transformations) for robust testing
│   │   │   ├── PDRAEnv.py       # PDRA environment class: manages state transitions and constraints
│   │   │   │                    # Handles node/vehicle state tracking (load, time, visited nodes)
│   │   │   │                    # Implements masking for invalid actions (e.g., exceeding time windows/load limits)
│   │   │   │                    # Supports batch processing and POMO (Policy Optimization with Multiple Optima)
│   │   │   ├── PDRAModel.py     # Unified Transformer model architecture for PDRA routing
│   │   │   │                    # Encoder: processes node features (coordinates, demand, time windows) via multi-head attention
│   │   │   │                    # Decoder: generates next-node predictions with state-aware attention
│   │   │   │                    # Includes RMS normalization and SwigLU activation for stable training
│   │   │   ├── PDRATrainer.py   # Training logic for the unified model
│   │   │   │                    # Implements reward-based optimization with EMA-Z-score normalization
│   │   │   │                    # Handles dynamic vehicle configuration sampling (drone count/capacity)
│   │   │   │                    # Logs training metrics (loss, reward) and saves model checkpoints
│   │   │   └── PDRATester.py    # Evaluation module for model performance
│   │   │                        # Computes solution quality (collected information value, path length)
│   │   │                        # Supports both augmented and non-augmented testing modes
│   │   │                        # Outputs detailed route information and comparative metrics
│   │   └── PDRAProblemDef.py    # PDRA problem definition and instance generation
│   │                           # Creates synthetic road networks (grid-based or real-world derived)
│   │                           # Defines constraints for different PDRA variants (TW/OR/MD)
│   │                           # Implements data augmentation via geometric transformations
│   └── utils/                   # Utility functions and helper modules
│       ├── utils.py             # General-purpose utilities for logging, metrics, and visualization
│       │                        # Includes LogData class for tracking training/testing metrics
│       │                        # Provides time estimation, distance calculation, and result saving
│       │                        # Supports log visualization (plotting loss/reward curves)
│       └── log_image_style/     # Configuration for log visualization styling
│           └── style_PDRA_20.json # Defines plot aesthetics (axes, colors, grids) for consistent visualization
├── checkpoints/                 # Storage directory for trained model checkpoints
│                               # Saves model weights, optimizer states, and training configurations
│                               # Organized by training date/problem type for easy retrieval
├── results/                     # Output directory for training logs and evaluation results
│                               # Contains CSV logs of metrics (loss, reward, inference time)
│                               # Stores visualization plots (training curves, route examples)
│                               # Includes test result summaries (average performance, best paths)
├── Model Architecture.pdf       # Documentation of the unified model's architecture
│                               # Details encoder/decoder design, attention mechanisms, and state integration
│                               # Includes flowcharts of environment-model interaction
├── README.md                    # Repository overview, usage instructions, and documentation
└── requirements.txt             # List of dependencies (PyTorch, NumPy, NetworkX, Matplotlib)
```

## Citation

If you use this code or model in your research, please cite the paper:
```bibtex
@article{gong2025unified,
  title={A Unified Model for Multi-Task Drone Routing in Post-Disaster Road Assessment},
  author={Gong, Huatian and Sheu, Jiuh-Biing and Wang, Zheng and Yang, Xiaoguang and Yan, Ran},
  journal={arXiv preprint arXiv:2510.21525},
  year={2025}
}
```



## Acknowledgements

💡 Our code builds on [POMO](https://github.com/yd-kwon/POMO), [AEDM](https://github.com/PJ-HTU/AEDM-for-Post-disaster-road-assessment), [routefinder](https://github.com/ai4co/routefinder). Big thanks!

