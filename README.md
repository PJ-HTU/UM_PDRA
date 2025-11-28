# UM_PDRA - Unified Model for Post-Disaster Road Assessment Drone Routing

This repository implements the **Unified Model (UM)** proposed in the paper *"A Unified Model for Multi-Task Drone Routing in Post-Disaster Road Assessment"*, aiming to solve multi-task drone routing optimization for post-disaster road damage assessment.

## Problem Statement

Given a road network affected by disaster, deploy a fleet of drones to:
- Maximize collected damage information across the road network
- Complete assessment within time constraints
- Respect battery flight time limits

## Model Architecture

![UM Architecture](https://raw.githubusercontent.com/PJ-HTU/UM_PDRA/main/Model%20Architecture.jgp)

The model processes road network coordinates, drone parameters, and constraints through:
- **Encoder**: Embeds node features and global parameters (drone count, time limits) via multi-head attention layers with modern Transformer components (RMS normalization, FlashAttention, SGLUFFN).
- **Decoder**: Sequentially constructs feasible routes with masking mechanisms to enforce time/battery constraints and avoid redundant assessments.

## Key Features

- **Rapid Performance**: 1-10 seconds inference time vs. 100-2,000 seconds for traditional methods
- **Superior Solution Quality**: Outperforms commercial solvers (Gurobi) and traditional heuristics
- **No Domain Expertise Required**: Eliminates need for hand-crafted algorithms through end-to-end learning
- **Strong Generalization**: Robust performance across varying problem scales, drone numbers, and time constraints
- **Multi-task Learning**: Handles diverse parameter combinations in a unified framework
- **8× Efficiency**: Consolidates 8 PDRA variants in one model, reducing training time and parameters by 8-fold
- **Adaptive to New Attributes**: Lightweight adapter mechanism enables efficient incorporation of unseen attributes (e.g., multi-depot settings) without full retraining

## Problem Variants

The model supports 8 PDRA variants through attribute combinations:

| Variant | Open Route (OR) | Time Window (TW) | Multi-Depot (MD) |
|---------|----------------|------------------|------------------|
| PDRA-Basic | - | - | - |
| PDRA-OR | ✓ | - | - |
| PDRA-TW | - | ✓ | - |
| PDRA-OR-TW | ✓ | ✓ | - |
| PDRA-MD | - | - | ✓ |
| PDRA-OR-MD | ✓ | - | ✓ |
| PDRA-TW-MD | - | ✓ | ✓ |
| PDRA-OR-TW-MD | ✓ | ✓ | ✓ |

**Attribute Descriptions:**
- **OR (Open Route)**: Drones do not need to return to depot, can land at any feasible location
- **TW (Time Window)**: Critical road segments must be assessed before specified deadlines
- **MD (Multi-Depot)**: Drones can launch from multiple depot locations

## Technical Highlights

### Key Innovations

1. **Network Transformation**: Converts link-based routing problems (assessing road segments) into node-based formulations to eliminate ambiguity and reduce computational complexity.

2. **Synthetic Data Generation**: Addresses large-scale training dataset scarcity by generating realistic road network instances (grid initialization → link pruning → node perturbation).

3. **Attention-based Encoder-Decoder**: Uses Transformer architecture to learn optimal routing strategies end-to-end via deep reinforcement learning (DRL).

4. **Multi-task Learning**: Handles simultaneous training across varying drone numbers and time constraints, eliminating the need for separate models per parameter combination.

5. **Modern Transformer Architecture**: Incorporates RMS normalization, pre-normalization configuration, FlashAttention, and SGLUFFN for enhanced computational efficiency and solution quality.

6. **Lightweight Finetuning**: Adapter mechanism enables efficient incorporation of new attributes (e.g., multi-depot) with minimal computational cost while preserving pre-trained knowledge.

## Performance Advantages

Compared to existing methods, UM achieves significant improvements:
- **vs. Single-task DRL**: 6-14% better solution quality
- **vs. Heuristic Algorithms**: 22-42% better solution quality
- **vs. Commercial Solvers**: 24-82% better solution quality
- **Computational Efficiency**: 1-10 seconds inference time for networks up to 1,000 nodes

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
- Training time: ~24 hours (on NVIDIA A100 GPU)

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
├── AEDM/                     # Core code directory (implements all model & task logic)
│   ├── PDRA/                 # Post-disaster Road Assessment (PDRA) task module
│   │   ├── POMO/             # Policy Optimization with Multiple Optima (POMO) implementation
│   │   ├── PDRAEnv.py        # PDRA environment class: simulates post-disaster road network scenarios
│   │   │   - Initializes dual networks (original road network for assessment + fully connected auxiliary network for transit)
│   │   │   - Implements environment interaction: reset() (reset scenario), step() (execute drone action and update state), and time/battery constraint checks
│   │   │   - Calculates road link assessment time, transit time, and information value collection
│   │   ├── PDRAModel.py      # UM model class: defines attention-based encoder-decoder architecture
│   │   │   - Encoder: Processes node features (coordinates, information value) and global parameters (K, p_max, Q) into high-dimensional embeddings via Transformer layers
│   │   │   - Decoder: Sequentially generates drone routes using MHA, single-head attention (SHA), and masking (blocks infeasible actions)
│   │   │   - Outputs route probability distributions and ensures feasible solutions
│   │   ├── PDRATrainer.py    # Model training logic class
│   │   │   - Loads training instances (synthetic road networks) and initializes model/optimizer
│   │   │   - Implements POMO-based training: multi-optima sampling, EMA-Z-score reward normalization (stabilizes multi-task training)
│   │   │   - Tracks training metrics (loss, collected information value) and saves checkpoints
│   │   └── PDRATester.py     # Model testing logic class
│   │       - Loads pre-trained models and test instances (synthetic/real-world road networks like Anaheim)
│   │       - Evaluates model performance: calculates solution quality (collected information value), inference time
│   │       - Supports 8-fold instance augmentation (coordinate flipping/swapping) to improve solution diversity
│   └── utils/                # Auxiliary tools directory (supports core logic execution)
│       ├── utils.py          # General utility functions
│       │   - Log data management: LogData class to record training/testing metrics (loss, score, time) for visualization
│       │   - Distance calculation: Computes Euclidean distance between nodes (for transit/assessment time estimation)
│       └── log_image_style/  # Log image styling configuration
│           └── style_PDRA_20.json # Defines visualization styles
├── train_n100.py             # Training entry script (for 100-node synthetic instances)
│   - Defines hyperparameters: embedding dimension (128), encoder layers (6), batch size (64), epochs (200)
│   - Calls PDRATrainer to start training: samples synthetic instances, runs POMO training, saves checkpoints to checkpoints/
├── test_n100.py              # Testing entry script (for 100-node instances, extendable to 1000-node)
│   - Loads pre-trained models from checkpoints/ and test instances (synthetic or real-world like Anaheim)
│   - Calls PDRATester to evaluate performance: outputs inference time, collected information value
└── checkpoints/              # Pre-trained model storage directory
```

## Experimental Results

### Performance on Synthetic Networks

Comprehensive experiments on 200-node and 400-node networks demonstrate:

- **Solution Quality**: UM consistently achieves the highest objective values across all PDRA variants
- **Computational Efficiency**: Generates valid solutions within 1-2 seconds (vs. 30 minutes for commercial solvers)
- **Robustness**: Maintains stable performance under varying time limits, drone fleet sizes, and network scales

### Validation on Real-World Networks

Validation on the Anaheim transportation network (1,330 nodes) confirms practical scalability:
- Maintains superior solution quality
- Rapid inference performance (~24 seconds for all 8 variants)
- Successfully handles irregular topology of real-world road networks

### Sensitivity Analysis

Extensive sensitivity analyses across three dimensions validate model robustness:
- **Time Constraints**: Performance gaps of −0.20% to 16.07% over best single-task AEDM across varying assessment time limits (20-50 minutes)
- **Fleet Size**: Consistent superiority with 1.20%–12.79% performance gaps across 4-10 drones
- **Network Scale**: Strong scalability with 0.17%–11.65% performance gaps on networks from 400-1,000 nodes

### Finetuning Results

Lightweight adapter mechanism enables efficient incorporation of unseen attributes (Multi-Depot):
- **UM-10-Epochs** (10 epochs finetuning) achieves higher solution quality than single-task AEDMs trained from scratch (200 epochs each)
- Demonstrates effective knowledge transfer with minimal computational cost
- Validates adaptability to evolving disaster response requirements

## Citation

If you use this code or model in your research, please cite the paper:
```bibtex
@article{gong2025unified,
  title={A Unified Model for Multi-Task Drone Routing in Post-Disaster Road Assessment},
  author={Gong, Huatian and Sheu, Jiuh-Biing and Wang, Zheng and Yang, Xiaoguang and Yan, Ran},
  journal={[Journal TBD]},
  year={2025}
}
```

## Acknowledgements

💡 Our code builds on [POMO](https://github.com/yd-kwon/POMO). Big thanks!

## License

[License information to be added]

## Contact

For questions or suggestions, please:
- Submit a GitHub Issue
- [Contact email to be added]

---

**Note**: This repository is under active development. More features and documentation will be continuously updated.
