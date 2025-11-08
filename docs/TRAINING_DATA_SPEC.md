# VLA-GR Training Data Specification

## 📊 Complete Training Dataset Requirements

### 1. Primary Navigation Datasets

#### 1.1 HM3D (Habitat-Matterport 3D) Dataset
- **Size**: ~2.5TB (full), ~100GB (minival)
- **Scenes**: 1000 training, 200 validation, 200 test
- **Resolution**: High-quality 3D meshes with textures
- **Annotations**: Semantic labels for 40+ object categories
- **Usage**: Main training dataset for navigation

```bash
# Download command
python -m habitat_sim.utils.datasets_download \
    --uids hm3d_train_v0.2 hm3d_val_v0.2 \
    --data-path data/
```

#### 1.2 MP3D (Matterport3D) Dataset
- **Size**: ~500GB
- **Scenes**: 61 train, 11 val, 18 test
- **Features**: Real indoor environments, dense annotations
- **Usage**: Cross-dataset evaluation

#### 1.3 Gibson Dataset
- **Size**: ~200GB
- **Scenes**: 572 buildings
- **Features**: Real spaces, physics simulation
- **Usage**: Robustness testing

#### 1.4 Replica Dataset
- **Size**: ~50GB
- **Scenes**: 18 high-fidelity apartment/office scenes
- **Features**: Perfect geometry, high-res textures
- **Usage**: Ablation studies, controlled experiments

### 2. Language Instruction Datasets

#### 2.1 R2R (Room-to-Room) Dataset
- **Size**: 10GB
- **Instructions**: 21,567 navigation instructions
- **Avg Length**: 29 words per instruction
- **Languages**: English (extensible to multilingual)
- **Usage**: Language grounding training

```json
{
  "path_id": 1357,
  "instruction": "Walk straight through the kitchen and turn left at the dining table. Continue to the living room and stop near the blue couch.",
  "trajectory": [[x1,y1,z1], [x2,y2,z2], ...],
  "scan": "2azQ1b91cZZ"
}
```

#### 2.2 RxR (Room-across-Room) Dataset
- **Size**: 15GB
- **Instructions**: 126,069 instructions in English, Hindi, Telugu
- **Features**: Longer paths, multilingual
- **Usage**: Cross-lingual transfer

#### 2.3 ALFRED Dataset
- **Size**: 25GB
- **Tasks**: 25,743 language-guided tasks
- **Features**: High-level goals + step-by-step instructions
- **Usage**: Hierarchical planning training

### 3. Vision-Language Pre-training Data

#### 3.1 Conceptual Captions 3M (CC3M)
- **Size**: 30GB
- **Pairs**: 3.3M image-caption pairs
- **Usage**: Vision-language alignment pre-training

#### 3.2 LAION-5B (subset)
- **Size**: 100GB (filtered subset)
- **Pairs**: 10M high-quality image-text pairs
- **Usage**: Large-scale pre-training

### 4. Depth Completion Training Data

#### 4.1 NYU Depth V2
- **Size**: 10GB
- **Images**: 1449 RGB-D pairs
- **Resolution**: 640×480
- **Usage**: Depth completion network training

#### 4.2 KITTI Depth
- **Size**: 50GB
- **Images**: 93k depth maps
- **Features**: Outdoor scenes, sparse LIDAR
- **Usage**: Robustness to sparse depth

#### 4.3 ScanNet
- **Size**: 1.2TB
- **Scenes**: 1513 indoor scans
- **Features**: Dense depth, instance segmentation
- **Usage**: Indoor depth completion

### 5. Semantic Segmentation Data

#### 5.1 ADE20K
- **Size**: 5GB
- **Images**: 25k annotated images
- **Classes**: 150 semantic categories
- **Usage**: Semantic understanding training

#### 5.2 COCO-Stuff
- **Size**: 20GB
- **Images**: 164k images
- **Classes**: 171 stuff + thing categories
- **Usage**: Object detection and segmentation

---

## 🎯 Module-Specific Training Requirements

### Module 1: Perception Module
**Training Data**:
- RGB-D pairs from HM3D/MP3D (primary)
- NYU Depth V2 (depth completion)
- ImageNet (vision backbone pre-training)
- CC3M (vision-language alignment)

**Training Process**:
```python
# Stage 1: Pre-train vision encoder (optional if using DINOv2)
train_vision_encoder(imagenet_dataset, epochs=100)

# Stage 2: Train depth completion
train_depth_completion(nyu_depth + scannet, epochs=50)

# Stage 3: Fine-tune on navigation data
train_perception(hm3d_dataset, epochs=30)
```

### Module 2: Language Encoder
**Training Data**:
- R2R instructions (primary)
- RxR multilingual instructions
- ALFRED task descriptions
- General text corpus (for LLM pre-training)

**Training Process**:
```python
# Use pre-trained Phi-2 or fine-tune on navigation
finetune_language_model(r2r_instructions, epochs=10)
```

### Module 3: Affordance Quantifier
**Training Data**:
- HM3D semantic annotations
- ADE20K semantic segmentation
- Custom affordance labels (mass, friction, traversability)

**Dataset Generation**:
```python
# Generate affordance labels from semantics
affordance_labels = {
    "floor": {"mass": 1000, "friction": 0.8, "traversable": 1.0},
    "wall": {"mass": 500, "friction": 0.6, "traversable": 0.0},
    "chair": {"mass": 10, "friction": 0.7, "traversable": 0.2},
    "table": {"mass": 30, "friction": 0.7, "traversable": 0.0},
    # ... more categories
}
```

### Module 4: GR Field Manager
**Training Data**:
- Simulated physics environments
- Affordance maps from Module 3
- Ground truth geodesics from A* planning

**Synthetic Data Generation**:
```python
# Generate training data with known geodesics
def generate_gr_training_data():
    # Create random affordance distributions
    affordance_map = generate_random_affordances()
    
    # Solve for ground truth geodesic
    optimal_path = compute_geodesic(affordance_map)
    
    # Compute metric tensor
    metric_tensor = solve_einstein_equations(affordance_map)
    
    return {
        "affordance": affordance_map,
        "metric": metric_tensor,
        "geodesic": optimal_path
    }
```

### Module 5: Path Optimizer
**Training Data**:
- HM3D/MP3D navigation episodes
- Expert demonstrations (shortest paths)
- Reinforcement learning rollouts

**Training Process**:
```python
# Imitation learning from expert paths
train_path_optimizer(expert_demonstrations, epochs=20)

# Fine-tune with RL
train_with_reinforcement_learning(env=habitat_env, steps=1000000)
```

### Module 6: VLA Transformer
**Training Data**:
- Complete navigation episodes from HM3D
- Language instructions from R2R
- All intermediate representations

**End-to-End Training**:
```python
# Combined dataset
vla_dataset = CombinedDataset(
    scenes=hm3d_scenes,
    instructions=r2r_instructions,
    depth_data=nyu_depth,
    semantics=ade20k,
    episodes=navigation_episodes
)

# Train end-to-end
train_vla_gr(vla_dataset, epochs=100)
```

---

## 📈 Training Schedule

### Phase 1: Component Pre-training (2 weeks)
1. Vision encoder on ImageNet (if needed)
2. Depth completion on NYU + ScanNet
3. Language model on navigation instructions
4. Semantic segmentation on ADE20K

### Phase 2: Module Training (1 week)
1. Affordance quantifier
2. GR field solver
3. Path optimizer with expert demos

### Phase 3: End-to-End Training (1 week)
1. Full VLA-GR on HM3D
2. Fine-tuning all components jointly
3. Reinforcement learning fine-tuning

### Phase 4: Evaluation (3 days)
1. Standard benchmarks
2. Ablation studies
3. Cross-dataset evaluation

---

## 💾 Storage Requirements

| Component | Storage Needed |
|-----------|---------------|
| Scene Datasets | 3TB |
| Language Data | 50GB |
| Vision-Language | 150GB |
| Depth Data | 100GB |
| Semantic Data | 30GB |
| Generated Episodes | 200GB |
| **Total** | **~3.5TB** |

---

## 🔧 Data Preprocessing Pipeline

```python
# scripts/prepare_training_data.py

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        
    def prepare_navigation_data(self):
        """Prepare HM3D/MP3D for training"""
        # 1. Generate episodes
        episodes = self.generate_episodes(
            num_episodes=100000,
            min_path_length=5,
            max_path_length=50
        )
        
        # 2. Extract observations
        observations = self.extract_observations(episodes)
        
        # 3. Generate ground truth paths
        gt_paths = self.compute_optimal_paths(episodes)
        
        return NavigationDataset(episodes, observations, gt_paths)
    
    def prepare_language_data(self):
        """Process R2R/RxR instructions"""
        # 1. Tokenize instructions
        # 2. Generate instruction embeddings
        # 3. Align with trajectories
        pass
    
    def prepare_depth_data(self):
        """Prepare depth completion training data"""
        # 1. Generate occlusion masks
        # 2. Create input-target pairs
        # 3. Augment with noise
        pass
```

---

## 📊 Data Augmentation Strategies

### Visual Augmentations
- Random cropping and resizing
- Color jittering (brightness, contrast, saturation)
- Gaussian noise injection
- Random occlusions (10-30% of pixels)
- Depth noise simulation

### Language Augmentations
- Paraphrasing with T5/GPT
- Instruction templating
- Back-translation
- Synonym replacement

### Trajectory Augmentations
- Path perturbations
- Start/goal position variations
- Subgoal injection
- Speed variations

---

## 🎯 Quality Control

### Data Validation
```python
def validate_dataset(dataset):
    """Ensure data quality"""
    checks = {
        "episodes_valid": check_episode_validity(dataset),
        "paths_reachable": check_path_reachability(dataset),
        "observations_complete": check_observation_completeness(dataset),
        "instructions_aligned": check_instruction_alignment(dataset),
        "depth_range_valid": check_depth_values(dataset)
    }
    
    return all(checks.values())
```

### Automatic Filtering
- Remove episodes with collision at start
- Filter paths shorter than 3 steps
- Exclude corrupted depth maps
- Remove ambiguous instructions
