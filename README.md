# QoR Predictor

A machine learning-based Quality of Results (QoR) predictor for estimating **delay** in digital circuit designs during logic synthesis. The project leverages the ABC synthesis tool and a custom deep learning model, to predict QoR metrics (primarily delay) for a given design and synthesis recipe, enabling rapid design space exploration without running computationally expensive synthesis.

## Project Overview

The **QoR Predictor** is designed to estimate the delay of a digital circuit design based on its And-Inverter Graph (AIG) structure and a synthesis recipe. The project includes:
- **Data Generation**: Generating labeled datasets by synthesizing designs with various recipes using the ABC tool.
- **Recipe Optimization**: Exploring standard, random, semi-random handcrafted, and simulated annealing-based synthesis recipes to optimize QoR.
- **ML Model**: A custom end-to-end deep learning model that combines a **Graph Encoder** (based on GraphSAGE), a **Recipe Encoder** (using Transformer layers), and a **Cross-Attention Fusion** module to predict delay.
- **Fine-Tuning**: Adapting the model to new, unseen designs by retraining on a small set of synthesis recipes.

The project uses **Python** (for ML model development and data processing), **C++** (for performance-critical tasks), and **Shell** scripts (for automation). It is built on top of the **OpenABC-D** dataset framework and integrates with the **Yosys-ABC** synthesis tool.

## Repository Structure

- **datagen/**: Scripts for generating synthesis data (e.g., `.pt` files, graph statistics).
- **models/**:
  - `graph_encoder.py`: Implements the GraphSAGE-based encoder for AIG structures.
  - `recipe_encoder.py`: Implements the Transformer-based encoder for synthesis recipes.
  - `fusion_model.py`: Implements the bidirectional cross-attention fusion module.
- **ptfiles/**: Stores PyTorch data objects (`.pt` files) representing AIGs for designs.
- **recipes/**: Contains synthesis recipe files in CSV format.
- **out_label/**: Stores delay labels for each design-recipe pair in CSV format.
- **predefined_recipe_runner.sh**: Shell script to run standard synthesis recipes and compute QoR metrics (delay, area).
- **handcrafted_recipe_generator.cpp**: C++ script to generate semi-random handcrafted synthesis recipes.
- **train.py**: Python script for training the Custom model.
- **Predefined Recipe Results.pdf**: Document containing results of standard recipe runs.
- **README.md**: This file.

## Installation

### Prerequisites
- **Python 3.8+**: For ML model training and data processing.
- **PyTorch**: For deep learning model implementation.
- **PyTorch Geometric**: For handling graph-structured AIG data.
- **NetworkX**: For graph manipulation.
- **NumPy, Pandas, Scikit-learn**: For data preprocessing and analysis.
- **Yosys-ABC**: For synthesis runs (ensure it is available in your PATH).
- **g++**: For compiling C++ code (e.g., `handcrafted_recipe_generator.cpp`).
- **Shell**: For running automation scripts.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ash-intosh/QoR-predictor.git
   cd QoR-predictor
   ```

2. **Set Up Python Environment**:
   Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install torch torch-geometric numpy pandas scikit-learn networkx
   ```

3. **Install Yosys-ABC**:
   Follow instructions from [Berkeley-ABC](https://github.com/berkeley-abc/abc) to install Yosys-ABC and ensure it is in your PATH.

4. **Compile C++ Code**:
   Compile the handcrafted recipe generator:
   ```bash
   g++ -o handcrafted_recipe_generator handcrafted_recipe_generator.cpp
   ```

5. **Fix Permissions**:
   Ensure shell scripts are executable:
   ```bash
   chmod +x predefined_recipe_runner.sh
   chmod +x datagen/*.sh
   ```

## Usage

### 1. Data Generation
Generate a dataset of AIGs and QoR metrics for training the model:
1. **Generate Synthesis Scripts**:
   ```bash
   python datagen/automate_synthesisScriptGen.py
   ```
   This creates 1500 synthesis recipes in the `synScripts/` folder.

2. **Run Synthesis**:
   Update `predefined_recipe_runner.sh` with the correct bench file (e.g., `simple_spi.bench`) and CSV output file name. Then run:
   ```bash
   ./predefined_recipe_runner.sh
   ```
   This generates log files and zips them.

3. **Convert to Graph Format**:
   ```bash
   python datagen/automate_synbench2Graphnm.py
   ```
   This converts bench files to GraphML format using 21 parallel threads.

4. **Generate PyTorch Data**:
   ```bash
   python datagen/synthID2SeqMapping.py
   python datagen/PyGDataAIG.py
   ```
   This creates `.pt` files in `ptfiles/` and numerical encodings in a pickle file.

5. **Collect Statistics**:
   ```bash
   python datagen/collectAreaAndDelay.py
   python datagen/collectGraphStatistics.py
   python datagen/pickleStatsFormML.py
   ```
   This generates delay labels and graph statistics in `out_label/` and pickle files.

### 2. Recipe Optimization
- **Run Standard Recipes**:
  Use `predefined_recipe_runner.sh` to evaluate standard recipes (e.g., `resyn`, `compress2`) from the ABC `abc.rc` file. Results are saved in `Predefined Recipe Results.pdf`.
- **Generate Semi-Random Handcrafted Recipes**:
  ```bash
  ./handcrafted_recipe_generator
  ```
  This creates semi-random recipes using operations like `b`, `fraig_store`, `rw`, and `fraig_restore`.
- **Simulated Annealing**:
  Run the simulated annealing algorithm (implemented in Python) to optimize recipes for the delay-area product (QoR). Results are stored in the repository.

### 3. Training the Custom Model
Train the model using:
```bash
python train.py --datadir OPENABC-D --rundir OUTPUT/NETV1_set1 --dataset set1 --lp 1 --lr 0.001 --epochs 20 --target delay
```
- **Parameters**:
  - `--datadir`: Path to dataset directory (`ptfiles/`, `recipes/`, `out_label/`).
  - `--rundir`: Output directory for model checkpoints and results.
  - `--dataset`: Split strategy (`set1`, `set2`, or `set3`).
  - `--lp`: Learning problem (1 for QoR prediction).
  - `--lr`: Learning rate (default: 0.001).
  - `--epochs`: Number of training epochs (default: 20).
  - `--target`: Target metric (`delay`).

### 4. Fine-Tuning
Fine-tune the model on new designs:
1. Generate new `.pt` files, recipes, and delay labels for the new design (follow data generation steps).
2. Load a pre-trained checkpoint and retrain:
   ```bash
   python train.py --datadir NEW_DATA --rundir OUTPUT/FINETUNE --dataset set1 --lp 1 --lr 0.001 --epochs 10 --target delay
   ```

### 5. Evaluation
Evaluate the model using **Mean Squared Error (MSE)**, **Mean Absolute Percentage Error (MAPE)**, and **R² Score**. Results are saved in the `rundir` specified during training.

## Key Features
- **Custom Model**:
  - **Graph Encoder**: Uses GraphSAGE with residual connections, BatchNorm, and dropout for robust AIG feature extraction.
  - **Recipe Encoder**: Employs Transformer layers with sinusoidal positional encoding to capture recipe sequence dependencies.
  - **Cross-Attention Fusion**: Combines graph and recipe embeddings using bidirectional cross-attention for context-aware predictions.
- **Dataset**: Supports `.pt` files for AIGs, CSV files for recipes and delay labels, and GraphML for intermediate representations.
- **Recipe Optimization**: Includes standard, semi-random handcrafted, and simulated annealing-based recipes.
- **Fine-Tuning**: Adapts the model to new designs with minimal retraining.

## Results
- **Standard Recipes**: Evaluated recipes like `resyn`, `compress2`, and `choice` from ABC’s `abc.rc` file.
- **Semi-Random Handcrafted Recipes**: Outperformed standard recipes in many cases, with key operations like `rs` (with cuts), `b`, and `fraig_restore`.
- **Simulated Annealing**: Further optimized recipes, achieving better QoR (delay-area product) for designs like `simple_spi` and `tv80`.
- **Custom Model**: Achieves low MSE and MAPE, with improved performance after fine-tuning on new designs.

## Challenges and Insights
- **Data Generation**: Permission issues with shell scripts were resolved using `chmod +x`. Incorrect step ordering in the original OpenABC-D pipeline required reordering (e.g., running `synthID2SeqMapping.py` before `PyGDataAIG.py`).
- **Dataset Limitations**: Step 0 AIGs (unsynthesized) lack recipe-specific variations, limiting model learning. Step 20 AIGs (post-synthesis) better reflect recipe effects.
- **Model Design**: GraphSAGE outperformed GCN due to neighborhood aggregation and residual connections. Bidirectional cross-attention improved prediction accuracy by capturing graph-recipe interactions.

## References
1. ABC: A System for Sequential Synthesis and Verification. [Online]. Available: https://people.eecs.berkeley.edu/~alanmi/abc/.
2. A. Mishchenko, S. Chatterjee, R. Jiang, and R. K. Brayton. "FRAIGs: A Unifying Representation for Logic Synthesis and Verification." ERL Technical Report, EECS Dept., UC Berkeley, 2005.
3. OpenABC-D Dataset and Tools. [Online]. Available: https://github.com/Berkeley-abc/abc.


