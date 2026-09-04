# iNaturalist Species Image Classification Pipeline

An end-to-end computer vision pipeline using PyTorch, **MobileNetV3** transfer learning, **iNaturalist REST API** data acquisition, and **Grad-CAM** (Gradient-weighted Class Activation Mapping) explainable AI.

---

## Project Workflow

```
Phase 1: Data Pipeline
  [iNaturalist API] 
       │ (scripts/01_fetch_data.py)
       ▼
  [data/raw/] ──► (scripts/02_preprocess_data.py) ──► [data/processed/] (train/val/test + class_map.json)

Phase 2: Model Training & Evaluation
  [data/processed/] ──► (scripts/03_train_model.py) ──► [models/species_classifier.pth] + [results/training_log.csv]
                           │
                           ▼
                       (scripts/04_evaluate_model.py) ──► [results/evaluation_metrics.json] + [results/confusion_matrix.png/csv]

Phase 3: Model Interpretation (XAI)
  [models/species_classifier.pth] ──► (scripts/05_run_grad_cam.py) ──► [results/grad_cam_outputs/*.png]
```

---

## Directory Structure

```
├── data/
│   ├── raw/                       # Raw downloaded images per species (10 species >= 200 images each)
│   └── processed/                 # Stratified train (70%), val (15%), test (15%) splits
│       └── class_map.json         # Class index mapping
├── models/
│   └── species_classifier.pth     # Fine-tuned MobileNetV3 PyTorch weights checkpoint
├── results/
│   ├── grad_cam_outputs/          # Grad-CAM heatmap overlays (PNG)
│   ├── training_log.csv           # Epoch-by-epoch losses and validation metrics
│   ├── evaluation_metrics.json    # Accuracy, Precision, Recall, F1-Score JSON metrics
│   ├── confusion_matrix.png       # Visualized confusion matrix heatmap
│   ├── confusion_matrix.csv       # Raw N x N confusion matrix data
│   └── performance_analysis.md    # Performance analysis report
├── scripts/
│   ├── 01_fetch_data.py           # API data acquisition script
│   ├── 02_preprocess_data.py      # Dataset stratification and preprocessing script
│   ├── 03_train_model.py          # MobileNetV3 fine-tuning script
│   ├── 04_evaluate_model.py       # Model evaluation script
│   └── 05_run_grad_cam.py         # Grad-CAM XAI visualization script
├── .env.example                   # Template environment variables
├── .env                           # Local environment variables
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Application Dockerfile (python:3.9-slim base)
├── docker-compose.yml             # Docker Compose application setup
├── README.md                      # Project documentation
├── requirements.txt               # Python package dependencies
└── video_script.md                # Video presentation script
```

---

## Prerequisites & Installation

### Option 1: Docker (Recommended)

1. Build the Docker container image:
   ```bash
   docker-compose up --build
   ```
2. Start an interactive bash session in the app container:
   ```bash
   docker-compose run app bash
   ```

### Option 2: Local Python Environment

1. Create a Python 3.9 virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Execution Guide

Run the pipeline steps sequentially:

### 1. Data Acquisition
Query the iNaturalist API for species in order *Lepidoptera* (Taxon ID `47157`) and download image observations:
```bash
python scripts/01_fetch_data.py
```

### 2. Data Preprocessing
Perform a stratified 70% train / 15% val / 15% test split and build `data/processed/class_map.json`:
```bash
python scripts/02_preprocess_data.py
```

### 3. Model Training
Fine-tune pre-trained MobileNetV3-Small on training split and log progress to `results/training_log.csv`:
```bash
python scripts/03_train_model.py
```

### 4. Model Evaluation
Evaluate best checkpoint on test set and generate `evaluation_metrics.json` and confusion matrix artifacts:
```bash
python scripts/04_evaluate_model.py
```

### 5. Grad-CAM Interpretation
Generate heatmap overlays showing spatial attention maps for test predictions in `results/grad_cam_outputs/`:
```bash
python scripts/05_run_grad_cam.py
```

---

## Video Script

A video presentation script is available in [video_script.md](file:///c:/Users/harsha%20vashi/Desktop/GPP/inaturalist-species-classifier/video_script.md).