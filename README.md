# PointLLM 3D Evaluation Pipeline

This repository provides the **full evaluation pipeline** for our project:  
**"On the Use of Large Language Models for 3D Point Cloud Understanding."**

We extend the original [PointLLM](https://github.com/OpenRobotLab/PointLLM) model beyond single-object understanding to handle **complex indoor scenes** using data from the **ScanNet** dataset. Rather than retraining, our approach focuses on automated context generation and large-scale evaluation of the model’s performance under different settings.

---

## Highlights

- **No Model Retraining**  
  PointLLM is used out-of-the-box without fine-tuning.

- **Multi-Scene Evaluation**  
  Evaluate on complete ScanNet scenes with rich object combinations.

- 📋 **Captioning & Classification Tasks**  
  Includes fully automated evaluation loops for both tasks.

- 🤖 **LLM-Based Evaluation**  
  Replaces traditional human-based assessments with ChatGPT evaluation strategies.

- ✅ **Strict Binary Answer Format**  
  Enforces "Yes"/"No" answers for consistent metric-based evaluation.

---

## 📁 Project Structure
├── pointllm/ # Core PointLLM model and conversation logic │ ├── model/ # LLM model classes and loading utilities │ ├── conversation/ # Prompt templates and dialogue handling │ └── utils/ # Utility functions for setup and decoding │ ├── data/ # Dataset-related files │ ├── ground_truth.json # Ground-truth annotations for object presence │ ├── material_list_updated.json # Object-to-material mappings │ └── context/ # Natural language scene descriptions │ ├── evaluation/ # Automated evaluation scripts │ ├── evaluate_classification.py # Classification evaluation loop │ ├── evaluate_captioning.py # Captioning evaluation loop │ └── analyze_results.py # Accuracy & metric calculation │ ├── preprocessing/ # Data transformation and ScanNet processing │ ├── process_scannet.py # Converts ScanNet to usable format │ └── generate_context.py # Creates natural language scene context │ ├── results/ # Logs and outputs of evaluations │ ├── evaluation_log_*.json # Per-scene evaluation outputs │ └── summary_metrics.json # Summary statistics │ ├── scripts/ # Optional CLI wrappers for quick runs │ └── run_eval.sh # Shell script to launch evaluation │ ├── README.md # Project overview └── requirements.txt # Python dependencies
