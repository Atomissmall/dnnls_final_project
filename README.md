DNNLS Final Project — Deep Learning System Implementation

This repository contains my final coursework project for Deep Neural Networks & Learning Systems (DNNLS).
The project focuses on designing, implementing, and evaluating a deep learning pipeline under realistic computational constraints, with emphasis on model architecture, training dynamics, and limitations.

The implementation is intentionally modular and reproducible, suitable for execution on Google Colab using limited GPU resources.

Project Overview

This project explores a deep neural network–based learning system applied to a structured dataset involving visual and/or temporal data (depending on configuration). The goals of the project are to:

Demonstrate understanding of deep learning architectures

Apply appropriate training strategies

Evaluate model behavior and limitations

Discuss trade-offs between performance, complexity, and compute

The system is not intended to be state-of-the-art; instead, it prioritizes clarity, correctness, and interpretability, as required for academic assessment.

Key Objectives

Design a working deep learning pipeline end-to-end

Train and validate models using realistic constraints (free Colab GPU)

Analyze convergence behavior and qualitative outputs

Discuss shortcomings and possible future improvements

Model Architecture (High-Level)

The system consists of the following components:

Feature Extraction

Convolutional layers for spatial feature learning (images / frames)

Optional temporal handling via sequence models

Latent Representation

Encodes high-level abstractions from raw inputs

Balances expressiveness with computational feasibility

Prediction / Reconstruction Head

Generates task-specific outputs

Trained using supervised or reconstruction-based losses

Architectural choices were made to balance learning capacity and training stability under limited GPU memory.

Training Strategy

Optimizer: Adam

Loss functions:

Reconstruction / classification loss (task-dependent)

Optional auxiliary losses

Batch sizes and resolution deliberately kept small

Training monitored using validation metrics and qualitative inspection

Training is controlled entirely via a configuration file, enabling easy experimentation.

Results (Summary)

After training:

The model learns coarse but meaningful representations

Predictions are structurally plausible, though not perfectly sharp

Performance reflects expected behavior for:

Small datasets

Limited epochs

Lightweight architectures

These results are acceptable and expected given the project constraints.

Limitations

This project is intentionally constrained:

No large pretrained foundation models

No large-scale datasets

No heavy transformer-based vision-language models

This Project	Large-Scale Systems
CNN-based models	Vision Transformers
LSTM / GRU (if used)	Multimodal Transformers
Small latent spaces	Large pretrained embeddings
Free Colab GPU	Dedicated training clusters

These limitations are acknowledged and discussed as part of the analysis.

Project Structure
dnnls_final_project/
├── notebooks/
│   └── Final_template.ipynb
├── src/
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── models/
├── results/
├── config.yaml
├── requirements.txt
└── README.md


notebooks/ – Main experiment notebook

src/ – Model definitions and training logic

models/ – Saved checkpoints

results/ – Training curves and outputs

config.yaml – All hyperparameters and paths

Reproducibility (Google Colab Workflow)
1. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

2. Clone repository into Drive
!git clone https://github.com/Atomissmall/dnnls_final_project.git

3. Change directory
%cd dnnls_final_project

4. Install dependencies
!pip install -r requirements.txt

5. Run the notebook

Open and run:

notebooks/Final_template.ipynb

Conclusion

This project demonstrates:

Practical understanding of deep learning system design

Ability to train and evaluate neural networks under constraints

Awareness of architectural trade-offs and limitations

Clear separation between educational implementation and production-scale systems

The work fulfills the assessment requirements for DNNLS and provides a solid foundation for future extensions.

Author

Abdulbasit (Atom)
<br>
MSc Artificial Intelligence
Deep Neural Networks & Learning Systems
Sheffield Hallam University