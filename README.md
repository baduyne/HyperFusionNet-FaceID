# HyperFusionNet-FaceID

**Multispectral Face Recognition in Low-Light Environments using VIS and LWIR Fusion**

##  Overview

**HyperFusionNet-FaceID** is a robust face recognition system designed to operate effectively in challenging lighting conditions, especially low-light or high-glare environments. The project fuses two image modalities:
- Visible spectrum (VIS) images
- Long-Wave Infrared (LWIR) thermal images

By extracting and combining features from both sources, the model significantly improves recognition accuracy in real-world scenarios such as nighttime environments, harsh outdoor lighting, or when subjects wear masks.

---

##  Features

-  **HyperFacePipeline**: A modular pipeline including encoding, fusion, decoding, and face embedding.
-  **Multispectral Fusion**: Combines VIS and LWIR images using a linear fusion approach (e.g., `0.8 * IR + 0.2 * VIS`).
-  **Face Embedding**: Extracts a 512-dimensional embedding vector using the FaceNet model.
-  **Matching**: Compares embeddings using Cosine Similarity against a user database.
- **Evaluation**: Supports evaluation metrics such as Accuracy, Rank-1 Recognition Rate, and CMC curve.
- **Demo**: Simple demo interface that accepts VIS + IR input and performs identity recognition.
![image](https://github.com/user-attachments/assets/96e83716-8e39-402e-a1ea-6e002b7de25d)


# Setup

-   Installation

```bash
pip install -r requirements.txt
```

-   Run the project (if you use Windows, run this command in Git Bash)

```bash
sh run.sh
```
