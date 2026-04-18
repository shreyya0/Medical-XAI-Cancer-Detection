# Medical-XAI-Cancer-Detection
End-to-end cancer classification using VGG16 with Explainable AI (Grad-CAM &amp; LIME) for diagnostic interpretability.
# Interpretability-First Cancer Detection Pipeline

Developed a deep learning diagnostic system to classify 8 cancer types from MRI scans, prioritizing **Explainable AI (XAI)** for clinical decision support.

## 🔬 Diagnostic Visualization (XAI)
![XAI Heatmap](./xai_results.png)
*Figure: Side-by-side comparison showing how Grad-CAM and LIME identify the specific features used for classification.*

## 🚀 Key Features
Architecture: Fine-tuned VGG16 backbone with custom top layers for high-precision multi-class classification.

Explainability (XAI): Integrated Grad-CAM and LIME to generate high-resolution diagnostic heatmaps, providing visual "proof" for model predictions.

Automated Clinical Reporting: Integrated LLM endpoints (Mistral-7B) to transform raw tensor data into patient-friendly, natural language diagnostic summaries.

Engineering Excellence: Developed a custom NumPy-aware JSON encoder to handle high-precision float32 serialization for seamless API integration.

Hardware Optimization: Successfully architected a cloud-based inference workflow to bypass local AVX instruction set limitations.

## 📦 Model Weights
Large-scale model artifacts and the full VGG16 checkpoint are managed externally due to GitHub size constraints.

Access Model: [Google drive link](https://drive.google.com/file/d/114Hq7VX7BtYmcLMszdYiUM1pXc8oCBtg/view?usp=sharing)] for local testing.

---
