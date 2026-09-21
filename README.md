# ai-sandbox

## 1. LLM-Huggingface-food-classifier
This repository contains a food classifier model built using Hugging Face's Transformers library. The model is trained to classify images of food into various categories. The project leverages the power of deep learning and transfer learning to achieve high accuracy in food classification tasks.

## GitHub Actions

The repository includes a `Python CI` workflow in `.github/workflows/ci.yml`. It runs for pull requests and pushes to `main` that change the classifier or workflow, installs the pinned Python dependencies, and checks that the Python files compile successfully.

Model training is intentionally not part of CI because it downloads a Hugging Face dataset, can require significant compute, and uploads the resulting model. Run `model.py` manually when training is needed:

```bash
cd LLM-Huggingface-food-classifier
python -m pip install -r requirements.txt
python model.py
```

To run the Gradio app after a model has been trained:

```bash
python app.py
```
