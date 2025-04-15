# Transformer Model Training

This project provides a framework for training Transformer models on custom datasets. It uses the Hugging Face Transformers library and PyTorch for implementation.

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

Your dataset should be in CSV format with at least two columns:
- `text`: The input text data
- `label`: The target label (integer)

Example:
```csv
text,label
"This is a positive review",1
"This is a negative review",0
```

## Usage

1. Prepare your dataset in the required format
2. Update the `data_path` variable in `train.py` to point to your dataset
3. Run the training script:
```bash
python train.py
```

## Configuration

You can modify the following parameters in `train.py`:
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for the optimizer

## Model Architecture

The project uses BERT (bert-base-uncased) as the base model for sequence classification. You can modify this in the `train.py` file by changing the model name in `AutoModelForSequenceClassification.from_pretrained()`.

## Monitoring

The training process is logged using Weights & Biases (wandb). You'll need to:
1. Create a wandb account
2. Log in to wandb
3. The metrics will be automatically logged during training

## Output

- The best model will be saved as `best_model.pt`
- Training metrics are logged to wandb
- Final test results are printed to the console

## License

This project is licensed under the MIT License - see the LICENSE file for details. 