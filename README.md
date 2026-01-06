# Handwritten Number Recognition Program

A dark-themed handwritten number recognition program built with PyTorch.

## Features

- 28*28 pixel drawing canvas
- Operation guidance display
- **Train Model** - Train a model using the MNIST dataset
- **Personal Training** - Train the model with your own handwriting

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- tkinter (usually included with Python)

## Usage

### Recommended steps

1. **First Use - Train Model**:
   - Click "Train Model" button
   - Confirm to start training
   - Wait for training to complete (~3-5 minutes)
   - First run will automatically download MNIST dataset

2. **Personal Training (Optional but Recommended)**:
   - Click "Personal Train" button
   - Write each number (0-9) sequentially
   - Write each number 3 times
   - System will automatically collect and train
   - After completion, AI will better recognize your handwriting style

3. **Draw numbers**:
   - Draw numbers (0-9) on the 28×28 canvas with mouse
   - Canvas displays your drawing in real-time
   - Supports semi-transparent border effect for more natural strokes

4. **Recognize numbers**:
   - Click "Recognize" button
   - View recognition result and confidence

5. **Clear Canvas**:
   - Click "Clear" button to clear canvas

## Interface Description

- **Canvas**: 28×28 pixel drawing area with semi-transparent border effect
- **Clear Button**: Clear canvas content
- **Recognize Button**: Recognize drawn number (disabled when canvas is empty)
- **Train Model Button**: Train model using MNIST dataset (green)
- **Personal Train Button**: Teach AI your handwriting style (purple)
- **Result Display**: Shows recognized number and confidence percentage
- **Progress Display**: Shows training progress and status

## Technical Details

### Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixels flattened)
- **Hidden Layer 1**: 256 neurons + ReLU activation
- **Hidden Layer 2**: 128 neurons + ReLU activation
- **Output Layer**: 10 neurons + Softmax activation

### Personal Training Principle

Personal training uses transfer learning technique:
1. First train base model on MNIST dataset
2. Collect 30 user handwriting samples (3 per number)
3. Fine-tune model with smaller learning rate (0.001)
4. Train for 50 epochs to fully learn user's handwriting style
5. Save optimized model

This way AI can more accurately recognize your personal handwriting style!

### Training Parameters

- Training set: 60,000 MNIST images
- Test set: 10,000 MNIST images
- Batch Size: 64
- Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Cross Entropy
- Training Epochs: 5 epochs
- Expected Accuracy: >95%

### Dark Theme Colors

- Background: #1e1e1e
- Text: #ffffff
- Canvas Background: #2d2d2d
- Drawing Color: #ffffff
- Button Background: #3c3c3c
- Train Button: #2d5a2d (green)
- Personal Train Button: #5a2d5a (purple)

## Notes

1. First run requires downloading MNIST dataset (~12MB)
2. Training takes 3-5 minutes, please be patient
3. UI may briefly freeze during training, this is normal
4. Model automatically saves to `number_model.pth` after training
5. Recommend training model first before using recognition

## Troubleshooting

### Training Fails
- Check network connection (needs to download MNIST dataset)
- Ensure sufficient disk space (~100MB)
- Check console for error messages

### Low Recognition Accuracy
- Ensure model training is complete
- Try drawing clearer numbers
- Draw numbers centered on canvas

### Dependency Installation Issues
- Ensure Python version >= 3.8
- Windows users: tkinter usually comes with Python
- Use virtual environment to avoid dependency conflicts
