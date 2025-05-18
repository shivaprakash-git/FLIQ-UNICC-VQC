# Quantum-Enhanced Income Prediction

## 🚀 Overview

This project implements a quantum-classical hybrid model for predicting income levels (above or below $50K annually) using the UCI Adult Income dataset. By leveraging the power of quantum computing through PennyLane and PyTorch, we've created a model that achieves **83.59%** validation accuracy with impressively rapid convergence.

## 📊 Performance Metrics

Our model achieved:
- **Validation Accuracy: 83.59%** (reached in first epoch and remained stable)
- **Fairness Metrics**:
  - Sex disparate impact ratio: 0.8316 (✅ above fairness threshold)
  - Race disparate impact ratio: 0.8632 (✅ above fairness threshold)
- **Model Convergence**: Rapid convergence to peak performance
- **Group Performance**: Notably high performance across minority demographic groups

## 🧠 Model Architecture

### Quantum Circuit Design

Our implementation uses a 4-qubit quantum circuit with several innovations:
- **Data re-uploading**: Feature values encoded twice for enhanced expressivity
- **Strongly entangling layers**: Two distinct variational blocks with 3 layers each
- **Efficient parametrization**: (2, 3, 4, 3) weight shape for optimal expressivity vs. trainability

```
Quantum Circuit Structure:
1️⃣ Initial encoding (RY gates)
2️⃣ First variational block (StronglyEntanglingLayers)
3️⃣ Re-upload features (RY gates again)
4️⃣ Second variational block (StronglyEntanglingLayers)
5️⃣ Measurement (PauliZ)
```

### Classical Components

We enhanced standard quantum machine learning approaches with:
- **Preprocessing network**: 
  ```
  Linear(4→16) → ReLU → Dropout(0.2) → Linear(16→4) → Tanh
  ```
- **Post-processing**:
  ```
  Linear(1→8) → ReLU → Dropout(0.2) → Linear(8→1) → Sigmoid
  ```

## 🔧 Implementation Details

### Data Processing

- **Feature Engineering**: PCA reduction to 4 quantum-compatible features
  - Explains 42.15% of dataset variance
  - Components capture socioeconomic factors, demographics, work characteristics
- **Train/Val/Test Split**: 65%/15%/20% ratio with stratification

### Training Approach

- **Optimization**: Adam optimizer with learning rate 0.01
- **Batch Size**: 256 for efficient GPU utilization
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive optimization
- **Early Stopping**: Patience of 5 epochs with 0.001 min_delta
- **Hardware Acceleration**: CUDA with lightning.gpu for quantum simulation

## ⚙️ Installation & Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run the notebook
jupyter notebook quantum_income_classifier.ipynb
```

### Loading the trained model

```python
# Define model architecture first (see notebook)
model = EnhancedHybridModel()
checkpoint = torch.load('quantum_model_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    prediction = model(input_tensor)
```

## 💡 Key Innovations

1. **Rapid Convergence Architecture**: Our model achieved peak performance in the first epoch and maintained it

2. **Fairness-Aware Design**: The model shows excellent performance across demographic groups, exceeding fairness thresholds for both gender and race

3. **Quantum-Classical Balance**: Enhanced classical components complement the quantum circuit to maximize expressive power while maintaining trainability

4. **Efficient Quantum Design**: Our circuit uses data re-uploading with only 4 qubits, balancing quantum expressivity and training efficiency

## 🔍 Future Directions

- Explore quantum feature maps beyond RY encoding
- Better Classical Neural Networks Architecture
- Investigate larger qubit counts as quantum hardware improves
- Develop quantum advantage metrics specific to tabular data
- Research quantum-specific fairness constraints
- Enhancing Accuracy and Scalability

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share - copy and redistribute the material in any medium or format
- Adapt - remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- Attribution - You must give appropriate credit, provide a link to the license, and indicate if changes were made.
