import numpy as np
import matplotlib.pyplot as plt

def generate_training_curve():
    epochs = np.arange(1, 101)
    
    # Simulate a typical loss curve
    loss = 5.0 * np.exp(-epochs / 20.0) + 0.5 + np.random.normal(scale=0.1, size=len(epochs))
    val_loss = 5.2 * np.exp(-epochs / 18.0) + 0.6 + np.random.normal(scale=0.15, size=len(epochs))
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (RMSD)')
    plt.title('Simulated Training Curve for Biomolecular Structure Prediction')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('report/images/training_curve.png')
    print("Saved figure to report/images/training_curve.png")

if __name__ == "__main__":
    generate_training_curve()
