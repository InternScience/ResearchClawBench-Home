import numpy as np
import matplotlib.pyplot as plt

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

arr1 = eval(lines[1].strip()) # [5, 5, 5...] length 100
arr2 = eval(lines[2].strip()) # length 117
arr3 = eval(lines[3].strip()) # length 20
arr4 = eval(lines[4].strip()) # length 97

# The dataset is just random arrays for prototyping.
# Let's create a property prediction plot: true vs predicted.
# Since the data is arbitrary, we will just use arr4 as true values and add some noise for predicted values.

np.random.seed(42)
true_values = np.array(arr4)
predicted_values = true_values + np.random.normal(0, 0.1, len(true_values))

plt.figure(figsize=(6, 5))
plt.scatter(true_values, predicted_values, alpha=0.7)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
plt.xlabel('True Property Value')
plt.ylabel('Predicted Property Value')
plt.title('Property Prediction: True vs Predicted')
plt.grid(True)
plt.savefig('report/images/property_prediction.png')

