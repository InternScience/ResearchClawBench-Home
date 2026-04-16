import numpy as np

with open('data/M-AI-Synth__Materials_AI_Dataset_.txt', 'r') as f:
    lines = f.readlines()

# property_prediction.py
arr1 = eval(lines[1].strip())
arr2 = eval(lines[2].strip())
arr3 = eval(lines[3].strip())
arr4 = eval(lines[4].strip())

print(f"Property Prediction:")
print(f"arr1: len {len(arr1)}, type {type(arr1[0])}, unique {np.unique(arr1)}")
print(f"arr2: len {len(arr2)}, type {type(arr2[0])}, min {min(arr2)}, max {max(arr2)}")
print(f"arr3: len {len(arr3)}, type {type(arr3[0])}, unique {np.unique(arr3)}")
print(f"arr4: len {len(arr4)}, type {type(arr4[0])}, min {min(arr4)}, max {max(arr4)}")

# structure_generation.py
arr5 = eval(lines[7].strip())
arr6 = eval(lines[8].strip())
print(f"\nStructure Generation:")
print(f"arr5: len {len(arr5)}, type {type(arr5[0])}, min {min(arr5)}, max {max(arr5)}")
print(f"arr6: len {len(arr6)}, type {type(arr6[0])}, min {min(arr6)}, max {max(arr6)}")

# autonomous_optimization.py
arr7 = eval(lines[11].strip())
arr8 = eval(lines[12].strip())
arr9 = eval(lines[13].strip())
arr10 = eval(lines[14].strip())
arr11 = eval(lines[15].strip())
arr12 = eval(lines[16].strip())
print(f"\nAutonomous Optimization:")
print(f"arr7: {arr7}")
print(f"arr8: {arr8}")
print(f"arr9: {arr9}")
print(f"arr10: {arr10}")
print(f"arr11: {arr11}")
print(f"arr12: {arr12}")

