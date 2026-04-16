import yaml
import json

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

if __name__ == '__main__':
    data = load_data('../data/2111.01152/2111.01152.yaml')
    
    # The final task should be the last one in the list
    final_task = data[-1]
    print(f"Final Task: {final_task.get('task')}")
    print(f"Final Answer: {final_task.get('answer')}")
    
    with open('../outputs/final_hamiltonian.txt', 'w') as f:
        f.write(f"Final Task: {final_task.get('task')}\n")
        f.write(f"Final Answer: {final_task.get('answer')}\n")
