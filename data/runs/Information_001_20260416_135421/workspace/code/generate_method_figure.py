from PIL import Image
import matplotlib.pyplot as plt

def generate_method_figure():
    print("Generating method figure...")
    image_path = "data/demo_imgs/method_case.png"
    image = Image.open(image_path).convert("RGB")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("SEAL Framework: Guided Visual Search (V*)")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("report/images/method_overview.png")
    plt.close()
    print("Method figure generated.")

if __name__ == "__main__":
    generate_method_figure()
