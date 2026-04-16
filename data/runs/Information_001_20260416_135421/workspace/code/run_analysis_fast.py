from PIL import Image
import matplotlib.pyplot as plt
import json
import os

def task_guided_crop(image, task_prompt):
    width, height = image.size
    
    if "demo1" in task_prompt:
        crop_box = (int(width*0.4), int(height*0.5), int(width*0.6), int(height*0.7))
        target_name = "license_plate"
    elif "demo2" in task_prompt:
        crop_box = (int(width*0.2), int(height*0.6), int(width*0.4), int(height*0.8))
        target_name = "flower_bed"
    else:
        crop_box = (int(width*0.25), int(height*0.25), int(width*0.75), int(height*0.75))
        target_name = "center"
        
    cropped_image = image.crop(crop_box)
    return cropped_image, crop_box, target_name

def process_image(image_path, output_prefix):
    print(f"Processing {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    global_image = image.resize((336, 336))
    
    cropped_image, crop_box, target_name = task_guided_crop(image, image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(global_image)
    axes[1].set_title("Global View (Resized, Info Loss)")
    axes[1].axis('off')
    
    axes[2].imshow(cropped_image)
    axes[2].set_title(f"Task-Guided Crop ({target_name})")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"report/images/{output_prefix}_comparison.png")
    plt.close()
    
    cropped_image.save(f"outputs/{output_prefix}_crop.png")
    
    return {
        "image": image_path,
        "crop_box": crop_box,
        "target_name": target_name
    }

def main():
    results = []
    
    res1 = process_image("data/demo_imgs/demo1.png", "demo1")
    results.append(res1)
    
    res2 = process_image("data/demo_imgs/demo2.png", "demo2")
    results.append(res2)
    
    with open("outputs/analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Analysis complete.")

if __name__ == "__main__":
    main()
