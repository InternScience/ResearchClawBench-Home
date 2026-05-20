"""
Synthetic dataset generation for proof-of-concept experiments.
Generates 32x32 RGB images of colored geometric shapes with text captions and VQA pairs.
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw

# Constants
COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'purple': (128, 0, 128),
    'cyan': (0, 255, 255),
}
SHAPES = ['circle', 'square', 'triangle', 'star']
IMG_SIZE = 32


def draw_shape(draw, shape, color, cx, cy, size):
    if shape == 'circle':
        draw.ellipse([cx-size, cy-size, cx+size, cy+size], fill=color)
    elif shape == 'square':
        draw.rectangle([cx-size, cy-size, cx+size, cy+size], fill=color)
    elif shape == 'triangle':
        draw.polygon([
            (cx, cy-size),
            (cx-size, cy+size),
            (cx+size, cy+size)
        ], fill=color)
    elif shape == 'star':
        # simple diamond-like star
        r = size
        draw.polygon([
            (cx, cy-r), (cx+r//2, cy-r//2),
            (cx+r, cy), (cx+r//2, cy+r//2),
            (cx, cy+r), (cx-r//2, cy+r//2),
            (cx-r, cy), (cx-r//2, cy-r//2)
        ], fill=color)


def generate_image(shape, color):
    bg = tuple(random.randint(180, 220) for _ in range(3))
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    c = COLORS[color]
    margin = 4
    size = random.randint(margin, IMG_SIZE//2 - margin)
    cx = random.randint(size + margin, IMG_SIZE - size - margin)
    cy = random.randint(size + margin, IMG_SIZE - size - margin)
    draw_shape(draw, shape, c, cx, cy, size)
    return img


def generate_dataset(n_samples, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data = []
    for _ in range(n_samples):
        color = random.choice(list(COLORS.keys()))
        shape = random.choice(SHAPES)
        img = generate_image(shape, color)
        caption = f"{color} {shape}"
        qa = [
            ("What is the color?", color),
            ("What is the shape?", shape),
            ("Describe the image.", caption),
        ]
        data.append({
            'image': img,
            'caption': caption,
            'color': color,
            'shape': shape,
            'qa': qa,
        })
    return data


def save_dataset(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(data):
        item['image'].save(os.path.join(out_dir, f"img_{i:05d}.png"))
    # Save metadata as npz-like structure using json
    import json
    meta = []
    for item in data:
        meta.append({
            'caption': item['caption'],
            'color': item['color'],
            'shape': item['shape'],
            'qa': item['qa'],
        })
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    train_data = generate_dataset(5000, seed=42)
    val_data = generate_dataset(500, seed=43)
    test_data = generate_dataset(500, seed=44)
    save_dataset(train_data, 'outputs/synthetic_train')
    save_dataset(val_data, 'outputs/synthetic_val')
    save_dataset(test_data, 'outputs/synthetic_test')
    print("Synthetic datasets generated.")
