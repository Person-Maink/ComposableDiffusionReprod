import os
import json
import random
import math
from PIL import Image, ImageDraw


def generate_shape_dataset(folder_location, num_images, image_size):
    os.makedirs(folder_location, exist_ok=True)
    shapes = ['circle', 'square', 'triangle', 'star', 'hexagon', 'diamond']
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'black': (0, 0, 0)
    }

    metadata = {}

    shape_size = min(image_size) // len(shapes)

    for i in range(num_images):
        # Create blank white image
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)

        # Generate random number of shapes (1-6)
        num_shapes = random.randint(1, len(shapes))
        selected_shapes = random.sample(shapes, num_shapes)

        # Store shape info for this image
        image_data = {
            'shapes': [],
            'coordinates': [],
            'colors': []
        }

        # List to keep track of placed shapes (x, y, size)
        placed_shapes = []

        for shape in selected_shapes:
            # Try to find a non-overlapping position (max attempts to prevent infinite loops)
            max_attempts = 100
            placed = False

            for _ in range(max_attempts):
                # Random position (ensure shape stays within bounds)
                margin = shape_size
                x = random.randint(margin, image_size[0] - margin)
                y = random.randint(margin, image_size[1] - margin)

                # Check for overlap with existing shapes
                overlap = False
                for (px, py, psize) in placed_shapes:
                    # Calculate distance between centers
                    distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                    # If distance is less than sum of radii, they overlap
                    if distance < (shape_size + psize):
                        overlap = True
                        break

                if not overlap:
                    # Found a good position
                    placed_shapes.append((x, y, shape_size))
                    placed = True
                    break

            if not placed:
                # If we couldn't find a non-overlapping position, skip this shape
                continue


            color_name, color_rgb = random.choice(list(colors.items()))

            if shape == 'circle':
                draw.ellipse([(x - shape_size, y - shape_size),
                              (x + shape_size, y + shape_size)],
                             fill=color_rgb, outline=color_rgb)
            elif shape == 'square':
                draw.rectangle([(x - shape_size, y - shape_size),
                                (x + shape_size, y + shape_size)],
                               fill=color_rgb, outline=color_rgb)
            elif shape == 'triangle':
                draw.polygon([(x, y - shape_size),
                              (x + shape_size, y + shape_size),
                              (x - shape_size, y + shape_size)],
                             fill=color_rgb, outline=color_rgb)
            elif shape == 'star':
                draw_star(draw, x, y, shape_size, color_rgb)
            elif shape == 'hexagon':
                draw_regular_polygon(draw, x, y, shape_size, 6, color_rgb)
            elif shape == 'diamond':
                draw_diamond(draw, x, y, shape_size, color_rgb)

            image_data['shapes'].append(shape)
            image_data['coordinates'].append((x, y))
            image_data['colors'].append(color_name)

        filename = f"{i:04d}.jpg"
        img.save(os.path.join(folder_location, filename))

        metadata[filename] = image_data

    with open(os.path.join('metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def draw_star(draw, x, y, size, fill_color):
    outer_radius = size
    inner_radius = size // 2
    points = []
    for i in range(10):
        angle = 2 * math.pi * i / 10 - math.pi / 2
        radius = inner_radius if i % 2 else outer_radius
        points.append((x + radius * math.cos(angle),
                       y + radius * math.sin(angle)))
    draw.polygon(points, fill=fill_color, outline=fill_color)

def draw_regular_polygon(draw, x, y, size, sides, fill_color):
    points = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        points.append((x + size * math.cos(angle),
                       y + size * math.sin(angle)))
    draw.polygon(points, fill=fill_color, outline=fill_color)

def draw_diamond(draw, x, y, size, fill_color):
    draw.polygon([(x, y - size),  # Top
                 (x + size, y),   # Right
                 (x, y + size),    # Bottom
                 (x - size, y)],  # Left
                fill=fill_color, outline=fill_color)

if __name__ == "__main__":
    generate_shape_dataset('./shape_dataset', 10000, (64, 64))