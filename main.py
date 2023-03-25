import csv
import numpy as np
from PIL import Image
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ColorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ColorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_image_colors(image_path, model, label_encoder):
    image = Image.open(image_path)
    pixels = np.array(image.getdata()) / 255.0
    pixels = torch.tensor(pixels, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        outputs = model(pixels)
        _, predicted_indices = torch.max(outputs, 1)
        color_names = label_encoder.inverse_transform(
            predicted_indices.numpy())

    color_counter = defaultdict(int)
    for color_name in color_names:
        color_counter[color_name] += 1

    return color_counter


def get_rgb_from_csv(colors_list, csv_file_path):
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        colors_dict = {row['Name'].lower(): (int(row['Red (8 bit)']), int(
            row['Green (8 bit)']), int(row['Blue (8 bit)'])) for row in reader}
    result = []
    for color_name in colors_list:
        if color_name.lower() in colors_dict:
            rgb = colors_dict[color_name.lower()]
            result.append((color_name, rgb))
        else:
            result.append((color_name, None))
    return result


def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def print_results(color_pixels, color_codes):
    print(f"\n{len(color_pixels)} COLOR RESULTS:\n")
    for (color_name, rgb_val), (_, pixels) in zip(color_codes, color_pixels):
        print(f"{color_name}, {rgb_val}, {pixels} pixels")


def show_graph(color_pixels, color_codes):
    color_graph_data = []
    for (color_name, rgb_val), (_, pixels) in zip(color_codes, color_pixels):
        color_graph_data.append((color_name, rgb_to_hex(rgb_val), pixels))

    pie_labels = [f"{i[0]}, {i[1]}" for i in color_codes]
    colors = [i[1] for i in color_graph_data]
    pixels = [i[2] for i in color_graph_data]

    plt.rcParams['figure.figsize'] = [12.8, 7.2]
    plt.pie(pixels, labels=pie_labels, colors=colors)
    plt.title("Color distribution", fontdict={
              "fontsize": 14, "fontweight": "bold"})
    plt.get_current_fig_manager().set_window_title("Color distribution")
    plt.show()


def main():
    model_path = 'color_classifier.pth'
    label_encoder_path = 'label_encoder.pkl'
    image_path = 'image.jpg'
    csv_file = "./colors.csv"

    num_classes = 0
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
        num_classes = len(label_encoder.classes_)

    model = ColorClassifier(3, num_classes)
    model.load_state_dict(torch.load(model_path))

    image_colors = get_image_colors(image_path, model, label_encoder)

    color_pixels = sorted(image_colors.items(),
                          key=lambda x: x[1], reverse=True)

    color_codes = get_rgb_from_csv([color[0]
                                    for color in color_pixels], csv_file)

    print_results(color_pixels, color_codes)
    show_graph(color_pixels, color_codes)


if __name__ == '__main__':
    main()
