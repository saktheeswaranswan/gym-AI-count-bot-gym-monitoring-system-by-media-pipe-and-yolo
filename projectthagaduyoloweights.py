import numpy as np
import csv

def read_yolo_weights(file_path):
    with open(file_path, 'rb') as f:
        header = f.read(4 * 5)  # Skip the header
        weights = np.fromfile(f, dtype=np.float32)  # Read weights
    return weights

def get_layer_shapes():
    # Define the shapes for each layer in YOLOv3-tiny
    layer_shapes = [
        (3 * 3 * 3 * 16, 'conv1'),     # Conv Layer 1
        (3 * 3 * 16 * 32, 'conv2'),    # Conv Layer 2
        (3 * 3 * 32 * 64, 'conv3'),     # Conv Layer 3
        (3 * 3 * 64 * 128, 'conv4'),    # Conv Layer 4
        (3 * 3 * 128 * 256, 'conv5'),   # Conv Layer 5
        (3 * 3 * 256 * 512, 'conv6'),   # Conv Layer 6
        (3 * 3 * 512 * 1024, 'conv7'),  # Conv Layer 7
        (1 * 1 * 512 * 256, 'conv8'),    # Conv Layer 8
        (3 * 3 * 256 * 512, 'conv9'),    # Conv Layer 9
        (1 * 1 * 512 * 255, 'conv10'),   # Conv Layer 10 (final layer before YOLO)
    ]
    return layer_shapes

def save_weights_to_csv(weights, layer_shapes, output_file):
    index = 0
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers for the CSV (Layer Name and then matrix values)
        writer.writerow(["Layer Name", "Matrix"])
        
        for shape, layer_name in layer_shapes:
            layer_weights = weights[index:index + shape].reshape(-1)
            index += shape
            
            # Reshape into 20x20 matrices, filling with zeros if necessary
            if len(layer_weights) % 400 != 0:  # If the number of weights is not a perfect multiple of 20x20
                padding = 400 - (len(layer_weights) % 400)
                layer_weights = np.pad(layer_weights, (0, padding), 'constant', constant_values=0)

            # Now reshape the layer weights into 20x20 matrices
            matrix = layer_weights.reshape(-1, 400)[:20]
            
            # Write layer name
            writer.writerow([layer_name])

            # Write matrix values row by row with 15 decimal precision
            for row in matrix:
                writer.writerow([f"{value:.15f}" for value in row])

def main():
    file_path = 'yolov3-tiny.weights'  # Path to your weights file
    weights = read_yolo_weights(file_path)

    layer_shapes = get_layer_shapes()

    # Output CSV file
    output_file = 'yolov3_tiny_weights.csv'
    
    # Save weights to CSV
    save_weights_to_csv(weights, layer_shapes, output_file)

if __name__ == '__main__':
    main()

