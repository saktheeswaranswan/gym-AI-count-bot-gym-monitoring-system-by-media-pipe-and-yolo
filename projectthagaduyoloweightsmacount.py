import numpy as np
import csv

def read_yolo_weights(file_path):
    """Reads the YOLO weights from a binary file."""
    with open(file_path, 'rb') as f:
        header = f.read(4 * 5)  # Skip the header (first 5 integers)
        weights = np.fromfile(f, dtype=np.float32)  # Read all weights as numpy array
    return weights

def get_layer_shapes():
    """Defines the shape of each layer's weights in YOLOv3-tiny."""
    layer_shapes = [
        (3 * 3 * 3 * 16, 'conv1'),     # Conv Layer 1
        (3 * 3 * 16 * 32, 'conv2'),    # Conv Layer 2
        (3 * 3 * 32 * 64, 'conv3'),    # Conv Layer 3
        (3 * 3 * 64 * 128, 'conv4'),   # Conv Layer 4
        (3 * 3 * 128 * 256, 'conv5'),  # Conv Layer 5
        (3 * 3 * 256 * 512, 'conv6'),  # Conv Layer 6
        (3 * 3 * 512 * 1024, 'conv7'), # Conv Layer 7
        (1 * 1 * 512 * 256, 'conv8'),  # Conv Layer 8
        (3 * 3 * 256 * 512, 'conv9'),  # Conv Layer 9
        (1 * 1 * 512 * 255, 'conv10'), # Conv Layer 10
    ]
    return layer_shapes

def print_layer_weights(weights, shape, layer_name):
    """Prints the layer weights as 20x20 matrices with 100 decimal places."""
    layer_weights = weights[:shape]
    weights = weights[shape:]
    
    # Reshape the layer weights into a 20x20 matrix
    num_elements = len(layer_weights)
    
    # Reshape into a 20x20 matrix, pad with zeros if necessary
    if num_elements % 400 != 0:
        padding = 400 - (num_elements % 400)  # Add padding to make the size multiple of 400
        layer_weights = np.pad(layer_weights, (0, padding), 'constant', constant_values=0)
    
    matrix = layer_weights.reshape(-1, 400)[:20]  # Reshape into 20 rows, 400 elements per row (20x20)
    
    # Print layer name
    print(f"Layer: {layer_name}")
    
    # Print each row of the matrix with 100 decimal places
    for row in matrix:
        print(" ".join([f"{value:.100f}" for value in row]))

def main():
    # Path to your YOLOv3-tiny weights file
    file_path = 'yolov3-tiny.weights'
    
    # Read the weights from the file
    weights = read_yolo_weights(file_path)
    
    # Get the layer shapes (dimensions of weights for each layer)
    layer_shapes = get_layer_shapes()
    
    index = 0
    for shape, layer_name in layer_shapes:
        print_layer_weights(weights[index:index + shape], shape, layer_name)
        index += shape  # Update the index for the next layer

if __name__ == '__main__':
    main()

