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

def calculate_gradients(layer_weights):
    """Calculate a simple measure of vanishing/exploding gradients for a given weight matrix."""
    # Example calculation of gradient based on weight variance (this is a simplification)
    weight_variance = np.var(layer_weights)  # Variance of the weights as a proxy for gradient magnitude
    
    if weight_variance < 1e-4:
        return "😴"  # Low variance indicates potential vanishing gradient (sleepy emoji)
    elif weight_variance > 1e+4:
        return "💥"  # High variance indicates potential exploding gradient (exploding emoji)
    else:
        return "😊"  # Otherwise, normal gradient behavior (smiling emoji)

def save_layer_weights_and_analysis_to_csv(weights, shape, layer_name, output_file):
    """Saves the layer weights and gradient analysis into a CSV file."""
    layer_weights = weights[:shape]
    weights = weights[shape:]
    
    # Reshape the layer weights into a 20x20 matrix
    num_elements = len(layer_weights)
    
    # Reshape into a 20x20 matrix, pad with zeros if necessary
    if num_elements % 400 != 0:
        padding = 400 - (num_elements % 400)  # Add padding to make the size multiple of 400
        layer_weights = np.pad(layer_weights, (0, padding), 'constant', constant_values=0)
    
    matrix = layer_weights.reshape(-1, 400)[:20]  # Reshape into 20 rows, 400 elements per row (20x20)

    # Calculate gradient behavior
    gradient_behavior = calculate_gradients(layer_weights)
    
    # Save to a CSV file with side-by-side weights and gradient analysis
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header if the file is new
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writerow(["Layer", "Weight Value", "Gradient Analysis"])
        
        # Write weights and gradient analysis row by row
        for row in matrix:
            for value in row:
                writer.writerow([layer_name, f"{value:.100f}", gradient_behavior])

def main():
    # Path to your YOLOv3-tiny weights file
    file_path = 'yolov3-tiny.weights'
    
    # Read the weights from the file
    weights = read_yolo_weights(file_path)
    
    # Get the layer shapes (dimensions of weights for each layer)
    layer_shapes = get_layer_shapes()
    
    # Output file to store all weights and gradient analysis
    output_file = 'yolov3_tiny_weights_and_gradients.csv'
    
    # Process each layer and save the weights and gradient analysis
    index = 0
    for shape, layer_name in layer_shapes:
        save_layer_weights_and_analysis_to_csv(weights[index:index + shape], shape, layer_name, output_file)
        index += shape  # Update the index for the next layer

if __name__ == '__main__':
    main()

