import numpy as np
import csv

def read_yolo_weights(file_path):
    """Reads the YOLO weights from a binary file."""
    with open(file_path, 'rb') as f:
        # Skip header (first 5 integers)
        header = f.read(4 * 5)
        weights = np.fromfile(f, dtype=np.float32)  # Load all weights as a numpy array
    return weights

def get_layer_shapes():
    """
    Define the shape of each layer's weights for YOLOv3-spp.
    Update these placeholder values to reflect the true layer structure.
    Each tuple is (number_of_weights, layer_name).
    """
    layer_shapes = [
        (3 * 3 * 3 * 32, 'conv1'),
        (3 * 3 * 32 * 64, 'conv2'),
        (3 * 3 * 64 * 32, 'conv3'),
        (3 * 3 * 32 * 64, 'conv4'),
        (3 * 3 * 64 * 128, 'conv5'),
        (3 * 3 * 128 * 256, 'conv6'),
        # ... (add all the layers based on your YOLOv3-spp architecture)
        (1 * 1 * 512 * 255, 'conv_last')  # placeholder for the final convolutional layer
    ]
    return layer_shapes

def calculate_gradients(layer_weights):
    """
    Calculate a simple measure of gradient behavior based on weight variance.
    Returns:
      😢 for vanishing gradient (very low variance),
      💥 for exploding gradient (very high variance),
      😆 for normal gradient behavior.
    """
    variance = np.var(layer_weights)
    if variance < 1e-4:
        return "😢"  # Vanishing gradient
    elif variance > 1e+4:
        return "💥"  # Exploding gradient
    else:
        return "😆"  # Normal gradient

def save_layer_weights_and_analysis_to_csv(weights, shape, layer_name, output_file):
    """
    Save the given layer's weights (formatted as a 20x20 matrix) and
    its gradient analysis to the output CSV file.
    Returns any unused weights from this slice (if any).
    """
    layer_weights = weights[:shape]
    
    # Determine number of elements and pad if necessary to fill 20x20 matrices (each matrix = 400 values)
    num_elements = len(layer_weights)
    if num_elements % 400 != 0:
        padding = 400 - (num_elements % 400)
        layer_weights = np.pad(layer_weights, (0, padding), 'constant', constant_values=0)
    
    # Reshape into a matrix with 20 columns per row; take the first 20 rows.
    matrix = layer_weights.reshape(-1, 400)[:20]
    
    # Calculate gradient analysis using weight variance.
    gradient = calculate_gradients(layer_weights)
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header if file is empty.
        csvfile.seek(0, 2)
        if csvfile.tell() == 0:
            writer.writerow(["Layer", "Weight Value", "Gradient"])
        # Write each weight value along with layer name and gradient emoji.
        for row in matrix:
            for value in row:
                writer.writerow([layer_name, f"{value:.100f}", gradient])
    
    # Return remaining weights (if needed).
    return weights[shape:]

def main():
    # Path to your YOLOv3-spp weights file.
    file_path = 'yolov3-spp.weights'
    weights = read_yolo_weights(file_path)
    
    # Get the layer shapes for YOLOv3-spp. Update this list as necessary.
    layer_shapes = get_layer_shapes()
    
    # Output CSV file which will contain the weights and gradient analysis.
    output_file = 'yolov3_spp_weights_and_gradients.csv'
    
    index = 0
    # Process each layer sequentially.
    for shape, layer_name in layer_shapes:
        # Process slice of weights for the current layer.
        slice_weights = weights[index:index + shape]
        # Save the layer's weights and gradient analysis to CSV.
        _ = save_layer_weights_and_analysis_to_csv(slice_weights, shape, layer_name, output_file)
        index += shape

if __name__ == '__main__':
    main()

