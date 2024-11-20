# inspect_model.py

import tensorflow as tf

def inspect_saved_model(model_dir):
    """
    Loads the SavedModel and prints the input and output tensor names.

    Args:
        model_dir (str): Path to the SavedModel directory.
    """
    # Load the SavedModel
    print(f"Loading model from directory: {model_dir}")
    model = tf.saved_model.load(model_dir)

    # List all available signatures
    print("\nAvailable Signatures:")
    for key in model.signatures.keys():
        print(f" - {key}")

    # Inspect each signature
    for signature_key in model.signatures:
        signature = model.signatures[signature_key]
        print(f"\nSignature: {signature_key}")

        # Print input tensors
        print("Inputs:")
        for input_key, tensor_spec in signature.structured_input_signature[1].items():
            print(f"  {input_key}:")
            print(f"    Shape: {tensor_spec.shape}")
            print(f"    DType: {tensor_spec.dtype}")
            print(f"    Name: {tensor_spec.name}")

        # Print output tensors
        print("Outputs:")
        for output_key, tensor in signature.structured_outputs.items():
            print(f"  {output_key}:")
            print(f"    Shape: {tensor.shape}")
            print(f"    DType: {tensor.dtype}")
            print(f"    Name: {tensor.name}")

if __name__ == "__main__":
    # Replace 'saved_movinet_model' with the path to your SavedModel directory if different
    model_directory = 'saved_movinet_model'
    inspect_saved_model(model_directory)
