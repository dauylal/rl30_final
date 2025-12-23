import numpy as np
import os

def calculate_absolute_difference(file1_path, file2_path):
    """
    Loads two .npy files, calculates the absolute difference between the 
    contained arrays, and prints the result.

    Args:
        file1_path (str): The path to the first .npy file.
        file2_path (str): The path to the second .npy file.
    """
    # 1. Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File not found at {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Error: File not found at {file2_path}")
        return

    try:
        # 2. Load the NumPy arrays from the .npy files
        array1 = np.load(file1_path)
        array2 = np.load(file2_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 3. Check if arrays have compatible shapes for subtraction
    if array1.shape != array2.shape:
        print(f"Error: Arrays must have the same shape for element-wise subtraction.")
        print(f"Array 1 shape: {array1.shape}")
        print(f"Array 2 shape: {array2.shape}")
        return

    # 4. Calculate the absolute difference element-wise
    # This uses the numpy.absolute function which is equivalent to np.abs()
    # It calculates |array1[i, j, ...] - array2[i, j, ...]| for every element.
    absolute_difference = np.absolute(array1 - array2)

    # 5. Print results (or you could save the result with np.save)
    print("--- Analysis Complete ---")
    print(f"Array 1 Shape: {array1.shape}")
    print(f"Array 2 Shape: {array2.shape}")
    print("\nAbsolute Difference Array (first 5 elements/rows):")
    # Print a snippet of the result, depending on the array's dimensionality
    if absolute_difference.ndim > 0 and absolute_difference.shape[0] > 0:
        print(absolute_difference[tuple([slice(None, 5)] + [slice(None)] * (absolute_difference.ndim - 1))])
    else:
        print(absolute_difference)
    
    # You might also want a summary metric, like the maximum absolute difference:
    max_abs_diff = np.max(absolute_difference)
    print(f"\nMaximum Absolute Difference: {max_abs_diff}")
    
    # Or the sum of all absolute differences:
    sum_abs_diff = np.sum(absolute_difference)
    print(f"Sum of Absolute Differences: {sum_abs_diff}")


# --- Example Usage ---
# NOTE: You must replace 'path/to/file1.npy' and 'path/to/file2.npy' 
# with the actual paths to your files.

# Create dummy files for demonstration if they don't exist
calculate_absolute_difference("motionA/0_out_feats.npy", "motionB/0_out_feats.npy")