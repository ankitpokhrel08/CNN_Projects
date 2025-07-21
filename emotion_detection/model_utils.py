import gzip
import os
import pickle

def compress_file_into_two_parts(input_file_path, output_part1_path, output_part2_path):
    """
    Compress a large file into two smaller compressed parts.
    """
    print(f"Compressing {input_file_path} into two parts...")
    
    with open(input_file_path, 'rb') as f_in:
        data = f_in.read()

    mid_point = len(data) // 2
    part1_data = data[:mid_point]
    part2_data = data[mid_point:]

    print(f"Original size: {len(data):,} bytes")
    print(f"Part 1 size: {len(part1_data):,} bytes")
    print(f"Part 2 size: {len(part2_data):,} bytes")

    compressed_part1 = gzip.compress(part1_data)
    compressed_part2 = gzip.compress(part2_data)

    print(f"Compressed part 1 size: {len(compressed_part1):,} bytes")
    print(f"Compressed part 2 size: {len(compressed_part2):,} bytes")

    with open(output_part1_path, 'wb') as f_out1:
        f_out1.write(compressed_part1)

    with open(output_part2_path, 'wb') as f_out2:
        f_out2.write(compressed_part2)
    
    print(f"Successfully created {output_part1_path} and {output_part2_path}")

def decompress_two_parts_to_file(input_part1_path, input_part2_path, output_file_path):
    """
    Decompress two compressed parts back into the original file.
    """
    print(f"Decompressing {input_part1_path} and {input_part2_path} to {output_file_path}...")
    
    with open(input_part1_path, 'rb') as f_in1:
        compressed_part1 = f_in1.read()

    with open(input_part2_path, 'rb') as f_in2:
        compressed_part2 = f_in2.read()

    decompressed_part1 = gzip.decompress(compressed_part1)
    decompressed_part2 = gzip.decompress(compressed_part2)

    combined_data = decompressed_part1 + decompressed_part2

    with open(output_file_path, 'wb') as f_out:
        f_out.write(combined_data)
    
    print(f"Successfully decompressed to {output_file_path}")
    print(f"Final size: {len(combined_data):,} bytes")

def load_model_from_parts(part1_path='model_part1.pkl.gz', part2_path='model_part2.pkl.gz', temp_model_path='temp_model.pkl'):
    """
    Load a model by first decompressing it from two parts.
    Returns the loaded model object.
    """
    if not os.path.exists(part1_path) or not os.path.exists(part2_path):
        raise FileNotFoundError(f"Model parts not found: {part1_path} or {part2_path}")
    
    # Decompress to temporary file
    decompress_two_parts_to_file(part1_path, part2_path, temp_model_path)
    
    # Load the model
    with open(temp_model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Clean up temporary file
    os.remove(temp_model_path)
    
    return model

if __name__ == "__main__":
    # Script to compress the existing model.pkl
    input_file = "model.pkl"
    part1_file = "model_part1.pkl.gz"
    part2_file = "model_part2.pkl.gz"
    
    if os.path.exists(input_file):
        compress_file_into_two_parts(input_file, part1_file, part2_file)
        print(f"\nCompression complete!")
        print(f"You can now commit {part1_file} and {part2_file} to GitHub")
        print(f"Original file {input_file} can be deleted or added to .gitignore")
    else:
        print(f"Error: {input_file} not found")
