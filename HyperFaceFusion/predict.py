# filename: create_database.py
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import json
import pandas as pd

# Import necessary components from your model and dataset files
from HyperFaceFusion.model import HyperFacePipeline
from HyperFaceFusion.dataset import HyperspectralFaceDataset

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_classes_actual = 113 # Number of actual classes in your dataset
model_path = 'hyperface_best_model.pth' # Path to your saved best model weights
database_filename = "database.json" # Filename for the database

# --- Main function to extract and save the database ---
def extract_and_save_database(model, full_dataset, database_filename, device):
    """
    Runs inference on the entire dataset, computes average embeddings and fused_faces
    per ID, normalizes embeddings, and saves them to a database.json file.
    """
    print(f"\nStarting data extraction and processing for database on {device}...")
    model.eval() # Set the model to evaluation mode

    all_embeddings = []
    all_fused_faces = []
    all_labels = []

    # Using batch_size=1 to easily get individual labels for each sample
    full_loader = DataLoader(full_dataset, batch_size=1, shuffle=False)

    data_bar = tqdm(full_loader, desc=f"[Extracting & Processing Database]")
    with torch.no_grad():
        for batch_idx, (ir_val, vis_val, labels_val) in enumerate(data_bar):
            ir_val, vis_val = ir_val.to(device), vis_val.to(device)
            label = labels_val.cpu().item() # Get the scalar label value from the tensor

            embeddings, fused_face = model(ir_val, vis_val)

            all_embeddings.append(embeddings.cpu().squeeze(0).numpy())
            # Squeeze fused_face to remove channel dimension if it's 1, then convert to numpy
            all_fused_faces.append(fused_face.cpu().squeeze(0).numpy())
            all_labels.append(label)

            data_bar.set_postfix({"Sample ID": f"{label} (batch {batch_idx+1})"})

    print(f"\nData extraction complete! Starting aggregation and averaging...")

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'label': all_labels,
        'embedding': all_embeddings,
        'fused_face': all_fused_faces
    })

    # Group by 'label' and compute the mean
    # For numpy arrays in columns, np.mean(list(x), axis=0) computes element-wise mean
    aggregated_results = df.groupby('label').agg(
        embedding_mean=('embedding', lambda x: np.mean(list(x), axis=0)),
        fused_face_mean=('fused_face', lambda x: np.mean(list(x), axis=0))
    ).reset_index()

    # Normalize the average embeddings to unit vectors
    def normalize_embedding(embedding_array):
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            return embedding_array # Avoid division by zero or for zero vectors
        return embedding_array / norm

    aggregated_results['embedding_normalized'] = aggregated_results['embedding_mean'].apply(normalize_embedding)

    final_json_output = {}
    
    # Using the numeric label directly as the key for the database
    # This aligns with the user's clarification that the label IS the label.
    for index, row in aggregated_results.iterrows():
        numeric_label = row['label'] 
        person_id_key = str(numeric_label) # Convert the numeric label to string to be used as a JSON key
        
        final_json_output[person_id_key] = {
            "embeddings_mean": row['embedding_mean'].tolist(),
            "embeddings_normalized": row['embedding_normalized'].tolist(),
            "fused_face_mean": row['fused_face_mean'].tolist()
        }

    with open(database_filename, 'w') as f:
        json.dump(final_json_output, f, indent=4)

    print(f"Results (embeddings and fused faces) grouped, averaged, normalized, and saved to '{database_filename}'")
    return final_json_output # Return the saved data for potential further use

if __name__ == '__main__':
    # --- Initialize and load model ---
    model_instance = HyperFacePipeline(num_classes=num_classes_actual).to(device)
    if os.path.exists(model_path):
        print(f"Loading saved model from '{model_path}'...")
        model_instance.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("Model loaded successfully.")
    else:
        print(f"Error: Saved model not found at '{model_path}'. Please train the model first.")
        exit()

    # --- Load the full dataset ---
    transform = transforms.ToTensor()
    full_dataset_instance = HyperspectralFaceDataset("../data/RGB_Thermal/rgb", "../data/RGB_Thermal/thermal", transform)

    # --- Run the database creation process ---
    extract_and_save_database(model_instance, full_dataset_instance, database_filename, device)
    print("\nDatabase creation script finished.")
