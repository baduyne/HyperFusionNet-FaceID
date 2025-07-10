import torch
import numpy as np
import os
import json
import cv2

# Preprocessing ảnh thành tensor [1,1,H,W]
def preprocess_single_image(img_path, img_size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {img_path}")
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor

# Load embedding chuẩn hóa từ file JSON
def load_database_embeddings(filename):
    if not os.path.exists(filename):
        print(f"Error: Database file '{filename}' not found.")
        return None

    with open(filename, 'r') as f:
        data = json.load(f)

    db_embeddings = {}
    for person_id, values in data.items():
        if "embeddings_normalized" in values:
            db_embeddings[person_id] = np.array(values["embeddings_normalized"])
        else:
            print(f"Warning: ID '{person_id}' lacks 'embeddings_normalized'. Skipped.")
    return db_embeddings

# Tìm người giống nhất
def find_match(model, ir_path, vis_path, database_embeddings, device, threshold=0.7):
    if not database_embeddings:
        return "Database is empty."

    ir_tensor = preprocess_single_image(ir_path).to(device)
    vis_tensor = preprocess_single_image(vis_path).to(device)

    model.eval()
    with torch.no_grad():
        embedding, _ = model(ir_tensor, vis_tensor)

    embedding_np = embedding.squeeze(0).cpu().numpy()
    norm = np.linalg.norm(embedding_np)
    if norm == 0:
        return "Zero vector embedding."
    input_embedding = embedding_np / norm

    max_similarity = -1.0
    matched_id = "Not Found"

    for person_id, db_embedding in database_embeddings.items():
        similarity = np.dot(input_embedding, db_embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            matched_id = person_id

    if max_similarity >= threshold:
        print(f"[MATCHED] ID: {matched_id}, similarity: {max_similarity:.4f}")
        return matched_id, max_similarity
    else:
        print(f"[NO MATCH] Highest similarity: {max_similarity:.4f}")
        return "Not Found", max_similarity

# def run():
#     # Load configuration
#     with open('configs.yaml', 'r') as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)  # Use safe_load for security\
#     model_path = config['model']['path']
#     database_filename = config['embedding']['path']

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     model = HyperFacePipeline().to(device)
#     if os.path.exists(model_path):
#         print(f"Loading model from '{model_path}'...")
#         model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#         print("Model loaded.")
#     else:
#         print(f"Model file not found at '{model_path}'.")
#         exit()

#     database = load_database_embeddings(database_filename)
#     if not database:
#         print("Failed to load database.")
#         exit()
#     print(f"Loaded {len(database)} identities from database.")

#     sample_ir = "data/RGB_Thermal/thermal/7-TD-A-2.jpg"
#     sample_vis = "data/RGB_Thermal/rgb/7-TD-A-2.jpg"

#     if not os.path.exists(sample_ir) or not os.path.exists(sample_vis):
#         print("Missing input images for testing.")
#         print(f"IR image: {sample_ir}")
#         print(f"VIS image: {sample_vis}")
#         exit()

#     result = find_match(model, sample_ir, sample_vis, database, device, threshold=0.7)
#     print(f"\nFinal Match Result: {result}")

# ================= MAIN =====================
# if __name__ == '__main__':
#     run()