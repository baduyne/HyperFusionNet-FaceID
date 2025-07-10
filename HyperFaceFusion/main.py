import os
import torch
import yaml

from HyperFaceFusion.model import HyperFacePipeline
from HyperFaceFusion.match_face import find_match, load_database_embeddings

class HyperFaceFusionAPI:
	"""
	HyperFaceFusion API for face matching.
	This class initializes the model and database, and provides a method to run face matching.
	"""

	def __init__(self):
		# Load configuration
		self.config = self.load_config('HyperFaceFusion/configs.yaml')
		self.model_path = self.config['model']['path']
		self.db_path = self.config['embedding']['path']
		self.threshold = self.config['threshold']['value']

		# Load model & database
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")

		self.model = HyperFacePipeline().to(self.device)
		if os.path.exists(self.model_path):
			print(f"Loading model from '{self.model_path}'...")
			self.model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
			print("Model loaded.")
		else:
			print(f"Model file not found at '{self.model_path}'.")
			exit()

		self.database = load_database_embeddings(self.db_path)
		if not self.database:
			print("Failed to load database.")
			exit()
		print(f"Loaded {len(self.database)} identities from database.")

	def load_config(self, config_file):
		with open(config_file, 'r') as file:
			return yaml.load(file, Loader=yaml.FullLoader)  # Use safe_load for security

	def run(self, ir_path, vis_path):
		result, score = find_match(self.model, ir_path, vis_path, self.database, self.device, self.threshold)
		return {
			"match_id": result,
			"similarity": round(score, 4) if score is not None else None
		}

# Endpoint
# @app.post("/match/")
# async def match_faces(ir_img: UploadFile = File(...), vis_img: UploadFile = File(...)):
#     temp_dir = "./temp"
#     os.makedirs(temp_dir, exist_ok=True)

#     # Save files temporarily
#     ir_path = os.path.join(temp_dir, f"ir_{uuid.uuid4().hex}.jpg")
#     vis_path = os.path.join(temp_dir, f"vis_{uuid.uuid4().hex}.jpg")

#     with open(ir_path, "wb") as f:
#         shutil.copyfileobj(ir_img.file, f)
#     with open(vis_path, "wb") as f:
#         shutil.copyfileobj(vis_img.file, f)

#     try:
#         ir_tensor = preprocess_single_image(ir_path).to(device)
#         vis_tensor = preprocess_single_image(vis_path).to(device)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})

#     person_id, similarity = find_match(model, ir_tensor, vis_tensor, database, device)

#     # Clean temp
#     os.remove(ir_path)
#     os.remove(vis_path)

#     return {
#         "match_id": person_id,
#         "similarity": round(similarity, 4)
#     }