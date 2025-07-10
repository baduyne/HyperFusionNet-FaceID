from fastapi import APIRouter, UploadFile, File
import shutil
import os
import uuid
from server.services.matcher import matcher_service

router = APIRouter()

@router.post("/match/")
async def match_endpoint(
	ir_img: UploadFile = File(...),
	vis_img: UploadFile = File(...)
):
	try:
		temp_dir = "./temp"
		os.makedirs(temp_dir, exist_ok=True)

		# Save files temporarily
		ir_path = os.path.join(temp_dir, f"ir_{uuid.uuid4().hex}.jpg")
		vis_path = os.path.join(temp_dir, f"vis_{uuid.uuid4().hex}.jpg")

		with open(ir_path, "wb") as f:
			shutil.copyfileobj(ir_img.file, f)
		with open(vis_path, "wb") as f:
			shutil.copyfileobj(vis_img.file, f)
	except Exception as e:
		ValueError(f"Error saving images: {e}")

	result = matcher_service(ir_path, vis_path)

	# Clean temp
	os.remove(ir_path)
	os.remove(vis_path)

	return result