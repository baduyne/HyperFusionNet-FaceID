from HyperFaceFusion import HyperFaceFusionAPI

# Initialize the HyperFaceFusion API
hyperfacefusion_api = HyperFaceFusionAPI()

def matcher_service(ir_file, vis_file):
	response = hyperfacefusion_api.run(ir_file, vis_file)
	return response
