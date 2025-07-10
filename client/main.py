import streamlit as st
import requests
import tempfile
import os

st.set_page_config(page_title="IR-VIS Face Matching", layout="centered")

st.title("🧠 IR-VIS Face Recognition Demo")

st.markdown("Upload a **thermal (IR)** image and a **visible light (VIS)** image of the same person to find a matching identity.")

# Upload images
ir_file = st.file_uploader("Upload IR image", type=["jpg", "jpeg", "png"], key="ir")
vis_file = st.file_uploader("Upload VIS image", type=["jpg", "jpeg", "png"], key="vis")

match_button = st.button("🔍 Match Identity")

if match_button:
    if not ir_file or not vis_file:
        st.warning("Please upload both IR and VIS images.")
    else:
        # Save to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_ir:
            tmp_ir.write(ir_file.read())
            ir_path = tmp_ir.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_vis:
            tmp_vis.write(vis_file.read())
            vis_path = tmp_vis.name

        # Prepare request
        with open(ir_path, "rb") as ir_data, open(vis_path, "rb") as vis_data:
            files = {
                "ir_img": ("ir.jpg", ir_data, "image/jpeg"),
                "vis_img": ("vis.jpg", vis_data, "image/jpeg"),
            }
            try:
                response = requests.post("http://localhost:8000/match/", files=files)
                result = response.json()

                if response.status_code == 200:
                    st.success(f"✅ **Matched ID**: `{result['match_id']}`")
                    st.info(f"🧪 **Similarity**: `{result['similarity']}`")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to API server: {e}")

        os.remove(ir_path)
        os.remove(vis_path)