#!/bin/bash

# === CONFIG ===
MODEL_URL="https://drive.usercontent.google.com/download?id=1_R2ZqjwkSfopc2QZB9i39vduAK8SAa3K&export=download&authuser=0&confirm=t&uuid=526ef9db-b4fe-4bf8-abf7-1c5de2416b1d&at=AN8xHoqatNUdApZkP0XouE89fFmX%3A1751391475810"
DB_URL="https://drive.usercontent.google.com/download?id=1_b7h1joLTowbKSsqmzRE2Rc8wuJYSFLv&export=download"
MODEL_ZIP="hyperface_best_model.zip"
DB_ZIP="database.zip"

MODEL_PATH="HyperFaceFusion/weights/hyperface_best_model.pth"
DB_PATH="HyperFaceFusion/weights/embedding/database.json"

# === STEP 1: CHECK & DOWNLOAD MODEL ZIP ===
if [ ! -f "$MODEL_PATH" ]; then
    echo " Downloading model file..."
    python -m wget -o "$MODEL_PATH" "$MODEL_URL" || { echo " Failed to download model."; exit 1; }
    # echo "Unzipping model..."
    # unzip -o "$MODEL_ZIP" -d "$MODEL_PATH" || { echo " Unzip failed."; exit 1; }
else
    echo " Model already exists: $MODEL_PATH"
fi

# === STEP 2: CHECK & DOWNLOAD DATABASE ZIP ===
if [ ! -f "$DB_PATH" ]; then
    echo " Downloading database file..."
    python -m wget -o "$DB_PATH" "$DB_URL" || { echo " Failed to download database."; exit 1; }
    # echo " Unzipping database..."
    # unzip -o "$DB_ZIP" -d "$DB_PATH" || { echo " Unzip failed."; exit 1; }
else
    echo " Database already exists: $DB_PATH"
fi

# === STEP 3: START FASTAPI ===
echo "🚀 Starting FastAPI..."
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# === STEP 4: START STREAMLIT ===
echo "🌐 Starting Streamlit..."
streamlit run client/main.py &
STREAMLIT_PID=$!

# === STEP 5: SUCCESS MESSAGE ===
echo " All services are running!"
echo " FastAPI:    http://localhost:8000/docs"
echo " Streamlit:  http://localhost:8501"

# === WAIT FOR PROCESSES (Optional) ===
wait $FASTAPI_PID
wait $STREAMLIT_PID
