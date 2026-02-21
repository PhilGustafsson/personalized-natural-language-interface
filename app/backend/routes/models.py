from flask import Blueprint, request, jsonify
from db import DATA_DIR, get_db
import os
import time

models_bp = Blueprint('models', __name__)


# todo: actually download models. (hook up to services).
# todo: actually get progress of download.

# -- download models --

download_progress = {"status": "idle", "progress": 0}

@models_bp.route("/status/", methods=["GET"])
def check_models_status():
    """
    Check if the TTS and STT models have been downloaded.
    """
    return jsonify({ # tmp: dummy code for now.
        "tts": False,
        "stt": True
    }), 200


@models_bp.route("/download/", methods=["POST"])
def start_download():
    """
    Download the TTS and STT models. 
    
    For now this is just a dummy process.
    """
    global download_progress
    download_progress = {"status": "downloading", "progress": 0}

    # Simulate async download in background (thread)
    import threading

    def simulate_download():
        global download_progress
        for i in range(1, 101):
            time.sleep(0.01)  # simulate work
            download_progress["progress"] = i
        download_progress["status"] = "done"

    threading.Thread(target=simulate_download).start()

    return jsonify({"message": "Download started."})

@models_bp.route("download/progress/", methods = ["GET"])
def get_download_progress():
    return jsonify(download_progress)