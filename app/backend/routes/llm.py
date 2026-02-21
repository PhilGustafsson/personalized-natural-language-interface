"""
Routes for LLM-based prompt generation.
Allows generating domain-specific training data for STT/TTS models.
"""

from flask import Blueprint, request, jsonify
import requests
import json

llm_bp = Blueprint('llm', __name__)

LLM_SERVICE_URL = "http://llm-app:8001"


@llm_bp.route("/generate-prompts/", methods=["POST"])
def generate_prompts():
    """
    Generate domain-specific prompts using LLM service.
    
    Expected JSON body:
    {
        "domain": "AI",
        "num_prompts": 20,
        "language": "sv",
        "difficulty": "intermediate",
        "sentence_length": "medium",
        "include_technical_terms": true,
        "style": "conversational"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'domain' not in data:
            return jsonify({"error": "domain is required"}), 400
        
        # Forward to LLM service
        response = requests.post(
            f"{LLM_SERVICE_URL}/generate_prompts",
            json=data,
            timeout=120  # LLM generation can take time
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        elif response.status_code == 503:
            return jsonify({
                "error": "LLM service not ready",
                "message": "Model may still be loading. Please try again in a moment."
            }), 503
        else:
            return jsonify({
                "error": "LLM service error",
                "details": response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to LLM service",
            "message": "LLM service may not be running"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Failed to generate prompts",
            "details": str(e)
        }), 500


@llm_bp.route("/load-model/", methods=["POST"])
def load_model():
    """
    Load LLM model.
    
    Optional JSON body:
    {
        "model_name": "Qwen/Qwen2.5-8B-Instruct",
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9
    }
    """
    try:
        data = request.get_json() or {}
        
        response = requests.post(
            f"{LLM_SERVICE_URL}/load_model",
            json=data,
            timeout=300  # Model loading can take several minutes
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": "Failed to load model",
                "details": response.text
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({
            "error": "Model loading timeout",
            "message": "Model loading is taking longer than expected. It may still be loading in the background."
        }), 504
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to LLM service",
            "message": "LLM service may not be running"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Failed to load model",
            "details": str(e)
        }), 500


@llm_bp.route("/health/", methods=["GET"])
def health():
    """Check LLM service health and model status."""
    try:
        response = requests.get(f"{LLM_SERVICE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": "LLM service unhealthy",
                "details": response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "unavailable",
            "model_loaded": False,
            "message": "LLM service not reachable"
        }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "details": str(e)
        }), 500


@llm_bp.route("/unload-model/", methods=["POST"])
def unload_model():
    """Unload LLM model to free VRAM."""
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/unload_model",
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                "error": "Failed to unload model",
                "details": response.text
            }), response.status_code
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            "error": "Cannot connect to LLM service",
            "message": "LLM service may not be running"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Failed to unload model",
            "details": str(e)
        }), 500


@llm_bp.route("/save-generated-prompts/", methods=["POST"])
def save_generated_prompts():
    """
    Save generated prompts to a domain file for later use.
    
    Expected JSON body:
    {
        "domain_name": "ai_generated",
        "prompts": [
            {"id": 1, "text": "..."},
            {"id": 2, "text": "..."}
        ],
        "language": "sv"
    }
    """
    try:
        import os
        from db import DATA_DIR
        
        data = request.get_json()
        
        if not data or 'domain_name' not in data or 'prompts' not in data:
            return jsonify({"error": "domain_name and prompts are required"}), 400
        
        domain_name = data['domain_name']
        prompts = data['prompts']
        language = data.get('language', 'sv')
        
        # Create domain file in assets/domains
        assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'domains')
        os.makedirs(assets_dir, exist_ok=True)
        
        filename = f"domain_{domain_name}_generated.json"
        filepath = os.path.join(assets_dir, filename)
        
        # Format as standard domain file
        domain_data = {
            "name": domain_name,
            "language": language,
            "prompts": prompts,
            "generated": True,
            "source": "llm"
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(domain_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "message": "Prompts saved successfully",
            "filepath": filepath,
            "filename": filename,
            "total_prompts": len(prompts)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": "Failed to save prompts",
            "details": str(e)
        }), 500
