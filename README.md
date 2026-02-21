# 🧍 Personalized Natural Language Interface

A user interface application for fine-tuning open-source text-to-speech (TTS) and speech-to-text (STT) models to specific language domains, locally. The system allows users to create personalized voice profiles by providing easy methods for collecting speech data, fine-tuning models on their own voice, and evaluating the results.

## 📊 Evaluation Results

The system was evaluated on a Swedish language dataset, demonstrating that personalization leads to noticeable performance gains for both STT and TTS models.

### Speech-to-Text (STT) Performance
* The fine-tuned KB-whisper model achieved an initial Word Error Rate (WER) of 17.11% on 23 held-out Swedish audio samples.
* This represents a 2.89 percentage point improvement over the validation set's WER of 20.0%, indicating good generalization.
* Training was highly efficient via the Unsloth framework, converging in 7.6 epochs over just 102 seconds.
* Error analysis revealed that two-thirds of all errors (one third numeric formatting, one third punctuation) were not comprehension failures and can easily be addressed with post-processing. 
* **When adjusting for these easily fixable formatting errors, the effective semantic Word Error Rate drops to approximately 5.7%.**

### Text-to-Speech (TTS) Performance
* Fine-tuning the Chatterbox model decreased the average Character Error Rate (CER) from 0.124 to 0.057.
* The average Word Error Rate (WER) decreased from 0.237 to 0.116.
* In subjective evaluations, the overall Mean Opinion Score (MoS) improved significantly from 2.92 for the base model to 4.23 for the fine-tuned model.
* The greatest improvement was in voice naturalness, where the fine-tuned model scored 1.77 points higher than the base model.
* Artifacts and uncanny noises frequently produced by the base model were largely eliminated in the fine-tuned model.

## 🏗️ Architecture

The project consists of several Dockerized environments to encapsulate the business logic of each component.

* **Frontend (Angular):** User interface running locally in the browser for recording audio, viewing transcripts, and managing profiles.
* **Backend (Python Flask):** API server coordinating between services, preparing data for fine-tuning, and managing shared mounted volumes.
* **STT Service (FastAPI):** Speech-to-text transcription and fine-tuning using the **KB-Whisper-Large** model. It utilizes **Unsloth** for faster, less memory-intensive fine-tuning, **Optuna** for hyperparameter search, and **WhisperX** for real-time transcription with word-level timestamps.
* **TTS Service (FastAPI):** Text-to-speech synthesis, evaluation, and fine-tuning utilizing the **Chatterbox Multilingual** zero-shot cloning model. 
* **LLM Service (vLLM):** Domain-specific prompt generation running the **qwen3-8b** large language model.

## 💻 Installation Instructions

### Prerequisites

* Docker and Docker Compose
* NVIDIA GPU with CUDA support (for training)
* NVIDIA Container Toolkit

### Setup

1. Clone the repository
2. Build and start services:

```bash
docker-compose up --build
```

The services will be available at:
* Frontend: http://localhost:4200
* Backend API: http://localhost:5001
* STT Service: http://localhost:5080
* TTS Service: http://localhost:8002
* LLM Service: http://localhost:8001

## 🚀 Usage

### Domain-Specific Prompt Generation

The LLM service generates tailored training prompts for your specific domain, creating realistic sentences that users can record for STT/TTS training.


### Fine-tuning Workflow

Both STT and TTS follow a unified training workflow inside the personalization pipeline.

#### 1. Prepare Dataset

Record audio samples with corresponding transcripts. The system expects:
* Audio files in WAV format
* For STT: JSONL metadata file with `audio_path` and `text` fields
* For TTS: metadata.txt file with format `filename|transcript`

#### 2. Start Fine-tuning

**STT Fine-tuning:**
```bash
# Via backend API
POST http://localhost:5001/finetuning/start-stt/
{
  "profileID": "user123"
}

# Or directly to STT service
POST http://localhost:5080/load_dataset
{
  "manifest_path": "/app/data/profiles/user123/audio_prompts/metadata.jsonl",
  "recordings_root": "/app/data/profiles/user123/audio_prompts",
  "user": "user123"
}

POST http://localhost:5080/fine_tune
{
  "user": "user123",
  "num_train_epochs": 8,
  "learning_rate": 0.000165
}
```

**TTS Fine-tuning:**
```bash
# Via backend API
POST http://localhost:5001/finetuning/start-tts/
{
  "profileID": "user123"
}

# Or directly to TTS service
POST http://localhost:8000/load_dataset
{
  "manifest_path": "/app/data/profiles/user123/audio_prompts/metadata.txt",
  "recordings_root": "/app/data/profiles/user123/audio_prompts",
  "user": "user123"
}

POST http://localhost:8000/fine_tune
{
  "user": "user123",
  "num_train_epochs": 10,
  "learning_rate": 0.00001
}
```

#### 3. Monitor Progress

```bash
# Check job status via backend
GET http://localhost:5001/finetuning/status/{jobId}

# Or directly check service status
GET http://localhost:5080/job_status/{job_id}  # STT
GET http://localhost:8000/job_status/{job_id}  # TTS
```

## 🛠️ Development

### Project Structure

```text
├── app/
│   ├── backend/          # Python Flask API server
│   │   ├── routes/       # API endpoints
│   │   ├── services/     # Business logic (jobs, storage)
│   │   └── data/         # Shared data volume
│   └── frontend/         # Angular UI
├── stt/                  # STT service
│   ├── src/app/          # FastAPI + Unsloth/WhisperX training code
│   └── Dockerfile
├── tts/                  # TTS service
│   ├── app/              # FastAPI synthesis API
│   ├── src/              # Training service
│   ├── chatterbox/       # Chatterbox TTS scripts
│   └── Dockerfile
├── llm/                  # LLM service
│   ├── app/              # FastAPI + vLLM wrapper
│   └── Dockerfile
└── docker-compose.yml    # Service orchestration
```

## 📄 License
The project is licensed under the MIT license (https://opensource.org/license/mit).

## 🔮 Future Work

Due to the immense computational power required to host all models at once. The intended final product where both STT and TTS were finetuned at once from LLM generetaded prompts was never fully implemented. Finalizing this pipeline would be the main goal for future work.

> 💡 **Note:** This repo was previously on gitlab. Which explains the commit history.