# LLM Service - Domain-Specific Prompt Generation

FastAPI service using vLLM to generate domain-specific training prompts for STT/TTS model fine-tuning.

## Overview

This service wraps **Qwen2-7B-Instruct** via vLLM to efficiently generate realistic, domain-tailored sentences that users can record for training personalized speech models. Instead of using static pre-written prompts, users can generate custom training data for their specific field (AI, medicine, finance, etc.).

## Features

- **Domain-Specific Generation**: Creates sentences with relevant terminology and phrasing
- **Multilingual Support**: Swedish and English (extensible)
- **Difficulty Levels**: Beginner, intermediate, advanced technical complexity
- **Style Customization**: Conversational, formal, technical writing styles
- **Efficient Inference**: vLLM for high-throughput batch generation
- **GPU Accelerated**: CUDA 12.8 support

## API Endpoints

### Load Model

```bash
POST /load_model
```

Load the LLM model with custom configuration.

**Request Body:**
```json
{
  "model_name": "Qwen/Qwen2.5-8B-Instruct",
  "tensor_parallel_size": 1,
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.9
}
```

**Response:**
```json
{
  "status": "success",
  "model_name": "Qwen/Qwen2.5-8B-Instruct",
  "message": "Model loaded successfully"
}
```

### Generate Prompts

```bash
POST /generate_prompts
```

Generate domain-specific training prompts.

**Request Body:**
```json
{
  "domain": "AI and machine learning",
  "num_prompts": 20,
  "language": "sv",
  "difficulty": "intermediate",
  "sentence_length": "medium",
  "include_technical_terms": true,
  "style": "conversational"
}
```

**Parameters:**
- `domain` (required): Target domain (e.g., "AI", "medicine", "finance")
- `num_prompts` (default: 10): Number of prompts to generate
- `language` (default: "sv"): Language code ("sv" or "en")
- `difficulty` (default: "intermediate"): "beginner", "intermediate", "advanced"
- `sentence_length` (default: "medium"): "short", "medium", "long"
- `include_technical_terms` (default: true): Include domain-specific terminology
- `style` (default: "conversational"): "conversational", "formal", "technical"

**Response:**
```json
{
  "prompts": [
    "Neurala nätverk kan lära sig att känna igen mönster i data genom att justera sina vikter.",
    "Transformermodeller använder self-attention för att förstå relationer mellan ord i sekvenser.",
    "Gradientnedstigning optimerar modellparametrar genom att minimera förlustfunktionen iterativt."
  ],
  "total": 3
}
```

### Generate Custom Prompt

```bash
POST /generate_custom_prompt
```

Generate a single prompt with a custom instruction.

**Request Body:**
```json
{
  "instruction": "Create a Swedish sentence about neural network training with technical details",
  "max_tokens": 100,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "prompt": "Träning av neurala nätverk kräver noggrann kalibrering av hyperparametrar som learning rate och batch size."
}
```

### Health Check

```bash
GET /health
```

Check service status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-8B-Instruct"
}
```

## Usage Examples

### Basic Generation

```python
import requests

# Generate AI/ML prompts in Swedish
response = requests.post(
    "http://localhost:8001/generate_prompts",
    json={
        "domain": "AI and machine learning",
        "num_prompts": 20,
        "language": "sv"
    }
)
prompts = response.json()["prompts"]
```

### Advanced Configuration

```python
# Generate medical prompts with formal style
response = requests.post(
    "http://localhost:8001/generate_prompts",
    json={
        "domain": "medicine",
        "num_prompts": 30,
        "language": "en",
        "difficulty": "advanced",
        "sentence_length": "long",
        "include_technical_terms": True,
        "style": "formal"
    }
)
```

### Via Backend API

```bash
curl -X POST http://localhost:5001/api/llm/generate-prompts/ \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "finance",
    "num_prompts": 15,
    "language": "sv",
    "difficulty": "intermediate"
  }'
```

## Model Configuration

### Default Settings

- **Model**: Qwen/Qwen2-7B-Instruct
- **Tensor Parallel Size**: 1 (single GPU)
- **Max Model Length**: 4096 tokens
- **GPU Memory Utilization**: 90%

### Custom Configuration

To use a different model or settings, call `/load_model`:

```bash
curl -X POST http://localhost:8001/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen/Qwen2-7B-Instruct",
    "tensor_parallel_size": 2,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85
  }'
```

## Development

### Local Testing

```bash
# Install dependencies
cd llm/app
pip install -r ../requirements.txt

# Run service
uvicorn llm_api:app --host 0.0.0.0 --port 8001
```

### Docker

```bash
# Build
docker build -t llm-service llm/

# Run
docker run --gpus all -p 8001:8001 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  llm-service
```

## Performance

- **First Load**: ~30-60 seconds (model download + initialization)
- **Generation Speed**: ~10-20 prompts/second (batch mode)
- **Memory**: ~16GB VRAM for 8B model
- **Context Length**: Up to 4096 tokens per prompt

## Integration Workflow

1. **Load Model**: Service starts, loads Qwen2.5-8B
2. **Generate Prompts**: User specifies domain (e.g., "AI")
3. **Save to Domain File**: Backend saves to `assets/domains/domain_ai_generated.json`
4. **User Records**: Frontend displays prompts, user records audio
5. **Train Models**: Recorded audio used to fine-tune STT/TTS

## Troubleshooting

### Model Not Loading

- Check GPU availability: `nvidia-smi`
- Verify CUDA version: 12.8.0 required
- Ensure sufficient VRAM: 16GB+ recommended

### Slow Generation

- Increase `gpu_memory_utilization` in `/load_model`
- Use smaller `max_model_len` if prompts are short
- Consider batching: generate 50+ prompts at once

### Out of Memory

- Reduce `gpu_memory_utilization` to 0.7-0.8
- Lower `max_model_len` to 2048
- Restart service to clear cache

## Future Enhancements

- [ ] Multi-GPU support (tensor parallelism)
- [ ] Streaming generation for real-time UI updates
- [ ] Custom fine-tuned domain models
- [ ] Prompt quality scoring and filtering
- [ ] Multi-turn conversation generation
- [ ] Automatic difficulty adjustment based on user level
