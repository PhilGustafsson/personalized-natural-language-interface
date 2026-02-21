"""
LLM Service API for generating domain-specific training prompts.
Uses vLLM with Qwen2.5-8B for efficient text generation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import os

from llm_service import LLMService

app = FastAPI(title="LLM Prompt Generation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM service instance
llm_service: Optional[LLMService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize service without loading model - lazy loading on first use"""
    global llm_service
    import logging
    logger = logging.getLogger("uvicorn")
    logger.info("LLM service started - model will be loaded on first use")
    llm_service = LLMService()  # Initialize but don't load model yet


class Language(str, Enum):
    SWEDISH = "sv"
    ENGLISH = "en"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class LoadModelRequest(BaseModel):
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = 4096
    gpu_memory_utilization: float = 0.65


class GeneratePromptsRequest(BaseModel):
    domain: str = Field(..., description="Domain/topic for prompt generation (e.g., 'AI', 'medicine', 'finance')")
    num_prompts: int = Field(default=20, ge=1, le=100, description="Number of prompts to generate")
    language: Language = Field(default=Language.SWEDISH, description="Language for generated prompts")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.INTERMEDIATE, description="Difficulty level")
    sentence_length: str = Field(default="medium", description="Desired sentence length: short, medium, or long")
    include_technical_terms: bool = Field(default=True, description="Whether to include domain-specific technical terms")
    style: str = Field(default="conversational", description="Style: conversational, formal, or educational")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=150, ge=10, le=500, description="Maximum tokens per prompt")


class GeneratedPrompt(BaseModel):
    id: int
    text: str
    domain: str
    language: str
    difficulty: str


class GeneratePromptsResponse(BaseModel):
    prompts: List[GeneratedPrompt]
    domain: str
    language: str
    total_generated: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the LLM service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy" if (llm_service and llm_service.llm) else "no_model_loaded",
        model_loaded=(llm_service is not None and llm_service.llm is not None),
        model_name=llm_service.model_name if llm_service else None
    )


@app.post("/load_model", status_code=200)
async def load_model(request: LoadModelRequest):
    """Load vLLM model. This may take a few minutes on first run."""
    global llm_service
    
    try:
        if llm_service and llm_service.llm:
            return {"message": f"Model already loaded: {llm_service.model_name}"}
        
        if not llm_service:
            llm_service = LLMService(
                model_name=request.model_name,
                tensor_parallel_size=request.tensor_parallel_size,
                max_model_len=request.max_model_len,
                gpu_memory_utilization=request.gpu_memory_utilization
            )
        
        llm_service.load_model(
            model_name=request.model_name,
            tensor_parallel_size=request.tensor_parallel_size,
            max_model_len=request.max_model_len,
            gpu_memory_utilization=request.gpu_memory_utilization
        )
        
        return {
            "message": "Model loaded successfully",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/unload_model", status_code=200)
async def unload_model():
    """Unload the model from memory to free VRAM"""
    global llm_service
    if llm_service is None:
        raise HTTPException(
            status_code=400,
            detail="LLM service not initialized"
        )
    
    try:
        llm_service.unload_model()
        return {
            "status": "success",
            "message": "Model unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_prompts", response_model=GeneratePromptsResponse)
async def generate_prompts(request: GeneratePromptsRequest):
    """
    Generate domain-specific training prompts for STT/TTS training.
    
    The prompts are designed to:
    - Cover domain-specific vocabulary and terminology
    - Vary in complexity and sentence structure
    - Be natural and readable for voice recording
    - Include code-mixing (Swedish with English technical terms) when appropriate
    """
    if llm_service is None:
        raise HTTPException(
            status_code=503,
            detail="LLM service not initialized"
        )
    
    # Lazy load model on first use
    if llm_service.llm is None:
        try:
            llm_service.load_model()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    try:
        prompts_data = llm_service.generate_domain_prompts(
            domain=request.domain,
            num_prompts=request.num_prompts,
            language=request.language.value,
            difficulty=request.difficulty.value,
            sentence_length=request.sentence_length,
            include_technical_terms=request.include_technical_terms,
            style=request.style,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Ensure we have a list
        if not isinstance(prompts_data, list):
            print(f"ERROR: prompts_data is not a list, got type: {type(prompts_data)}")
            print(f"prompts_data content: {prompts_data}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid response format from LLM service: expected list, got {type(prompts_data)}"
            )
        
        generated_prompts = [
            GeneratedPrompt(
                id=i + 1,
                text=prompt,
                domain=request.domain,
                language=request.language.value,
                difficulty=request.difficulty.value
            )
            for i, prompt in enumerate(prompts_data)
        ]
        
        return GeneratePromptsResponse(
            prompts=generated_prompts,
            domain=request.domain,
            language=request.language.value,
            total_generated=len(generated_prompts)
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR in generate_prompts: {str(e)}")
        print(f"Traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate_custom_prompt")
async def generate_custom_prompt(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 200
) -> Dict[str, Any]:
    """Generate text with custom system and user prompts."""
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call /load_model first."
        )
    
    try:
        result = llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "generated_text": result,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
