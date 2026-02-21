import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { environment } from '../../environments/environment';

export interface GeneratePromptsRequest {
    domain: string;
    num_prompts?: number;
    language?: string;
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    sentence_length?: 'short' | 'medium' | 'long';
    include_technical_terms?: boolean;
    style?: 'conversational' | 'formal' | 'technical';
}

export interface GeneratePromptsResponse {
    prompts: Array<{
        id: number;
        text: string;
        domain: string;
        language: string;
        difficulty: string;
    }>;
    domain: string;
    language: string;
    total_generated: number;
}

export interface LLMHealthResponse {
    status: string;
    model_loaded: boolean;
    model_name?: string;
    message?: string;
}

@Injectable({
    providedIn: 'root'
})
export class LLMService {
    constructor(private http: HttpClient) {}

    /**
     * Generate domain-specific prompts using the LLM service.
     * Returns an array of prompt strings.
     */
    generatePrompts(request: GeneratePromptsRequest): Observable<string[]> {
        const url = `${environment.apiUrl}/llm/generate-prompts/`;
        return this.http.post<GeneratePromptsResponse>(url, request).pipe(
            map(response => response.prompts.map(p => p.text))
        );
    }

    /**
     * Simple version for backward compatibility - just pass domain string.
     */
    generatePromptsSimple(domain: string, numPrompts: number = 20): Observable<string[]> {
        return this.generatePrompts({ domain, num_prompts: numPrompts });
    }

    /**
     * Load the LLM model with custom configuration.
     */
    loadModel(modelName?: string, tensorParallelSize?: number): Observable<any> {
        const url = `${environment.apiUrl}/llm/load-model/`;
        const body: any = {};
        if (modelName) body.model_name = modelName;
        if (tensorParallelSize) body.tensor_parallel_size = tensorParallelSize;
        return this.http.post(url, body);
    }

    /**
     * Check LLM service health and model status.
     */
    checkHealth(): Observable<LLMHealthResponse> {
        const url = `${environment.apiUrl}/llm/health/`;
        return this.http.get<LLMHealthResponse>(url);
    }

    /**
     * Save generated prompts to a domain file for later use.
     */
    saveGeneratedPrompts(domainName: string, prompts: Array<{id: number, text: string}>, language: string = 'sv'): Observable<any> {
        const url = `${environment.apiUrl}/llm/save-generated-prompts/`;
        return this.http.post(url, { domain_name: domainName, prompts, language });
    }
}