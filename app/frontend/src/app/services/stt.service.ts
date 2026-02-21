import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import { BehaviorSubject, Observable } from 'rxjs';
import { HttpClient } from '@angular/common/http';

export interface STTResult {
    text: string;
}

export interface JobInfo {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    created_at: string;
    updated_at: string;
    message: string;
    result?: any;
}

export interface ModelSelect {
    model_dir: string;
    whisper_language: string;
}

export interface LoadDatasetRequest {
    manifest_path?: string;
    recordings_root?: string;
    user?: string;
    seed?: number;
    split_ratios?: { [key: string]: number };
    use_data_augmentation?: boolean;
}

export interface FineTuneRequest {
    user?: string;
    saved_model_dir?: string;
    seed?: number;
    warmup_ratio?: number;
    learning_rate?: number;
    num_train_epochs?: number;
    weight_decay?: number;
    per_device_train_batch_size?: number;
    per_device_eval_batch_size?: number;
    max_grad_norm?: number;
    label_smoothing_factor?: number;
    lr_scheduler_type?: string;
    gradient_accumulation_steps?: number;
    optimizer?: string;
}

export interface OptunaRequest {
    user?: string;
    seed?: number;
    n_trials?: number;
    learning_rate_range?: number[];
    num_train_epochs_range?: number[];
    weight_decay_range?: number[];
    warmup_ratio_range?: number[];
    max_grad_norm_range?: number[];
    per_device_train_batch_size_range?: number[];
    per_device_eval_batch_size_range?: number[];
    gradient_accumulation_steps_range?: number[];
    label_smoothing_factor_range?: number[];
    lr_scheduler_type_choices?: string[];
    optimizer?: string[];
    pruning_warmup_trials?: number;
    pruning_warmup_epochs?: number;
    max_wer_threshold?: number;
    pruning_wer_patience?: number;
}

export interface EvaluateModelRequest {
    eval_split?: string;
    per_device_eval_batch_size?: number;
}

export interface TranscribeResponse {
    text: string;
    language?: string;
    timestamps?: any[];
}

export interface WhisperXTranscribeRequest {
    audio_path: string;
    do_align?: boolean;
}

export interface WhisperXLoadRequest {
    model_name?: string;
    align_model_name?: string;
    cache_dir?: string;
    base_model_path?: string;
    language?: string;
    compute_type?: string;
    batch_size?: number;
    chunk_size?: number;
}

@Injectable({
    providedIn: 'root'
})
export class SttService {
    private apiUrl = `${environment.apiUrl}/inference/stt`;
    // @ts-ignore
    private sttApiUrl = environment.sttApiUrl;

    constructor(
        private http: HttpClient
    ) {
    }

    toText(audioFile: Blob, useFinetuned = false, profileID: string|null = null): Observable<STTResult> {
        console.log("STT: Converting audio to text: ", audioFile);
        const formData = new FormData();
        formData.append('audio', audioFile);
        formData.append('use_finetuned', ""+useFinetuned);
        if (profileID != null) formData.append('profile_id', profileID);

        return this.http.post<STTResult>(`${this.apiUrl}/`, formData);
    }

    uploadAudio(audioFile: Blob): Observable<{path: string}> {
        const formData = new FormData();
        formData.append('audio', audioFile);
        // The backend endpoint is /inference/upload-audio based on blueprint prefix?
        // inference_bp is registered in app.py. Let's check app.py to be sure about the prefix.
        // Assuming it is /inference based on previous apiUrl usage.
        // The previous apiUrl was `${environment.apiUrl}/inference/stt`.
        // So the base for inference blueprint is likely `${environment.apiUrl}/inference`.
        return this.http.post<{path: string}>(`${environment.apiUrl}/inference/upload-audio`, formData);
    }

    // STT Service API methods

    getJobStatus(jobId: string): Observable<JobInfo> {
        return this.http.get<JobInfo>(`${this.sttApiUrl}/jobs/${jobId}`);
    }

    getFinetunedModels(): Observable<any> {
        return this.http.get<any>(`${this.sttApiUrl}/finetuned_models`);
    }

    selectModel(request: ModelSelect): Observable<void> {
        return this.http.post<void>(`${this.sttApiUrl}/select_model`, request);
    }

    loadDataset(request: LoadDatasetRequest): Observable<any> {
        return this.http.post<any>(`${this.sttApiUrl}/load_dataset`, request);
    }

    fineTune(request: FineTuneRequest): Observable<JobInfo> {
        return this.http.post<JobInfo>(`${this.sttApiUrl}/fine_tune`, request);
    }

    optunaSearch(request: OptunaRequest): Observable<JobInfo> {
        return this.http.post<JobInfo>(`${this.sttApiUrl}/optuna_search`, request);
    }

    evaluateModel(request: EvaluateModelRequest): Observable<JobInfo> {
        return this.http.post<JobInfo>(`${this.sttApiUrl}/evaluate_model`, request);
    }

    transcribeWhisperX(request: WhisperXTranscribeRequest): Observable<TranscribeResponse> {
        return this.http.post<TranscribeResponse>(`${this.sttApiUrl}/transcribe_whisperx`, request);
    }

    loadWhisperXModel(request: WhisperXLoadRequest): Observable<void> {
        return this.http.post<void>(`${this.sttApiUrl}/load_whisperx_model`, request);
    }

    selectModelBackend(request: ModelSelect): Observable<void> {
        return this.http.post<void>(`${environment.apiUrl}/inference/select_model`, request);
    }

    transcribeBackend(audioPath: string, language: string = 'sv'): Observable<TranscribeResponse> {
        return this.http.post<TranscribeResponse>(`${environment.apiUrl}/inference/transcribe`, {
            audio_path: audioPath,
            transcribe_language: language
        });
    }

    evaluateTranscription(profileId: string, filename: string, transcription: string, folder: string = 'audio_prompts'): Observable<any> {
        return this.http.post<any>(`${environment.apiUrl}/inference/evaluate_transcription`, {
            profile_id: profileId,
            filename: filename,
            transcription: transcription,
            folder: folder
        });
    }

    getDatasets(profileId: string): Observable<any> {
        return this.http.get<any>(`${environment.apiUrl}/inference/datasets/${profileId}`);
    }
}