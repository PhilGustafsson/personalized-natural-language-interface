import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { distinctUntilChanged, interval, Observable, switchMap, takeWhile } from 'rxjs';
import { environment } from '../../environments/environment';

export interface FineTuningRequest {
    profileID: string;
}

export interface FineTuningResponse {
    jobId: string;
    progress: number;
    estimatedTimeRemaining: number;
    error?: string;
    result?: {
        finetuned_dir: string;
        final_wer: number;
    };
}

@Injectable({
    providedIn: 'root'
})
export class FineTuningService {
    private apiUrl = `${environment.apiUrl}/finetuning`;

    constructor(private http: HttpClient) { }

    startFineTuningTTS(request: FineTuningRequest): Observable<FineTuningResponse> {
        return this.http.post<FineTuningResponse>(`${this.apiUrl}/start-tts/`, request);
    }

    startFineTuningSTT(request: FineTuningRequest): Observable<FineTuningResponse> {
        return this.http.post<FineTuningResponse>(`${this.apiUrl}/start-stt/`, request);
    }

    /**
     * Emits once.
     */
    getFineTuningStatus(jobId: string): Observable<FineTuningResponse> {
        return this.http.get<FineTuningResponse>(`${this.apiUrl}/status/${jobId}`);
    }

    /**
     * Emits every 2 seconds.
     */
    pollFineTuningStatus(jobId: string): Observable<FineTuningResponse> {
        return interval(2000).pipe( // poll every 2s (adjust as needed)
            switchMap(() => this.http.get<FineTuningResponse>(`${this.apiUrl}/status/${jobId}`)),
            takeWhile(res => res.progress < 100, true),
            distinctUntilChanged()
        );
    }

    /**
     * Cancel a fine-tuning job.
     */
    cancelFineTuning(jobId: string) {
        return this.http.post<{ message: string }>(`${this.apiUrl}/cancel/${jobId}`, {});
    }
}