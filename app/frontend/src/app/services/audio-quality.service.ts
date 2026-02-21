
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { map, Observable, Subject, timer } from 'rxjs';
import { environment } from '../../environments/environment';


export interface Quality {
    score: number; // 0-100,
    passed: boolean; // true if score >= threshold
}

/**
 * A service for analyzing quality.
 */
@Injectable({
    providedIn: 'root'
})
export class AudioQualityService {

    constructor(private http: HttpClient) { }

    checkQuality(audioBlob: Blob): Observable<Quality> {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.wav');
        return this.http.post<Quality>(`${environment.apiUrl}/audio/quality-check-clip/`, formData);
    }
}