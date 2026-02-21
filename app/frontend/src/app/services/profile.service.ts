import { effect, Injectable, signal } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { BehaviorSubject, interval, map, Observable, Subject, take, tap, catchError, throwError } from 'rxjs';

export interface CreateUserResponse {
  profile_id: string;
  username?: string;
  message: string;
}

export interface ModelStatusResponse {
    tts: boolean;
    stt: boolean;
}

export interface BackendResponse {
    message: string;
}

@Injectable({
  providedIn: 'root'
})
export class ProfileService {
    public hasProfile = false;
    public readonly profileID = signal<string | null>(null);
    public readonly profileUsername = signal<string | null>(null);
    public readonly modelDownloadProgress = signal(-1);

    constructor(private http: HttpClient) {
        this.profileID.set(localStorage.getItem('profile-id'));
        this.profileUsername.set(localStorage.getItem('profile-username'));
        console.log("Loaded profile ID from local storage:", this.profileID());
        console.log("Loaded profile username from local storage:", this.profileUsername());

        // Save profile ID and username to local storage when they change.
        effect(() => {
            const id = this.profileID();
            if (id != null) {
                localStorage.setItem('profile-id', id);
            }
            else {
                localStorage.removeItem('profile-id');
            }
        });

        effect(() => {
            const u = this.profileUsername();
            if (u != null) {
                localStorage.setItem('profile-username', u);
            }
            else {
                localStorage.removeItem('profile-username');
            }
        });
    }

    // Accept an optional username and include it in the creation request
    newProfile(username?: string): Observable<CreateUserResponse> {
        this.hasProfile = true;
        const body: any = {
            consent: true,
            language: 'Swedish',
            device_info: `${navigator.userAgent}`
        };
        if (username) {
            body.username = username;
        }
        const response = this.http.post<CreateUserResponse>(`${environment.apiUrl}/profiles/`, body).pipe(
            tap(r => {
                this.profileID.set(r.profile_id);
                // Prefer backend-confirmed username, fall back to provided username.
                this.profileUsername.set(r.username ?? username ?? null);
            }),
            catchError((err: any) => {
                // Check for 409 Conflict (duplicate username)
                if (err.status === 409) {
                    const errorMsg = err.error?.error || 'Username already exists. Please choose a different name.';
                    return throwError(() => new Error(errorMsg));
                }
                // Re-throw other errors as-is
                return throwError(() => err);
            })
        );
        return response;
    }

    deleteProfile(): Observable<BackendResponse> {
        const id = this.profileID();
        console.log("Deleting profile with ID:", id);
        this.profileID.set(null);
        this.profileUsername.set(null);
        localStorage.removeItem('profile-id');
        localStorage.removeItem('profile-username');
        return this.http.delete<BackendResponse>(`${environment.apiUrl}/profiles/<${id}>`);
    }

    /**
     * List profile folder names stored on the backend filesystem.
     */
    listStoredProfiles(): Observable<string[]> {
        return this.http.get<string[]>(`${environment.apiUrl}/profiles/filesystem`);
    }

    /**
     * Delete a profile folder on the backend filesystem by name.
     */
    deleteStoredProfile(name: string): Observable<BackendResponse> {
        return this.http.delete<BackendResponse>(
            `${environment.apiUrl}/profiles/filesystem/${encodeURIComponent(name)}`
        );
    }

    /**
     * Check if the models have been downloaded.
     */
    checkModelsStatus(): Observable<ModelStatusResponse> {
        return this.http.get<ModelStatusResponse>(`${environment.apiUrl}/models/status/`);
    }

    /**
     * Check if the models have been downloaded. Similar to checkModelsStatus, 
     * however, this just emits a boolean: true iff both models are downloaded.
     */
    areModelsDownloaded(): Observable<boolean> {
        return this.checkModelsStatus().pipe(
            map(res => res.stt && res.tts)
        );
    }

    /**
     * Tell backend to download models. This also starts tracking the 
     * download progress which can be read from modelDownloadProgress.
     */
    downloadModels() {
        this.modelDownloadProgress.set(0);
        this.http.post(`${environment.apiUrl}/models/download/`, {}).subscribe(
            () => this.pollDownloadProcess()
        );
    }

    /**
     * Poll the backend at set intervals of the download progress. Stops 
     * polling when download is complete.
     */
    private pollDownloadProcess() {
        const interval = setInterval(() => {
        this.http.get<{status: string; progress: number}>(
            `${environment.apiUrl}/models/download/progress/`
            ).subscribe((res: {status: string; progress: number}) => {
                this.modelDownloadProgress.set(res.progress);
                if (res.status === 'done') {
                    clearInterval(interval);
                }
            });
        }, 200);
    }

    /**
     * Upload a transcribe recording blob to the backend under the current profile.
     * Endpoint accepts multipart/form-data with field "file".
     */
    uploadTranscribeRecording(file: Blob, filename: string) {
        const id = this.profileID();
        if (!id) {
            throw new Error('No profile selected');
        }
        const fd = new FormData();
        fd.append('file', file, filename);
        return this.http.post<BackendResponse>(`${environment.apiUrl}/profiles/filesystem/${encodeURIComponent(id)}/transcribe/`, fd);
    }
}
