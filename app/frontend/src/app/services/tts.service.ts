import { Injectable, model, signal } from '@angular/core';
import { ProfileService } from './profile.service';
import { HttpClient } from '@angular/common/http';
import { map, Observable, of } from 'rxjs';
import { environment } from '../../environments/environment';

/**
 * The settings for this service.
 */
export interface Settings {
    model: 'default' | 'fineTuned';

    // Fine-tuned models need a profile ID.
    profileID?: string; // Use this if using useProfileService = false.
    useProfileService?: boolean; // Whether to retrieve the profile ID from service.
}

export interface TTSResults {
    text: string,
    audioBlob: Blob,
    audioURL: string,
}

@Injectable({
    providedIn: 'root'
})
export class TtsService {
    private settings = signal<Settings>({ model: 'default' });
    private apiUrl = `${environment.apiUrl}/inference/tts`;

    constructor(
        private profileService: ProfileService,
        private http: HttpClient
    ) { }

    setSettings(settings: Settings) {
        this.settings.set(settings);
    }

    /**
     * Convert to audio.
     */
    toAudio(text: string): Observable<TTSResults> {
        console.log("TTS: Converting text to audio: ", text);
        const settings = this.settings();
        if (settings.model == 'default') {
            return this.toAudioDefault(text);
        }
        else {
            const profileID = settings.useProfileService ? this.profileService.profileID() : settings.profileID;
            if (profileID == null) {
                console.error("TTS: No profile ID set for fine-tuned model.");
                throw new Error("No profile ID set for fine-tuned model.");
            }
            return this.toAudioFineTuned(text, profileID);
        }
    }

    /**
     * Convert to audio use the default base model.
     */
    toAudioDefault(text: string): Observable<TTSResults> {
        return this.toAudioCall(text, `${this.apiUrl}/default/`);
    }

    /**
     * Convert to audio.
     */
    toAudioFineTuned(text: string, profileID: string): Observable<TTSResults> {
        return this.toAudioCall(text, `${this.apiUrl}/fine-tuned/${profileID}`);
    }

    /**
     * Call the backend.
     */
    private toAudioCall(text: string, apiUrl: string): Observable<TTSResults> {
        return this.http.post(apiUrl, { text }, { responseType: 'arraybuffer' })
            .pipe(
                map((arrayBuffer: ArrayBuffer) => {
                    const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' });
                    return {
                        text: text,
                        audioBlob: blob,
                        audioURL: URL.createObjectURL(blob)
                    };
                })
            );
    }

    
}