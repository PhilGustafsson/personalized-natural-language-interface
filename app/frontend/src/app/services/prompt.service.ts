import { computed, effect, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { BehaviorSubject, map, Observable, tap, timer, throwError } from 'rxjs';
import { ProfileService } from './profile.service';
import { ActivatedRoute, Router } from '@angular/router';

export interface Prompt {
    id: string,
    text: string
}

export interface Prompts {
    name: string, // the name of the collection
    language: string,
    prompts: Prompt[]
}

export interface RecordingData {
    recording: Blob,
    qualityCheckedByUser: boolean,
    passedAutomaticQualityCheck: boolean,
    automaticQualityScore: number
}

export interface SaveRecordingRespone {
    message: string,
    filepath: string
}


@Injectable({
    providedIn: 'root'
})
export class PromptService {

    private prompts = signal<Prompts | null>(null);
    public readonly arePromptsLoaded = computed(() => this.prompts() != null);

    private readonly promptIndex = signal(0);
    private readonly currentPrompt = signal<Prompt | null>(null);
    public readonly currentPromptText = computed(() => this.currentPrompt()?.text ?? "null");
    public readonly promptingComplete = signal(false);
    public readonly currentPromptNumber = computed(() => this.promptIndex() + 1);
    public readonly totalPrompts = computed(() => this.prompts()?.prompts.length ?? 0);
    
    constructor(
        private http: HttpClient, 
        private profileService: ProfileService,
        private router: Router,
        private route: ActivatedRoute,
    ) {
        console.log('PromptService constructor called');
        
        // Load prompt state from query params.
        route.queryParams.subscribe(params => {
            const promptIndex = params['promptIndex'] ? +params['promptIndex'] : -1;
            if (promptIndex >= 0 && promptIndex != this.promptIndex()) {
                this.promptIndex.set(promptIndex);
            }
        });

        // Update query params when prompt index changes.
        effect(() => {
            const promptIndex = this.promptIndex();
            this.router.navigate([], {
                queryParams: {promptIndex},
                queryParamsHandling: 'merge'
            });
        });

        // Auto-refresh current prompt when prompts load or index changes
        effect(() => {
            const prompts = this.prompts();
            const index = this.promptIndex();
            
            console.log('Prompt effect triggered - prompts:', prompts?.name, 'index:', index);
            
            if (prompts && prompts.prompts.length > 0) {
                if (index < prompts.prompts.length) {
                    this.currentPrompt.set(prompts.prompts[index]);
                    this.promptingComplete.set(false);
                    console.log('Set current prompt:', prompts.prompts[index].text);
                } else {
                    this.currentPrompt.set(null);
                    this.promptingComplete.set(true);
                    console.log('Prompting complete - no more prompts');
                }
            } else {
                console.log('No prompts loaded yet');
            }
        });

        // todo: persist the collection name in queury params.
    }

    /**
     * Loads the standard Swedish prompts from the backend.
     */
    loadSvStandardPrompts() {
        console.log('loadSvStandardPrompts called');
        this.http.get<Prompts>(`${environment.apiUrl}/prompts/sv-standard2/`).subscribe({
            next: prompts => {
                console.log('Setting prompts from backend:', prompts.name);
                this.prompts.set(prompts),
                this.refreshPrompt();
                console.log('Loaded prompts:', prompts.name);
                console.log('First prompt:', prompts.prompts[0]);
            }
        });
    }

    /**
     * Loads dummy prompts for testing purposes.
     */
    loadDummyPrompts() {
        const dummyPrompts: Prompts = {
            name: 'dummy',
            language: 'en',
            prompts: [
                { id: '1', text: 'Hello world.' },
                { id: '2', text: 'The quick brown fox jumps over the lazy dog.' }
            ]
        };
        this.prompts.set(dummyPrompts);
        this.refreshPrompt();
        console.log('Loaded dummy prompts:', dummyPrompts.name);
        console.log('First dummy prompt:', dummyPrompts.prompts[0]);
    }

    /**
     * Refresh the current prompt based on the current prompt index.
     */
    refreshPrompt() {
        const promptsData = this.prompts();
        if (!promptsData || !promptsData.prompts || promptsData.prompts.length === 0) {
            this.currentPrompt.set(null);
            return;
        }
        
        const prompts = promptsData.prompts;
        if (prompts.length <= this.promptIndex()) {
            this.promptingComplete.set(true);
            this.currentPrompt.set(null);
        }
        else {
            this.currentPrompt.set(prompts[this.promptIndex()]);
        }
    }

    /**
     * Move to the next prompt. If there are no more prompts, set promptingComplete to true.
     */
    nextPrompt() {
        this.promptIndex.update(i => i + 1);
        this.refreshPrompt();
    }

    /**
     * Save the recording associated with the current prompt.
     * The recording must be of type 'audio/wav'.
     */
    
    saveRecording(recordingData: RecordingData): Observable<SaveRecordingRespone> {
        // Prefer the human-readable username folder when available. If the
        // username is not set, fall back to the DB profile id (uuid).
        const profileID = this.profileService.profileUsername() || this.profileService.profileID()!;
        const prompt = this.currentPrompt()!;
        const promptID = `${this.prompts()?.name}-${prompt.id}`;
        const filename = `${promptID}.wav`;

        const formData = new FormData();
        formData.append('file', recordingData.recording, filename);
        formData.append('profileID', profileID);
        formData.append('promptID', promptID);
        formData.append('promptText', prompt.text);
        formData.append('qualityCheckedByUser', recordingData.qualityCheckedByUser.toString());
        formData.append('passedAutomaticQualityCheck', recordingData.passedAutomaticQualityCheck.toString());
        formData.append('automaticQualityScore', recordingData.automaticQualityScore.toString());

        return this.http.post<SaveRecordingRespone>(`${environment.apiUrl}/prompts/save-recording`, formData);
    }

    /**
     * Sets the prompts from a list of strings (e.g. from LLM).
     */
    setPrompts(promptTexts: string[]) {
        console.log('PromptService.setPrompts called with', promptTexts.length, 'prompts');
        console.log('First prompt text:', promptTexts[0]);
        
        const newPrompts: Prompts = {
            name: 'generated',
            language: 'en', // Defaulting to en, maybe make this configurable
            prompts: promptTexts.map((text, index) => ({
                id: `gen-${index + 1}`,
                text: text
            }))
        };
        this.prompts.set(newPrompts);
        this.refreshPrompt();
        
        console.log('Prompts set. Current prompt:', this.currentPrompt());
        console.log('Total prompts:', this.totalPrompts());
    }

    /**
     * Public accessor for the current prompts value.
     * Some components (and legacy code) read the prompts collection directly.
     */
    getPrompts(): Prompts | null {
        return this.prompts();
    }
}
