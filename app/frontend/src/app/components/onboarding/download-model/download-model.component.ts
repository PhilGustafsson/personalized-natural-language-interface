import { Component, computed, effect, EventEmitter, Output, Signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ProgressBarModule } from 'primeng/progressbar';
import { ModelStatusResponse, ProfileService } from '../../../services/profile.service';
import { toSignal } from '@angular/core/rxjs-interop';

export enum DownloadState {
    Ready,
    Downloading,
    Complete
}

@Component({
    selector: 'app-download-model',
    templateUrl: './download-model.component.html',
    imports: [CommonModule, ButtonModule, ProgressBarModule]
})
export class DownloadModelComponent {
    @Output() ok = new EventEmitter<void>;
    @Output() cancel = new EventEmitter<void>;

    DownloadState = DownloadState;
    readonly progress: Signal<number>;
    readonly downloadState: Signal<DownloadState>;

    private readonly modelStatus: Signal<ModelStatusResponse|undefined>;
    readonly isTtsDownloaded = computed(() => this.modelStatus()?.tts);
    readonly isSttDownloaded = computed(() => this.modelStatus()?.stt);

    constructor(protected profileService: ProfileService) {
        this.modelStatus = toSignal(this.profileService.checkModelsStatus());
        this.progress = computed(() => profileService.modelDownloadProgress());
        this.downloadState = computed(() => {
            const p = this.progress();
            if (p == -1) {
                return DownloadState.Ready;
            }
            else if(p == 100) {
                return DownloadState.Complete;
            }
            else {
                return DownloadState.Downloading;
            }
        });
    }

    startDownload() {
        if (this.downloadState() == DownloadState.Ready) {
            this.profileService.downloadModels();
        }
        else {
            console.warn('Download already in progress or completed.');
        }
    }
}