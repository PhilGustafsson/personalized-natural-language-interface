import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { MainMenuOnboardingComponent as TranscribeComponent } from '../transcribe/transcribe-component';
import { TtsMenuComponent } from '../tts/tts-menu.component';
import { ProfileService } from '../../../services/profile.service';

@Component({
    selector: 'app-post-profile-menu',
    standalone: true,
    templateUrl: './post-profile-menu.component.html',
    imports: [CommonModule, ButtonModule, TranscribeComponent, TtsMenuComponent],
})
export class PostProfileMenuComponent {
    @Input() profileId?: string;
    @Output() back = new EventEmitter<void>();

    mode: 'menu' | 'stt' | 'tts' = 'menu';
    username: string | null = null;

    constructor(private router: Router, private profileService: ProfileService) {}

    ngOnInit() {
        this.username = this.profileService.profileUsername();
    }

    startStt() {
        this.mode = 'stt';
    }

    startTts() {
        this.mode = 'tts';
    }

    goBack() {
        this.mode = 'menu';
        this.back.emit();
    }

    cancelToMain() {
        this.router.navigate(['/']);
    }
}
