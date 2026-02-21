import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { SelectModule } from 'primeng/select';
import { ProfileService } from '../../services/profile.service';
import { TtsMenuComponent } from '../onboarding/tts/tts-menu.component';
import { WriteOwnPromptComponent } from '../onboarding/write-own-prompt/write-own-prompt.component';
import { SttManagerComponent } from '../shared/stt-manager/stt-manager.component';

@Component({
    selector: 'app-main-menu',
    standalone: true,
    templateUrl: './main-menu.component.html',
    imports: [
        CommonModule, 
        FormsModule, 
        ButtonModule, 
        CardModule, 
        SelectModule,
        TtsMenuComponent,
        WriteOwnPromptComponent,
        SttManagerComponent
    ]
})
export class MainMenuComponent implements OnInit {
    profiles: any[] = [];
    selectedProfile: any = null;
    
    activeView: 'dashboard' | 'stt' | 'tts' | 'train' = 'dashboard';

    constructor(private router: Router, private profileService: ProfileService) {}

    ngOnInit() {
        this.loadProfiles();
    }

    loadProfiles() {
        this.profileService.listStoredProfiles().subscribe(profiles => {
            this.profiles = [
                { label: 'Base Model (No Profile)', value: null },
                ...profiles.map(p => ({ label: p, value: p }))
            ];
            
            const currentId = this.profileService.profileID();
            if (currentId) {
                this.selectedProfile = this.profiles.find(p => p.value === currentId) || this.profiles[0];
            } else {
                this.selectedProfile = this.profiles[0];
            }
        });
    }

    onProfileChange() {
        if (this.selectedProfile && this.selectedProfile.value) {
             this.profileService.profileID.set(this.selectedProfile.value);
             this.profileService.profileUsername.set(this.selectedProfile.value); // Assuming username is same as ID for now based on listStoredProfiles
        } else {
             this.profileService.profileID.set(null);
             this.profileService.profileUsername.set(null);
        }
    }

    startStt() {
        this.activeView = 'stt';
    }

    startTts() {
        this.activeView = 'tts';
    }
    
    startTrain() {
        this.activeView = 'train';
    }

    openSttManager() {
        this.router.navigate(['/stt-manager']);
    }

    backToDashboard() {
        this.activeView = 'dashboard';
    }
    
    createNewProfile() {
        this.router.navigate(['/new-profile']);
    }

    deleteProfile() {
        this.router.navigate(['/delete-profile']);
    }
}