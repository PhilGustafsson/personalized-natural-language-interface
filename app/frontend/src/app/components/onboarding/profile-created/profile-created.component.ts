import { Component, EventEmitter, Input, Output, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ProfileService } from '../../../services/profile.service';

@Component({
    selector: 'app-profile-created',
    standalone: true,
    imports: [CommonModule, ButtonModule],
    templateUrl: './profile-created.component.html'
})
export class ProfileCreatedComponent {
    @Input({required: true}) profileID!: string;
    // Accept the username (optional). When provided we'll show it instead of the raw ID.
    @Input() username?: string | null;

    @Output() ok = new EventEmitter<void>();
    @Output() cancel = new EventEmitter<void>();
    
    constructor(private profile: ProfileService) {}

    ngOnInit(): void {
        // If the parent did not pass the username or profileID, try to read from the service signals.
        if (!this.username || (typeof this.username === 'string' && this.username.trim().length === 0)) {
            const u = this.profile.profileUsername();
            if (u) this.username = u;
        }

        if (!this.profileID) {
            const id = this.profile.profileID();
            if (id) this.profileID = id;
        }
    }

    continue() {
        // proceed to next step in onboarding
        this.ok.emit();
    }
}