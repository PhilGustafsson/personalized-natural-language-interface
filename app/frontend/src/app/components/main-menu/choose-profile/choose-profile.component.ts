import { Component, EventEmitter, Output, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ProfileService } from '../../../services/profile.service';
import { ButtonModule } from 'primeng/button';
import { ProgressSpinnerModule } from 'primeng/progressspinner';

@Component({
  selector: 'app-choose-profile',
  standalone: true,
  imports: [CommonModule, ButtonModule, ProgressSpinnerModule],
  templateUrl: './choose-profile.component.html'
})
export class ChooseProfileComponent implements OnInit {
  @Output() profileSelected = new EventEmitter<string>();
  @Output() back = new EventEmitter<void>();

  profiles: string[] = [];
  loading = true;
  error = '';

  constructor(private profileService: ProfileService) {}

  ngOnInit(): void {
    this.loadProfiles();
  }

  loadProfiles(): void {
    this.loading = true;
    this.error = '';
    this.profileService.listStoredProfiles().subscribe({
      next: (profiles) => {
        this.profiles = profiles;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error loading profiles:', err);
        this.error = 'Failed to load profiles. Please try again.';
        this.loading = false;
      }
    });
  }

  selectProfile(profileName: string): void {
    this.profileService.profileID.set(profileName);
    this.profileSelected.emit(profileName);
  }

  goBack(): void {
    this.back.emit();
  }
}
