import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { ListboxModule } from 'primeng/listbox';
import { ProfileService } from '../../../services/profile.service';

@Component({
  selector: 'app-choose-profile',
  standalone: true,
  imports: [CommonModule, FormsModule, ButtonModule, ListboxModule],
  templateUrl: './choose-profile.component.html'
})
export class ChooseProfileComponent implements OnInit {
  profiles: string[] = [];
  loading = false;
  selectedProfile: string | null = null;
  message = '';

  constructor(
    private profileService: ProfileService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadProfiles();
  }

  loadProfiles(): void {
    this.loading = true;
    this.profileService.listStoredProfiles().subscribe({
      next: (list: string[]) => {
        this.profiles = list;
        this.loading = false;
      },
      error: (err: any) => {
        this.message = 'Failed to load profiles';
        this.loading = false;
        console.error(err);
      }
    });
  }

  selectProfile(profileName: string): void {
    this.selectedProfile = profileName;
  }

  confirm(): void {
    if (!this.selectedProfile) {
      this.message = 'Please select a profile';
      return;
    }
    // Set the selected profile in the service (store the folder name as ID)
    this.profileService.profileID.set(this.selectedProfile);
    this.profileService.profileUsername.set(this.selectedProfile);
    // Navigate back to main menu
    this.router.navigate(['/']);
  }

  cancel(): void {
    this.router.navigate(['/']);
  }
}
