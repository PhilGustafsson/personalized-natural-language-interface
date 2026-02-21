import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { ListboxModule } from 'primeng/listbox';
import { ProfileService } from '../../../services/profile.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-delete-profile',
  standalone: true,
  imports: [CommonModule, FormsModule, ButtonModule, InputTextModule, ListboxModule],
  templateUrl: './delete-profile.component.html'
})
export class DeleteProfileComponent {
  profiles: string[] = [];
  target = '';
  message = '';
  loading = false;

  constructor(private profileService: ProfileService, private router: Router) {}

  ngOnInit() {
    this.load();
  }

  load() {
    this.profileService.listStoredProfiles().subscribe({
      next: (list: string[]) => {
        this.profiles = list;
      },
      error: (err: any) => {
        this.message = 'Failed to load profiles';
        console.error(err);
      }
    });
  }

  select(name: string) {
    this.target = name;
  }

  delete() {
    if (!this.target) {
      this.message = 'Please select or type a profile to delete';
      return;
    }
    this.loading = true;
    this.profileService.deleteStoredProfile(this.target).subscribe({
      next: (res: any) => {
        this.message = res.message;
        this.loading = false;
        this.target = '';
        this.load();
      },
      error: (err: any) => {
        this.message = 'Delete failed';
        console.error(err);
        this.loading = false;
      }
    });
  }

  cancel() {
    // Navigate back to main menu
    this.router.navigate(['/']);
  }
}
