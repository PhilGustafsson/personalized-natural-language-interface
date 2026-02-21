import { Component, OnInit } from '@angular/core';
import { DataService } from '../../services/data.service';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { AudioRecorderComponent } from "../onboarding/audio-recorder/audio-recorder.component";  // if you use HTTP

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  standalone: true,
  imports: [CommonModule, HttpClientModule, AudioRecorderComponent],
})
export class Home implements OnInit {
  message: string = '';

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService.getData().subscribe((res: any) => {
      this.message = res.message;
    });
  }
}
