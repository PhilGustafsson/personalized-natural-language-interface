import { Component, OnInit, OnDestroy, Input } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ProfileService } from '../../../services/profile.service';

@Component({
  selector: 'app-audio-recorder',
  standalone: true,
  templateUrl: './audio-recorder.component.html',
  imports: [CommonModule, DecimalPipe, ButtonModule]
})
export class AudioRecorderComponent implements OnInit, OnDestroy {
  private mediaRecorder!: MediaRecorder;
  private ws!: WebSocket;
  protected isRecording = false;
  protected isProcessing = false;
  protected elapsedTime = 0;
  private startTime!: number;
  private timerInterval: any;
  private recordedChunks: Blob[] = []; // Store chunks here
  private processingInterval: any; // For periodic processing
  protected lastAnalysis: any = null;
  private profileUsername: string | null = null;
  
  @Input() promptName: string = ''; // Optional prompt identifier (e.g., "prompt_1")

  constructor(private profileService: ProfileService) {}

  ngOnInit(): void {
    // Get the current profile USERNAME from the service (this is the folder name)
    this.profileUsername = this.profileService.profileUsername();
    console.log('AudioRecorderComponent initialized with profileUsername:', this.profileUsername);
  }

  ngOnDestroy(): void {
    this.stopRecording();
    if (this.ws) {
      this.ws.close();
    }
  }
  
  getQualityColor(score: number): string {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  }
  
  getQualityText(score: number): string {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  }

  startRecording() {
    if (!this.profileUsername) {
      console.error('No profile username available. Cannot start recording.');
      return;
    }

    this.ws = new WebSocket('ws://localhost:5001/audio');
    this.isRecording = true;
    this.isProcessing = false;
    this.lastAnalysis = null;
    this.recordedChunks = []; // Clear previous chunks

    this.ws.onopen = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true
          } 
        });
        
        this.mediaRecorder = new MediaRecorder(stream, { 
          mimeType: 'audio/webm; codecs=opus',
          audioBitsPerSecond: 128000
        });

        // Collect chunks instead of sending immediately
        this.mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            this.recordedChunks.push(event.data);
          }
        };

        // Send complete recordings every 3 seconds
        this.mediaRecorder.onstop = () => {
          this.sendCompleteRecording();
        };

        // Start recording and set up periodic processing
        this.mediaRecorder.start();
        this.startTime = Date.now();
        
        // Update timer every second
        this.timerInterval = setInterval(() => {
          this.elapsedTime = Math.floor((Date.now() - this.startTime) / 1000);
        }, 1000);

        // Process audio every 3 seconds
        this.processingInterval = setInterval(() => {
          if (this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            // Restart recording for continuous capture
            setTimeout(() => {
              if (this.isRecording) {
                this.recordedChunks = []; // Clear for next batch
                this.mediaRecorder.start();
              }
            }, 100);
          }
        }, 3000);
      } catch (err) {
        console.error("Error accessing microphone:", err);
        this.isRecording = false;
        alert("Could not access microphone. Please ensure you have granted permission.");
      }
    };

    this.ws.onmessage = (event) => {
      console.log("Raw WS message:", event.data);
      this.isProcessing = false;
      try {
        this.lastAnalysis = JSON.parse(event.data);
        console.log("Parsed analysis:", this.lastAnalysis);
      } catch (err) {
        console.warn("Invalid JSON from backend:", event.data, err);
      }
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.isRecording = false;
    };
    
    this.ws.onclose = () => {
        console.log("WebSocket closed");
    };
  }

  private async sendCompleteRecording() {
    if (this.recordedChunks.length === 0 || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.isProcessing = true;

    try {
      // Create complete WebM blob from all chunks
      const completeBlob = new Blob(this.recordedChunks, { 
        type: 'audio/webm; codecs=opus' 
      });
      
      console.log(`Sending complete recording: ${completeBlob.size} bytes with profileUsername: ${this.profileUsername}`);
      
      // Convert to ArrayBuffer and send as binary with profile_username and prompt_name
      // Format: [username_length_4bytes][username][prompt_length_4bytes][prompt_name][audio_data]
      const usernameBytes = new TextEncoder().encode(this.profileUsername || '');
      const promptBytes = new TextEncoder().encode(this.promptName || '');
      const arrayBuffer = await completeBlob.arrayBuffer();
      
      // Create DataView to write big-endian integers
      const headerSize = 4 + usernameBytes.length + 4 + promptBytes.length;
      const wrappedBuffer = new ArrayBuffer(headerSize + arrayBuffer.byteLength);
      const view = new DataView(wrappedBuffer);
      
      let offset = 0;
      
      // Write username length (4 bytes, big-endian)
      view.setUint32(offset, usernameBytes.length, false); // false = big-endian
      offset += 4;
      
      // Write username bytes
      new Uint8Array(wrappedBuffer).set(usernameBytes, offset);
      offset += usernameBytes.length;
      
      // Write prompt length (4 bytes, big-endian)
      view.setUint32(offset, promptBytes.length, false); // false = big-endian
      offset += 4;
      
      // Write prompt bytes
      new Uint8Array(wrappedBuffer).set(promptBytes, offset);
      offset += promptBytes.length;
      
      // Write audio data
      new Uint8Array(wrappedBuffer).set(new Uint8Array(arrayBuffer), offset);
      
      this.ws.send(wrappedBuffer);
      
    } catch (error) {
      console.error("Error sending audio data:", error);
      this.isProcessing = false;
    }
  }

  stopRecording() {
    clearInterval(this.timerInterval);
    clearInterval(this.processingInterval);
    this.isRecording = false;
    
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    
    if (this.mediaRecorder?.stream) {
      this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    
    // Do NOT close WS immediately to allow last message to be sent/received.
    // It will be closed on destroy or next start.
    setTimeout(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close();
        }
    }, 2000);
  }
}