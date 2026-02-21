import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { SelectModule } from 'primeng/select';
import { InputTextModule } from 'primeng/inputtext';
import { CardModule } from 'primeng/card';
import { ProgressSpinnerModule } from 'primeng/progressspinner';
import { DividerModule } from 'primeng/divider';
import { CheckboxModule } from 'primeng/checkbox';
import { TooltipModule } from 'primeng/tooltip';
import { SttService, JobInfo, ModelSelect, LoadDatasetRequest, FineTuneRequest, OptunaRequest, EvaluateModelRequest, WhisperXLoadRequest } from '../../../services/stt.service';

@Component({
  selector: 'app-stt-manager',
  standalone: true,
  imports: [
    CommonModule, 
    FormsModule, 
    ButtonModule, 
    SelectModule, 
    InputTextModule, 
    CardModule, 
    ProgressSpinnerModule,
    DividerModule,
    CheckboxModule,
    TooltipModule
  ],
  templateUrl: './stt-manager.component.html',
  styleUrls: ['./stt-manager.component.scss']
})
export class SttManagerComponent implements OnInit {
  
  // Models
  finetunedModels: any[] = [];
  selectedModelDir: string = 'models/kb-whisper-large';
  whisperLanguage: string = 'sv';
  
  // Dataset
  datasetUser: string = 'David';
  datasetManifest: string = 'data_generation/ai_manus.jsonl';
  
  // Finetuning
  finetuneUser: string = 'default_user';
  finetuneEpochs: number = 5;

  // UI State
  showExpertSettings = false;
  useOptuna = false;

  // Optuna Params
  optunaTrials = 5;
  optunaEpochsMin = 9;
  optunaEpochsMax = 12;
  optunaLrMin = 1e-6;
  optunaLrMax = 5e-4;
  
  // Evaluation
  evalSplit: string = 'test';
  
  // Transcription
  transcriptionText: string = '';
  isRecording = false;
  mediaRecorder: MediaRecorder | null = null;
  audioChunks: Blob[] = [];
  
  // Status
  loading = false;
  statusMessage = '';
  currentJobId: string | null = null;
  jobStatus: JobInfo | null = null;

  constructor(private sttService: SttService) {}

  ngOnInit() {
    this.loadFinetunedModels();
  }

  loadFinetunedModels() {
    this.sttService.getFinetunedModels().subscribe({
      next: (models) => {
        // Assuming models is a dict or list. Adjust based on actual response.
        // The API returns a dict where keys are model names.
        this.finetunedModels = Object.keys(models).map(key => ({ label: key, value: models[key] }));
        // Add default base model
        this.finetunedModels.unshift({ label: 'Base Model (kb-whisper-large)', value: 'models/kb-whisper-large' });
      },
      error: (err) => console.error('Failed to load models', err)
    });
  }

  onSelectModel() {
    this.loading = true;
    this.statusMessage = 'Selecting model...';
    const request: ModelSelect = {
      model_dir: this.selectedModelDir,
      whisper_language: this.whisperLanguage
    };
    
    // Also load WhisperX model as requested "always need to load_whisperx_model before transcribe"
    // But select_model might be for finetuning context?
    // The user said: "And when finetuning we always need to select model then load the dataset..."
    // And "The most important thing here is that we always need to load_whisperx_model before transcribe."
    
    this.sttService.selectModel(request).subscribe({
      next: () => {
        this.loading = false;
        this.statusMessage = 'Model selected for finetuning context.';
      },
      error: (err) => {
        this.loading = false;
        this.statusMessage = 'Error selecting model: ' + err.message;
      }
    });
  }

  onLoadWhisperXModel() {
    this.loading = true;
    this.statusMessage = 'Loading WhisperX model...';
    const request: WhisperXLoadRequest = {
        model_name: this.selectedModelDir,
        align_model_name: "models/wav2vec2-large-voxrex-swedish",
        cache_dir: "models/",
        base_model_path: "models/kb-whisper-large",
        compute_type: "float16",
        batch_size: 64,
        chunk_size: 30
    };
    this.sttService.loadWhisperXModel(request).subscribe({
        next: () => {
            this.loading = false;
            this.statusMessage = 'WhisperX model loaded.';
        },
        error: (err) => {
            this.loading = false;
            this.statusMessage = 'Error loading WhisperX model: ' + err.message;
        }
    });
  }

  onLoadDataset() {
    this.loading = true;
    this.statusMessage = 'Loading dataset...';
    const request: LoadDatasetRequest = {
      user: this.datasetUser,
      manifest_path: this.datasetManifest
    };
    this.sttService.loadDataset(request).subscribe({
      next: () => {
        this.loading = false;
        this.statusMessage = 'Dataset loaded.';
      },
      error: (err) => {
        this.loading = false;
        this.statusMessage = 'Error loading dataset: ' + err.message;
      }
    });
  }

  onFineTune() {
    this.loading = true;
    this.statusMessage = this.useOptuna ? 'Starting Optuna search...' : 'Starting finetuning...';
    
    if (this.useOptuna) {
        const request: OptunaRequest = {
            user: this.finetuneUser,
            n_trials: this.optunaTrials,
            num_train_epochs_range: [this.optunaEpochsMin, this.optunaEpochsMax],
            learning_rate_range: [this.optunaLrMin, this.optunaLrMax]
        };
        this.sttService.optunaSearch(request).subscribe({
            next: (job) => {
                this.loading = false;
                this.currentJobId = job.job_id;
                this.statusMessage = 'Optuna search started. Job ID: ' + job.job_id;
                this.pollJobStatus();
            },
            error: (err) => {
                this.loading = false;
                this.statusMessage = 'Error starting Optuna search: ' + err.message;
            }
        });
    } else {
        const request: FineTuneRequest = {
          user: this.finetuneUser,
          num_train_epochs: this.finetuneEpochs
        };
        this.sttService.fineTune(request).subscribe({
          next: (job) => {
            this.loading = false;
            this.currentJobId = job.job_id;
            this.statusMessage = 'Finetuning started. Job ID: ' + job.job_id;
            this.pollJobStatus();
          },
          error: (err) => {
            this.loading = false;
            this.statusMessage = 'Error starting finetuning: ' + err.message;
          }
        });
    }
  }

  onEvaluate() {
    this.loading = true;
    this.statusMessage = 'Starting evaluation...';
    const request: EvaluateModelRequest = {
      eval_split: this.evalSplit
    };
    this.sttService.evaluateModel(request).subscribe({
      next: (job) => {
        this.loading = false;
        this.currentJobId = job.job_id;
        this.statusMessage = 'Evaluation started. Job ID: ' + job.job_id;
        this.pollJobStatus();
      },
      error: (err) => {
        this.loading = false;
        this.statusMessage = 'Error starting evaluation: ' + err.message;
      }
    });
  }

  pollJobStatus() {
    if (!this.currentJobId) return;
    
    const interval = setInterval(() => {
      this.sttService.getJobStatus(this.currentJobId!).subscribe({
        next: (job) => {
          this.jobStatus = job;
          if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(interval);
            this.statusMessage = `Job ${job.status}: ${job.message}`;
          }
        },
        error: () => clearInterval(interval)
      });
    }, 2000);
  }

  // Transcription
  startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];
      this.mediaRecorder.ondataavailable = event => {
        this.audioChunks.push(event.data);
      };
      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        this.uploadAndTranscribe(audioBlob);
      };
      this.mediaRecorder.start();
      this.isRecording = true;
    });
  }

  stopRecording() {
    if (this.mediaRecorder) {
      this.mediaRecorder.stop();
      this.isRecording = false;
    }
  }

  uploadAndTranscribe(audioBlob: Blob) {
    this.statusMessage = 'Uploading audio...';
    this.sttService.uploadAudio(audioBlob).subscribe({
      next: (res) => {
        this.statusMessage = 'Transcribing...';
        this.sttService.transcribeWhisperX({ audio_path: res.path }).subscribe({
          next: (transcription) => {
            this.transcriptionText = transcription.text;
            this.statusMessage = 'Transcription complete.';
          },
          error: (err) => {
            this.statusMessage = 'Transcription failed: ' + err.message;
          }
        });
      },
      error: (err) => {
        this.statusMessage = 'Upload failed: ' + err.message;
      }
    });
  }
}
