import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SttService } from '../../services/stt.service';
import { ProfileService } from '../../services/profile.service';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { ButtonModule } from 'primeng/button';
import { environment } from '../../../environments/environment';
import { FormsModule } from '@angular/forms';

interface MetadataEntry {
  filename: string;
  path: string;
  text: string;
}

interface EvaluationResult {
    filename: string;
    groundTruth: string;
    baseTranscription: string;
    baseWer: number;
    baseCer: number;
    baseDiffHtml: SafeHtml;
    finetunedTranscription: string;
    finetunedWer: number;
    finetunedCer: number;
    finetunedDiffHtml: SafeHtml;
}

@Component({
  selector: 'app-evaluation',
  standalone: true,
  imports: [CommonModule, ButtonModule, FormsModule],
  templateUrl: './evaluation.component.html'
})
export class EvaluationComponent implements OnInit {
  public entries: MetadataEntry[] = [];
  public evaluationResults: EvaluationResult[] = [];
  public loading = false;
  public error: string | null = null;
  
  public datasets: string[] = [];
  public selectedDataset: string = 'audio_prompts';
  
  public finetunedModels: string[] = [];
  public selectedFinetunedModel: string | null = null;
  
  public baseModel = "models/kb-whisper-large";

  public averageMetrics = {
      baseWer: 0,
      baseCer: 0,
      finetunedWer: 0,
      finetunedCer: 0
  };

  constructor(
    private sttService: SttService,
    private profileService: ProfileService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.loadDatasets();
    this.loadFinetunedModels();
  }

  private getProfileFolder(): string {
    return this.profileService.profileUsername() || this.profileService.profileID()!;
  }

  loadDatasets() {
      const profileId = this.getProfileFolder();
      this.sttService.getDatasets(profileId).subscribe({
          next: (res: any) => {
              this.datasets = res.datasets || [];
              if (this.datasets.includes('audio_prompts')) {
                  this.selectedDataset = 'audio_prompts';
              } else if (this.datasets.length > 0) {
                  this.selectedDataset = this.datasets[0];
              }
              this.loadMetadata();
          },
          error: (err) => console.error("Failed to load datasets", err)
      });
  }

  loadFinetunedModels() {
      this.sttService.getFinetunedModels().subscribe({
          next: (res: any) => {
              const models = res.models || [];
              const profileId = this.getProfileFolder();
              this.finetunedModels = models.filter((m: string) => m.includes(profileId)).sort().reverse();
              
              if (this.finetunedModels.length > 0) {
                  this.selectedFinetunedModel = this.finetunedModels[0];
              }
          },
          error: (err) => console.error("Failed to load models", err)
      });
  }

  loadMetadata() {
    if (!this.selectedDataset) return;
    
    this.loading = true;
    this.error = null;
    const folder = this.getProfileFolder();
    
    console.log(`Loading metadata for ${folder}/${this.selectedDataset}`);

    fetch(`${environment.apiUrl}/inference/audio-metadata/${folder}/${this.selectedDataset}`)
      .then((r: Response) => r.json())
      .then((data: any) => {
        this.loading = false;
        if (data.error) {
          this.error = data.error;
          console.error("Metadata error:", data.error);
          return;
        }
        this.entries = data.metadata || [];
        console.log(`Loaded ${this.entries.length} entries`);
        if (this.entries.length === 0) {
            this.error = "No recordings found in this dataset.";
        }
      })
      .catch((err: any) => {
        this.loading = false;
        this.error = String(err);
        console.error("Fetch error:", err);
      });
  }

  async runComparison() {
      if (!this.selectedFinetunedModel) {
          this.error = "Please select a fine-tuned model.";
          return;
      }
      
      this.loading = true;
      this.evaluationResults = [];
      this.error = null;
      
      try {
          // 1. Run Base Model Evaluation
          console.log("Running Base Model Evaluation...");
          await this.sttService.selectModelBackend({
              model_dir: this.baseModel,
              whisper_language: 'Swedish'
          }).toPromise();
          
          const baseResults = await this.processEntries(this.baseModel);
          
          // 2. Run Fine-tuned Model Evaluation
          console.log("Running Fine-tuned Model Evaluation...");
          await this.sttService.selectModelBackend({
              model_dir: this.selectedFinetunedModel,
              whisper_language: 'Swedish'
          }).toPromise();
          
          const finetunedResults = await this.processEntries(this.selectedFinetunedModel);
          
          // 3. Combine Results
          this.combineResults(baseResults, finetunedResults);
          
      } catch (e: any) {
          console.error(e);
          this.error = "Evaluation failed: " + (e.message || e);
      } finally {
          this.loading = false;
      }
  }

  async processEntries(modelName: string): Promise<any[]> {
      const results = [];
      for (const entry of this.entries) {
          try {
              const res = await this.sttService.transcribeBackend(entry.path).toPromise();
              const transcription = res?.text || "";
              
              const evalRes = await this.sttService.evaluateTranscription(
                  this.getProfileFolder(),
                  entry.filename,
                  transcription,
                  this.selectedDataset
              ).toPromise();
              
              results.push({
                  filename: entry.filename,
                  groundTruth: evalRes.ground_truth,
                  transcription: transcription,
                  wer: evalRes.wer,
                  cer: evalRes.cer
              });
          } catch (e) {
              console.error(`Failed to process ${entry.filename} with ${modelName}`, e);
              results.push({
                  filename: entry.filename,
                  groundTruth: "",
                  transcription: "[Error]",
                  wer: 0,
                  cer: 0
              });
          }
      }
      return results;
  }

  combineResults(base: any[], finetuned: any[]) {
      this.evaluationResults = [];
      let totalBaseWer = 0, totalBaseCer = 0;
      let totalFtWer = 0, totalFtCer = 0;
      let count = 0;

      for (let i = 0; i < base.length; i++) {
          const b = base[i];
          const f = finetuned.find(x => x.filename === b.filename) || { transcription: "", wer: 0, cer: 0 };
          
          const baseDiff = this.calculateDiff(b.groundTruth, b.transcription);
          const ftDiff = this.calculateDiff(b.groundTruth, f.transcription);

          this.evaluationResults.push({
              filename: b.filename,
              groundTruth: b.groundTruth,
              baseTranscription: b.transcription,
              baseWer: b.wer,
              baseCer: b.cer,
              baseDiffHtml: this.sanitizer.bypassSecurityTrustHtml(baseDiff),
              finetunedTranscription: f.transcription,
              finetunedWer: f.wer,
              finetunedCer: f.cer,
              finetunedDiffHtml: this.sanitizer.bypassSecurityTrustHtml(ftDiff)
          });

          if (b.groundTruth) {
              totalBaseWer += b.wer;
              totalBaseCer += b.cer;
              totalFtWer += f.wer;
              totalFtCer += f.cer;
              count++;
          }
      }

      if (count > 0) {
          this.averageMetrics = {
              baseWer: totalBaseWer / count,
              baseCer: totalBaseCer / count,
              finetunedWer: totalFtWer / count,
              finetunedCer: totalFtCer / count
          };
      }
  }

  private calculateDiff(expected: string, actual: string): string {
      const normalize = (s: string) => s.toLowerCase().replace(/[.,?!]/g, '').trim();
      
      const eWordsOriginal = expected.trim() ? expected.trim().split(/\s+/) : [];
      const aWordsOriginal = actual.trim() ? actual.trim().split(/\s+/) : [];

      const eWords = eWordsOriginal.map(normalize);
      const aWords = aWordsOriginal.map(normalize);
      
      const ops = this.levenshteinOps(eWords, aWords);
      const parts: string[] = [];
      
      for (const op of ops) {
          if (op.type === 'match') {
              parts.push(`<span class="text-green-600 font-medium">${this.escapeHtml(aWordsOriginal[op.j])}</span>`);
          } else if (op.type === 'sub') {
              parts.push(`<span class="text-red-600 font-bold">${this.escapeHtml(aWordsOriginal[op.j])}</span>`);
          } else if (op.type === 'ins') {
              parts.push(`<span class="text-red-600 font-bold">${this.escapeHtml(aWordsOriginal[op.j])}</span>`);
          } 
          // Ignore 'del'
      }

      return parts.join(' ');
  }

  private levenshteinOps(expected: string[], actual: string[]): any[] {
      const dp: any[][] = [];
      for (let i = 0; i <= expected.length; i++) {
          dp[i] = [];
          for (let j = 0; j <= actual.length; j++) {
              dp[i][j] = 0;
          }
      }

      for (let i = 0; i <= expected.length; i++) dp[i][0] = i;
      for (let j = 0; j <= actual.length; j++) dp[0][j] = j;

      for (let i = 1; i <= expected.length; i++) {
          for (let j = 1; j <= actual.length; j++) {
              if (expected[i - 1] === actual[j - 1]) {
                  dp[i][j] = dp[i - 1][j - 1];
              } else {
                  dp[i][j] = Math.min(
                      dp[i - 1][j - 1] + 1, // sub
                      dp[i][j - 1] + 1,     // ins
                      dp[i - 1][j] + 1      // del
                  );
              }
          }
      }

      // Backtrack
      let i = expected.length;
      let j = actual.length;
      const ops = [];

      while (i > 0 || j > 0) {
          if (i > 0 && j > 0 && expected[i - 1] === actual[j - 1]) {
              ops.unshift({ type: 'match', i: i - 1, j: j - 1 });
              i--; j--;
          } else if (i > 0 && j > 0 && dp[i][j] === dp[i - 1][j - 1] + 1) {
              ops.unshift({ type: 'sub', i: i - 1, j: j - 1 });
              i--; j--;
          } else if (j > 0 && dp[i][j] === dp[i][j - 1] + 1) {
              ops.unshift({ type: 'ins', j: j - 1 });
              j--;
          } else {
              ops.unshift({ type: 'del', i: i - 1 });
              i--;
          }
      }
      return ops;
  }

  private escapeHtml(s: string) {
    return (s || '').replace(/[&<>\"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'} as any)[c] || c);
  }
}
