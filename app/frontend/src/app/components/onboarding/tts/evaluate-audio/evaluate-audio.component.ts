import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';

interface EvaluationRow {
  criteria: string;
  selectedGrade: number | null;
}

@Component({
    selector: 'app-evaluate-audio',
    standalone: true,
    templateUrl: './evaluate-audio.component.html',
    imports: [CommonModule, ButtonModule]
})
export class EvaluateAudioComponent implements OnInit {
    evaluationRows: EvaluationRow[] = [
        { criteria: 'Likeness', selectedGrade: null },
        { criteria: 'Naturalness', selectedGrade: null },
        { criteria: 'Intelligability', selectedGrade: null },
        { criteria: 'Artifacts', selectedGrade: null }
    ];

    grades = [1, 2, 3, 4, 5];

    constructor() {}

    ngOnInit() {
    }

    selectGrade(rowIndex: number, grade: number) {
        this.evaluationRows[rowIndex].selectedGrade = grade;
    }

    isGradeSelected(rowIndex: number, grade: number): boolean {
        return this.evaluationRows[rowIndex].selectedGrade === grade;
    }

    isAllGraded(): boolean {
        return this.evaluationRows.every(row => row.selectedGrade !== null);
    }

    saveToCsv() {
        if (!this.isAllGraded()) {
            alert('Please evaluate all criteria before saving.');
            return;
        }

        alert('Evaluation saved successfully!');
        // Reset evaluation for next item
        this.resetEvaluation();
    }

    resetEvaluation() {
        this.evaluationRows = [
            { criteria: 'Likeness', selectedGrade: null },
            { criteria: 'Naturalness', selectedGrade: null },
            { criteria: 'Intelligability', selectedGrade: null },
            { criteria: 'Artifacts', selectedGrade: null }
        ];
    }
}


