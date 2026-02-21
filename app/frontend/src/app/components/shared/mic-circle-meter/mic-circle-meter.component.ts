import { Component, Input, ViewChild, ElementRef, AfterViewInit, OnDestroy, effect, EffectRef, Signal, input } from '@angular/core';

@Component({
    selector: 'app-mic-circle-meter',
    template: `<canvas #canvas></canvas>`,
    styles: [`
    canvas {
      width: 100px;
      height: 100px;
      display: block;
    }
  `]
})
export class MicCircleMeterComponent implements AfterViewInit, OnDestroy {
    @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
    level = input(0); // mic level signal (0–1)

    private ctx!: CanvasRenderingContext2D;
    private stopEffect!: EffectRef;

    /** Adjust this exponent: < 1 = more sensitive at low levels, > 1 = less sensitive */
    private readonly power = 0.5; // square-root scaling

    constructor() {
        this.stopEffect = effect(() => {
            this.drawCircle(this.level());
        });
    }

    ngAfterViewInit() {
        const canvas = this.canvasRef.nativeElement;
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        this.ctx = canvas.getContext('2d')!;
        this.drawCircle(this.level());
    }

    private drawCircle(level: number) {
        if (!this.ctx) return;

        const canvas = this.canvasRef.nativeElement;
        const w = canvas.width;
        const h = canvas.height;

        // Apply power-law scaling
        const scaledLevel = Math.pow(level, this.power);

        const minRadius = 20;
        const maxRadius = 50;
        const radius = minRadius + scaledLevel * (maxRadius - minRadius);

        this.ctx.clearRect(0, 0, w, h);

        this.ctx.beginPath();
        this.ctx.arc(w / 2, h / 2, radius, 0, 2 * Math.PI);

        const alpha = 0.4 + 0.6 * scaledLevel;
        this.ctx.fillStyle = `rgba(0, 123, 255, ${alpha})`;
        this.ctx.fill();
    }

    ngOnDestroy() {
        if (this.stopEffect) this.stopEffect.destroy();
    }
}
