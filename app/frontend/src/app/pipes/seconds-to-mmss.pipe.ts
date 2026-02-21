import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'secondsToMmss',
  standalone: true
})
export class SecondsToMmssPipe implements PipeTransform {
  transform(value: number): string {
    if (value == null || isNaN(value)) {
      return '00:00';
    }

    const mins = Math.floor(value / 60);
    const secs = Math.floor(value % 60);

    const pad = (num: number) => (num < 10 ? '0' + num : '' + num);

    return `${pad(mins)}:${pad(secs)}`;
  }
}
