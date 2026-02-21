
# todo: move this file to /services/ directory.

import numpy as np
import scipy
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa

class STTQualityAnalyzer:
    """Comprehensive audio quality analyzer for STT training data"""
    
    def __init__(self):
        # Google's requirements for STT microphones
        self.TARGET_SAMPLE_RATE = 16000
        self.MIN_FREQUENCY = 125  # Hz
        self.MAX_FREQUENCY = 8000  # Hz
        self.MAX_THD_PERCENT = 1.0  # 1% THD maximum
        self.MAX_FREQUENCY_DEVIATION = 3.0  # ±3dB frequency response
        
    def analyze_full_quality(self, audio, sample_rate):
        """Run comprehensive quality analysis for STT training"""
        results = {}
        
        # Basic quality checks
        results['snr'] = self.compute_snr(audio)
        results['clipping'] = self.check_clipping(audio)
        results['silence'] = self.check_silence(audio)
        
        # STT-specific checks
        results['frequency_response'] = self.check_frequency_response(audio, sample_rate)
        results['thd'] = self.estimate_thd(audio, sample_rate)
        results['speech_clarity'] = self.check_speech_clarity(audio, sample_rate)
        results['background_noise'] = self.analyze_background_noise(audio, sample_rate)
        results['dynamic_range'] = self.compute_dynamic_range(audio)
        results['spectral_balance'] = self.analyze_spectral_balance(audio, sample_rate)
        results['room_acoustics'] = self.detect_room_issues(audio, sample_rate)
        
        # Overall quality score
        results['quality_score'] = self.compute_quality_score(results)
        results['recommendations'] = self.generate_recommendations(results)
        
        return results
    
    def compute_snr(self, audio):
        """Compute Signal-to-Noise Ratio"""
        if len(audio) == 0:
            return 0.0
        
        # Use top 10% as signal, bottom 10% as noise estimate
        sorted_audio = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_audio[:int(0.1 * len(sorted_audio))])
        signal_level = np.mean(sorted_audio[int(0.9 * len(sorted_audio)):])
        
        if noise_floor == 0:
            return 60.0  # Very high SNR
        
        snr = 20 * np.log10(signal_level / noise_floor)
        return float(np.clip(snr, 0, 60))
    
    def check_clipping(self, audio, threshold=0.95):
        """Check for digital clipping"""
        clipped_samples = np.sum(np.abs(audio) >= threshold)
        clipping_percentage = (clipped_samples / len(audio)) * 100
        return {
            'is_clipping': clipping_percentage > 0.1,  # More than 0.1% clipped
            'percentage': float(clipping_percentage),
            'severity': 'high' if clipping_percentage > 1 else 'medium' if clipping_percentage > 0.1 else 'low'
        }
    
    def check_silence(self, audio, threshold_db=-40):
        """Detect silence periods"""
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms)
        
        silent_frames = np.sum(rms_db < threshold_db)
        silence_percentage = (silent_frames / len(rms_db)) * 100
        
        return {
            'is_mostly_silent': silence_percentage > 80,
            'silence_percentage': float(silence_percentage),
            'has_speech': silence_percentage < 95  # At least 5% non-silent
        }
    
    def check_frequency_response(self, audio, sample_rate):
        """Check if frequency response is suitable for speech (125Hz-8kHz)"""
        # Compute power spectral density
        frequencies, psd = signal.welch(audio, sample_rate, nperseg=2048)
        
        # Focus on speech-relevant frequencies
        speech_mask = (frequencies >= 125) & (frequencies <= 8000)
        speech_freqs = frequencies[speech_mask]
        speech_psd = psd[speech_mask]
        
        if len(speech_psd) == 0:
            return {'quality': 'poor', 'flatness_score': 0}
        
        # Check flatness (variation should be < 6dB for good quality)
        psd_db = 10 * np.log10(speech_psd + 1e-10)
        max_variation = np.max(psd_db) - np.min(psd_db)
        
        # Key frequency ranges for speech intelligibility
        low_speech = (speech_freqs >= 300) & (speech_freqs <= 1000)
        mid_speech = (speech_freqs >= 1000) & (speech_freqs <= 4000)
        high_speech = (speech_freqs >= 4000) & (speech_freqs <= 8000)
        
        energy_distribution = {
            'low_300_1k': float(np.mean(psd_db[low_speech])) if np.any(low_speech) else -60,
            'mid_1k_4k': float(np.mean(psd_db[mid_speech])) if np.any(mid_speech) else -60,
            'high_4k_8k': float(np.mean(psd_db[high_speech])) if np.any(high_speech) else -60
        }
        
        return {
            'quality': 'excellent' if max_variation < 6 else 'good' if max_variation < 12 else 'poor',
            'flatness_score': float(max(0, 100 - max_variation * 5)),
            'variation_db': float(max_variation),
            'energy_distribution': energy_distribution
        }
    
    def estimate_thd(self, audio, sample_rate):
        """Estimate Total Harmonic Distortion"""
        # Simple THD estimation using spectral analysis
        if len(audio) < 2048:
            return {'thd_percent': 0, 'quality': 'unknown'}
        
        # Apply window and compute FFT
        windowed = audio * signal.windows.hann(len(audio))
        fft_result = np.abs(fft(windowed))
        freqs = fftfreq(len(audio), 1/sample_rate)
        
        # Find dominant frequency (fundamental)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_result[:len(fft_result)//2]
        
        # Look for fundamental in speech range (80-1000 Hz)
        speech_fundamental_mask = (positive_freqs >= 80) & (positive_freqs <= 1000)
        if not np.any(speech_fundamental_mask):
            return {'thd_percent': 0, 'quality': 'no_fundamental'}
        
        fundamental_idx = np.argmax(positive_fft[speech_fundamental_mask])
        fundamental_freq = positive_freqs[speech_fundamental_mask][fundamental_idx]
        fundamental_power = positive_fft[speech_fundamental_mask][fundamental_idx]
        
        # Estimate harmonic distortion (simplified)
        harmonic_power = 0
        for harmonic in range(2, 6):  # Check 2nd-5th harmonics
            harmonic_freq = fundamental_freq * harmonic
            if harmonic_freq < sample_rate / 2:
                freq_idx = np.argmin(np.abs(positive_freqs - harmonic_freq))
                harmonic_power += positive_fft[freq_idx] ** 2
        
        if fundamental_power > 0:
            thd_estimate = np.sqrt(harmonic_power) / fundamental_power * 100
        else:
            thd_estimate = 0
            
        return {
            'thd_percent': float(min(thd_estimate, 10)),  # Cap at 10%
            'quality': 'excellent' if thd_estimate < 1 else 'good' if thd_estimate < 3 else 'poor'
        }
    
    def check_speech_clarity(self, audio, sample_rate):
        """Analyze speech clarity and intelligibility"""
        # Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        avg_centroid = np.mean(spectral_centroids)
        
        # Zero crossing rate (consonant clarity)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        avg_zcr = np.mean(zcr)
        
        # Spectral rolloff (frequency content)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        avg_rolloff = np.mean(rolloff)
        
        return {
            'spectral_centroid': float(avg_centroid),
            'zero_crossing_rate': float(avg_zcr),
            'spectral_rolloff': float(avg_rolloff),
            'clarity_score': float(min(100, (avg_centroid / 4000 + avg_zcr * 1000) * 50))
        }
    
    def analyze_background_noise(self, audio, sample_rate):
        """Analyze background noise characteristics"""
        # Detect noise floor
        rms_frames = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        noise_floor = np.percentile(rms_frames, 10)  # Bottom 10%
        
        # Check for consistent noise (AC hum, fan noise, etc.)
        frequencies, psd = signal.welch(audio, sample_rate, nperseg=4096)
        
        # Look for specific noise frequencies
        ac_hum_50hz = self._check_frequency_peak(frequencies, psd, 50, tolerance=2)
        ac_hum_60hz = self._check_frequency_peak(frequencies, psd, 60, tolerance=2)
        
        return {
            'noise_floor_db': float(20 * np.log10(noise_floor + 1e-10)),
            'has_ac_hum_50hz': ac_hum_50hz,
            'has_ac_hum_60hz': ac_hum_60hz,
            'noise_consistency': float(np.std(rms_frames) / np.mean(rms_frames))
        }
    
    def _check_frequency_peak(self, frequencies, psd, target_freq, tolerance=1):
        """Check for significant peak at specific frequency"""
        mask = (frequencies >= target_freq - tolerance) & (frequencies <= target_freq + tolerance)
        if not np.any(mask):
            return False
        
        target_power = np.max(psd[mask])
        avg_power = np.mean(psd)
        
        return target_power > avg_power * 3  # 3x above average
    
    def compute_dynamic_range(self, audio):
        """Compute dynamic range of the audio"""
        if len(audio) == 0:
            return 0
        
        rms_values = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = 20 * np.log10(rms_values + 1e-10)
        
        dynamic_range = np.max(rms_db) - np.min(rms_db)
        return {
            'range_db': float(dynamic_range),
            'quality': 'excellent' if dynamic_range > 20 else 'good' if dynamic_range > 12 else 'poor'
        }
    
    def analyze_spectral_balance(self, audio, sample_rate):
        """Analyze spectral balance for speech training"""
        # Divide spectrum into speech-relevant bands
        frequencies, psd = signal.welch(audio, sample_rate, nperseg=2048)
        psd_db = 10 * np.log10(psd + 1e-10)
        
        bands = {
            'sub_speech': (frequencies < 125),
            'low_speech': (frequencies >= 125) & (frequencies < 500),
            'mid_speech': (frequencies >= 500) & (frequencies < 2000),
            'high_speech': (frequencies >= 2000) & (frequencies < 8000),
            'ultra_high': (frequencies >= 8000)
        }
        
        band_energy = {}
        for band_name, mask in bands.items():
            if np.any(mask):
                band_energy[band_name] = float(np.mean(psd_db[mask]))
            else:
                band_energy[band_name] = -60.0
        
        # Calculate balance score (ideal: most energy in mid_speech)
        balance_score = band_energy['mid_speech'] - np.mean([
            band_energy['low_speech'], 
            band_energy['high_speech']
        ])
        
        return {
            'band_energy': band_energy,
            'balance_score': float(balance_score),
            'is_balanced': balance_score > -6  # Within 6dB
        }
    
    def detect_room_issues(self, audio, sample_rate):
        """Detect room acoustic issues that affect STT quality"""
        # Echo/reverb detection using autocorrelation
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for secondary peaks (echoes)
        # Skip first 50ms to avoid direct signal
        skip_samples = int(0.05 * sample_rate)
        if len(autocorr) > skip_samples:
            echo_region = autocorr[skip_samples:skip_samples + int(0.5 * sample_rate)]
            max_echo = np.max(echo_region) if len(echo_region) > 0 else 0
            main_peak = autocorr[0]
            echo_ratio = max_echo / main_peak if main_peak > 0 else 0
        else:
            echo_ratio = 0
        
        return {
            'echo_ratio': float(echo_ratio),
            'has_echo': echo_ratio > 0.3,
            'reverb_estimate': float(min(echo_ratio * 100, 100))
        }
    
    def compute_quality_score(self, results):
        """Compute overall quality score for STT training suitability"""
        scores = []
        weights = []
        
        # SNR (20% weight)
        if results['snr'] > 30:
            scores.append(100)
        elif results['snr'] > 20:
            scores.append(80)
        elif results['snr'] > 15:
            scores.append(60)
        else:
            scores.append(30)
        weights.append(0.20)
        
        # Clipping (15% weight)
        clipping_score = 100 if not results['clipping']['is_clipping'] else max(0, 100 - results['clipping']['percentage'] * 20)
        scores.append(clipping_score)
        weights.append(0.15)
        
        # Frequency response (20% weight)
        scores.append(results['frequency_response']['flatness_score'])
        weights.append(0.20)
        
        # THD (15% weight)
        thd_score = 100 if results['thd']['thd_percent'] < 1 else max(0, 100 - results['thd']['thd_percent'] * 10)
        scores.append(thd_score)
        weights.append(0.15)
        
        # Dynamic range (10% weight)
        dr_score = 100 if results['dynamic_range']['range_db'] > 20 else max(0, results['dynamic_range']['range_db'] * 5)
        scores.append(dr_score)
        weights.append(0.10)
        
        # Speech clarity (10% weight)
        scores.append(results['speech_clarity']['clarity_score'])
        weights.append(0.10)
        
        # Room acoustics (10% weight)
        room_score = 100 if not results['room_acoustics']['has_echo'] else max(0, 100 - results['room_acoustics']['reverb_estimate'])
        scores.append(room_score)
        weights.append(0.10)
        
        overall_score = np.average(scores, weights=weights)
        return float(overall_score)
    
    def generate_recommendations(self, results):
        """Generate specific recommendations for improving audio quality"""
        recommendations = []
        
        if results['snr'] < 20:
            recommendations.append("Increase microphone gain or move closer to reduce background noise")
        
        if results['clipping']['is_clipping']:
            recommendations.append("Reduce input gain to prevent clipping distortion")
        
        if results['frequency_response']['quality'] == 'poor':
            recommendations.append("Use a microphone with flatter frequency response (±3dB 125Hz-8kHz)")
        
        if results['thd']['thd_percent'] > 3:
            recommendations.append("Check microphone and audio interface for distortion issues")
        
        if results['room_acoustics']['has_echo']:
            recommendations.append("Improve room acoustics - add sound absorption or move to quieter space")
        
        if results['background_noise']['has_ac_hum_50hz'] or results['background_noise']['has_ac_hum_60hz']:
            recommendations.append("Check for electrical interference (AC hum detected)")
        
        if results['silence']['silence_percentage'] > 90:
            recommendations.append("Speak louder or move closer to microphone")
        
        if results['dynamic_range']['range_db'] < 12:
            recommendations.append("Vary speech volume for better dynamic range")
        
        if not results['spectral_balance']['is_balanced']:
            recommendations.append("Adjust microphone positioning for better frequency balance")
        
        if not recommendations:
            recommendations.append("Audio quality is good for STT training!")
        
        return recommendations
    
def quality_check_clip(audio_file):
    """
    Perform a quick quality check on an audio clip for STT suitability. 
    Returns a tuple (passed: bool, quality_score: float). The quality_score is between 0-100.
    """
    # todo: implement this function.
    
    import random
    return True, random.uniform(70, 100)