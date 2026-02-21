import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import json
import os
import sys
import threading
import time
import csv

# Default constant, can be overridden by selection
DEFAULT_DATA_FILE = "data_script.jsonl"
OUTPUT_DIR = "recordings"
SAMPLE_RATE = 44100
CHANNELS = 1

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.stream = None

    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback)
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        return np.concatenate(self.audio_data, axis=0) if self.audio_data else None

    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder")
        self.root.geometry("900x600")
        
        self.username = self.get_username()
        if not self.username:
            self.root.destroy()
            return

        self.data_file = self.get_manuscript()
        if not self.data_file:
            self.root.destroy()
            return
            
        self.manuscript_name = os.path.splitext(os.path.basename(self.data_file))[0]
        
        # Create user directory specific to the manuscript to avoid collisions
        self.user_dir = os.path.join(OUTPUT_DIR, self.username, self.manuscript_name)
        os.makedirs(self.user_dir, exist_ok=True)
        
        self.sentences = []
        self.load_sentences()
        
        self.recorder = AudioRecorder()
        self.current_index = 0
        self.is_recording = False
        self.is_playing = False
        self.updating_ui = False
        
        self.setup_ui()
        self.load_sentence(0)
        
        # Bind spacebar to record toggle
        self.root.bind('<space>', self.toggle_recording)
        # Bind navigation and delete keys
        self.root.bind('<Left>', self.prev_sentence)
        self.root.bind('<Right>', self.next_sentence)
        self.root.bind('<Delete>', lambda e: self.delete_recording())

    def get_username(self):
        # Check if username was passed as arg
        if len(sys.argv) > 1:
            return sys.argv[1]
        return simpledialog.askstring("Username", "Input username:")

    def get_manuscript(self):
        # List jsonl files in current directory
        files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
        if not files:
            messagebox.showerror("Error", "No .jsonl files found in current directory!")
            return None
        
        # If only one, just use it? Or still ask? User said "choose".
        # Let's show a dialog to choose.
        
        # Create a simple dialog window for selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Manuscript")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        selected_file = tk.StringVar()
        
        ttk.Label(dialog, text="Choose a manuscript:").pack(pady=10)
        
        combo = ttk.Combobox(dialog, textvariable=selected_file, values=files)
        combo.pack(pady=5)
        if files:
            combo.current(0)
            
        def on_ok():
            dialog.destroy()
            
        ok_btn = ttk.Button(dialog, text="OK", command=on_ok)
        ok_btn.pack(pady=10)
        
        self.root.wait_window(dialog)
        
        return selected_file.get()

    def load_sentences(self):
        if not os.path.exists(self.data_file):
            messagebox.showerror("Error", f"Data file {self.data_file} not found!")
            sys.exit(1)
            
        with open(self.data_file, encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                self.sentences.append(entry)
            except json.JSONDecodeError:
                continue

    def setup_ui(self):
        # Main layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar (List of sentences)
        left_frame = ttk.Frame(main_frame, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        ttk.Label(left_frame, text="Sentences").pack(anchor=tk.W)
        
        self.sentence_listbox = tk.Listbox(left_frame, width=30)
        self.sentence_listbox.pack(fill=tk.BOTH, expand=True)
        self.sentence_listbox.bind('<<ListboxSelect>>', self.on_select_sentence)
        self.sentence_listbox.bind('<space>', self.toggle_recording)
        
        # Right content
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Info
        info_text = f"User: {self.username} | Manuscript: {self.manuscript_name}"
        self.info_label = ttk.Label(right_frame, text=info_text, font=("Arial", 12, "bold"))
        self.info_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Text Display
        self.text_display = tk.Text(right_frame, height=8, width=50, font=("Arial", 16), wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        self.text_display.config(state=tk.DISABLED)
        
        # Jump to index
        jump_frame = ttk.Frame(right_frame)
        jump_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(jump_frame, text="Jump to Index:").pack(side=tk.LEFT)
        self.index_entry = ttk.Entry(jump_frame, width=10)
        self.index_entry.pack(side=tk.LEFT, padx=5)
        self.index_entry.bind('<Return>', self.jump_to_index)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_index).pack(side=tk.LEFT)

        # Status
        self.status_label = ttk.Label(right_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(pady=(0, 10))
        
        # Controls
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        self.prev_btn = ttk.Button(controls_frame, text="< Previous", command=self.prev_sentence, takefocus=0)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.prev_btn.bind('<space>', self.toggle_recording)
        
        self.record_btn = ttk.Button(controls_frame, text="Record (Space)", command=self.toggle_recording, takefocus=0)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        self.record_btn.bind('<space>', self.toggle_recording)
        
        self.play_btn = ttk.Button(controls_frame, text="Play", command=self.play_recording, takefocus=0)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn.bind('<space>', self.toggle_recording)
        
        self.delete_btn = ttk.Button(controls_frame, text="Delete/Undo", command=self.delete_recording, takefocus=0)
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        self.delete_btn.bind('<space>', self.toggle_recording)
        
        self.next_btn = ttk.Button(controls_frame, text="Next >", command=self.next_sentence, takefocus=0)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn.bind('<space>', self.toggle_recording)
        
        self.populate_list()

    def populate_list(self):
        self.sentence_listbox.delete(0, tk.END)
        for i, s in enumerate(self.sentences):
            self.update_list_item(i, insert=True)

    def update_list_item(self, index, insert=False):
        if not (0 <= index < len(self.sentences)): return
        
        self.updating_list = True
        try:
            s = self.sentences[index]
            filename = self.get_filename(s['sentence'])
            exists = os.path.exists(filename)
            status = "✓" if exists else " "
            text = f"{status} {s['sentence']}: {s['text'][:20]}..."
            
            if not insert:
                self.sentence_listbox.delete(index)
                
            self.sentence_listbox.insert(index, text)
            if exists:
                self.sentence_listbox.itemconfig(index, {'fg': 'gray50'})
            else:
                self.sentence_listbox.itemconfig(index, {'fg': 'black'})
                
            # Restore selection if this was the selected item
            if index == self.current_index:
                 self.sentence_listbox.selection_set(index)
                 self.sentence_listbox.activate(index)
                 
        finally:
            self.updating_list = False

    def update_selection(self):
        # Clear previous selection
        self.sentence_listbox.selection_clear(0, tk.END)
        
        # Set new selection
        if 0 <= self.current_index < self.sentence_listbox.size():
            self.sentence_listbox.selection_set(self.current_index)
            self.sentence_listbox.activate(self.current_index)
            self.sentence_listbox.see(self.current_index)

    def get_filename(self, idx_str):
        try:
            idx = int(idx_str)
        except ValueError:
            idx = idx_str
        return os.path.join(self.user_dir, f"{self.username}_sentence{idx}.wav")

    def jump_to_index(self, event=None):
        try:
            idx = int(self.index_entry.get())
            if self.is_recording:
                self.stop_recording(auto_advance=False)
            self.load_sentence(idx)
        except ValueError:
            pass

    def load_sentence(self, index):
        if 0 <= index < len(self.sentences):
            self.current_index = index
            s = self.sentences[index]
            
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete("1.0", tk.END)
            self.text_display.insert("1.0", s['text'])
            self.text_display.config(state=tk.DISABLED)
            
            # Update index entry
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, str(index))
            
            self.update_selection()
            self.update_buttons()
            self.status_label.config(text=f"Sentence {s['sentence']}")
            
            # Ensure focus is on listbox for keyboard navigation
            self.sentence_listbox.focus_set()

    def update_buttons(self):
        if not self.sentences: return
        s = self.sentences[self.current_index]
        filename = self.get_filename(s['sentence'])
        exists = os.path.exists(filename)
        
        self.play_btn.config(state=tk.NORMAL if exists else tk.DISABLED)
        self.delete_btn.config(state=tk.NORMAL if exists else tk.DISABLED)
        
        if self.is_recording:
            self.record_btn.config(text="Stop (Space)")
            self.play_btn.config(state=tk.DISABLED)
            self.delete_btn.config(state=tk.DISABLED)
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.sentence_listbox.config(state=tk.DISABLED)
        else:
            self.record_btn.config(text="Record (Space)")
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            self.sentence_listbox.config(state=tk.NORMAL)

    def on_select_sentence(self, event):
        if hasattr(self, 'updating_list') and self.updating_list:
            return
            
        selection = self.sentence_listbox.curselection()
        if selection:
            if selection[0] != self.current_index:
                if self.is_recording:
                    self.stop_recording(auto_advance=False)
                self.load_sentence(selection[0])

    def prev_sentence(self, event=None):
        if self.is_recording:
            self.stop_recording(auto_advance=False)
        if self.current_index > 0:
            self.load_sentence(self.current_index - 1)
        return "break"

    def next_sentence(self, event=None):
        if self.is_recording:
            self.stop_recording(auto_advance=False)
        if self.current_index < len(self.sentences) - 1:
            self.load_sentence(self.current_index + 1)
        return "break"

    def toggle_recording(self, event=None):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
        return "break"

    def start_recording(self):
        self.is_recording = True
        self.status_label.config(text="Recording... Press Space to Stop", foreground="red")
        self.update_buttons()
        self.recorder.start_recording()

    def stop_recording(self, auto_advance=True):
        self.is_recording = False
        self.status_label.config(text="Saving...", foreground="black")
        audio = self.recorder.stop_recording()
        
        if audio is not None:
            s = self.sentences[self.current_index]
            filename = self.get_filename(s['sentence'])
            sf.write(filename, audio, SAMPLE_RATE)
            self.update_metadata(s, filename, len(audio)/SAMPLE_RATE)
            self.status_label.config(text="Saved.")
            
            # Ensure file exists before moving on (short wait)
            start_wait = time.time()
            while not os.path.exists(filename) and time.time() - start_wait < 1.0:
                time.sleep(0.05)
            
            # Update the list item to show checkmark and gray color
            self.update_list_item(self.current_index)
            
            if auto_advance:
                # Auto-advance to next sentence
                if self.current_index < len(self.sentences) - 1:
                    self.next_sentence()
                else:
                    # If last sentence, just refresh to show the checkmark
                    self.load_sentence(self.current_index)
        else:
            self.status_label.config(text="No audio recorded.")
            self.load_sentence(self.current_index) # Refresh UI


    def play_recording(self):
        s = self.sentences[self.current_index]
        filename = self.get_filename(s['sentence'])
        if os.path.exists(filename):
            data, fs = sf.read(filename)
            self.status_label.config(text="Playing...")
            sd.play(data, fs)

    def delete_recording(self):
        s = self.sentences[self.current_index]
        filename = self.get_filename(s['sentence'])
        if os.path.exists(filename):
            if messagebox.askyesno("Confirm", "Delete this recording?"):
                os.remove(filename)
                self.update_list_item(self.current_index)
                self.load_sentence(self.current_index)
                self.status_label.config(text="Deleted.")

    def update_metadata(self, sentence_entry, filename, duration):
        # Update metadata.json only
        meta_path = os.path.join(self.user_dir, "metadata.json")
        metadata = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except:
                pass
        
        metadata = [m for m in metadata if m['sentence'] != sentence_entry['sentence']]
        
        metadata.append({
            "username": self.username,
            "manuscript": self.manuscript_name,
            "sentence": sentence_entry['sentence'],
            "filename": os.path.basename(filename),
            "text": sentence_entry['text'],
            "duration": round(duration, 2),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        def sort_key(m):
            try:
                return int(m['sentence'])
            except:
                return m['sentence']
        
        metadata.sort(key=sort_key)
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def save_csv(self):
        # Re-generate CSV from metadata.json
        meta_path = os.path.join(self.user_dir, "metadata.json")
        csv_path = os.path.join(self.user_dir, f"{self.username}_{self.manuscript_name}_recordings_data.csv")
        
        if not os.path.exists(meta_path):
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except:
            return

        # Filter out entries where file doesn't exist (sync with reality)
        valid_metadata = []
        for m in metadata:
            # Check if file actually exists
            full_path = os.path.join(self.user_dir, m['filename'])
            if os.path.exists(full_path):
                valid_metadata.append(m)
        
        # Write CSV
        with open(csv_path, "w", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_ALL)
            for m in valid_metadata:
                writer.writerow([m['filename'], m['text']])
        
        print(f"CSV saved to {csv_path}")

    def on_closing(self):
        self.save_csv()
        self.root.destroy()

def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    
    app = RecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
