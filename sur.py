import librosa
import numpy as np
import matplotlib.pyplot as plt

# =======================
# Sargam Reference (Madhya Saptak)
# =======================
sargam_frequencies = {
    'Sa': 240,
    'Re': 270,
    'Ga': 288,
    'Ma': 320,
    'Pa': 360,
    'Dha': 405,
    'Ni': 432
}

# =======================
# Load Audio
# =======================
y, sr = librosa.load("itna na mujhse tu pyar.wav")

# =======================
# Pitch Detection (BEST METHOD)
# =======================
f0, voiced_flag, _ = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C6')
)

# Remove unvoiced frames
f0 = f0[~np.isnan(f0)]
# Step 3: Convert pitch to nearest musical note
notes = librosa.hz_to_note(f0)
# Step 4: Compute tuning deviation in cents
deviation = librosa.hz_to_octs(f0)
print (f"First 10 Deviations: {deviation[:10]}")
print(f" First 10 Notes: {notes[:10]}")
print(f"First 10 Frequencies: {f0[:10]}")
print (f"Length: {len(f0)}")
print(y)


# =======================
# Sargam Matching Function
# =======================
def detect_sargam(freq, tolerance=5):
    octaves = {
        0.5: "Mandra",
        1.0: "Madhya",
        2.0: "Taar"
    }

    for note, base_freq in sargam_frequencies.items():
        for mult, saptak in octaves.items():
            ref_freq = base_freq * mult
            if abs(freq - ref_freq) <= tolerance:
                percent_error = abs(freq - ref_freq) / ref_freq * 100
                return note, saptak, ref_freq, percent_error

    return None, None, None, None

# =======================
# Detect Notes
# =======================
results = []

for freq in f0:
    note, saptak, ref_freq, error = detect_sargam(freq)
    if note:
        results.append((freq, note, saptak, ref_freq, error))

# =======================
# Remove Consecutive Duplicates
# =======================
final_notes = []
for r in results:
    if not final_notes or final_notes[-1][1] != r[1]:
        final_notes.append(r)
print(f"Detected sequence of Sargam notes: {final_notes}")
# =======================
# Print Results
# =======================
print("\nDetected Sargam Notes:\n")
for freq, note, saptak, ref, err in final_notes[:10]:
    print(f"{freq:.2f} Hz → {note} ({saptak}) | Ref: {ref} Hz | Error: {err:.2f}%")

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert to magnitude spectrogram in decibels
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Get the frequencies corresponding to the STFT bins
frequencies = librosa.fft_frequencies(sr=sr, n_fft=D.shape[0]*2 - 2) # n_fft is usually 2 * (number of rows in D - 1)

# Find the peak frequency for each time frame
peak_frequencies = []


for t in range(S_db.shape[1]): # Iterate through each time frame
    # Find the index of the maximum magnitude in the current frame
    peak_bin_index = np.argmax(S_db[:, t])
    # Get the corresponding frequency
    peak_frequencies.append(frequencies[peak_bin_index])
peak_frequencies_np = np.array(peak_frequencies)


# You can now analyze or visualize the 'peak_frequencies' list
print(f"Peak frequencies over time: {peak_frequencies[:10]}...") # Print first 10 for brevity

# Optional: Visualize the spectrogram and the peak frequencies
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.plot(librosa.times_like(peak_frequencies_np), peak_frequencies_np, color='black', linestyle='--', label='Peak Frequency')
plt.colorbar(format="%+2.f dB")
plt.title('Spectrogram with Peak Frequencies')
plt.legend()
plt.tight_layout()
plt.show()
tolerance = 3.0 

print("\nTotal detected frames:", len(f0))
print("Unique notes detected:", len(final_notes))