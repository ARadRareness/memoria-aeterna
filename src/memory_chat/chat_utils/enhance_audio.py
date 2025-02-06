# pip install pedalboard
# pip install noisereduce
from pedalboard.io import AudioFile
from pedalboard import (
    Pedalboard,
    NoiseGate,
    Compressor,
    LowShelfFilter,
    Limiter,
)
import noisereduce as nr


def enhance_audio(input_file: str, output_file: str, sample_rate: int = 44100):
    # loading audio
    with AudioFile(input_file).resampled_to(sample_rate) as f:
        audio = f.read(f.frames)

    # noisereduction
    reduced_noise = nr.reduce_noise(
        y=audio, sr=sample_rate, stationary=True, prop_decrease=0.75
    )

    # enhancing through pedalboard
    board = Pedalboard(
        [
            NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
            Compressor(threshold_db=-16, ratio=4),
            LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
            Limiter(),
        ]
    )

    effected = board(reduced_noise, sample_rate)

    with AudioFile(output_file, "w", sample_rate, effected.shape[0]) as f:
        f.write(effected)


if __name__ == "__main__":
    enhance_audio(
        input_file="input_20250113_005308.wav", output_file="audio_enhanced2.wav"
    )
