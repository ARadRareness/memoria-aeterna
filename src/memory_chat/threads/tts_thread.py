from pathlib import Path
import uuid
from pydub import AudioSegment
from pydub.playback import play
from PySide6.QtCore import QThread


class TTSThread(QThread):
    def __init__(self, text, amp_client):
        super().__init__()
        self.text = text
        self.amp_client = amp_client

    def run(self):
        try:
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)

            for wav_data in self.amp_client.text_to_speech(self.text):
                if wav_data:
                    try:
                        temp_path = temp_dir / f"tts_{uuid.uuid4()}.wav"

                        with open(temp_path, "wb") as f:
                            f.write(wav_data)

                        AudioSegment.converter = temp_dir
                        audio = AudioSegment.from_wav(str(temp_path))
                        play(audio)

                    finally:
                        if temp_path.exists():
                            temp_path.unlink()

        except Exception as e:
            print(f"Error in TTS process: {e}")
            import traceback

            traceback.print_exc()
