import requests, os, shutil, logging
from pydub import AudioSegment
import numpy as np
import aubio
import coloredlogs
from scipy.io import wavfile
from scipy.fft import fft

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
coloredlogs.install(level="DEBUG")

class AudioRequestError(Exception):
    pass

class AudioFormatError(Exception):
    pass

def download_mp3(url:str, path:str):
    log.debug(f"Downloading alert audio from {url}")
    headers = {
        "User-Agent": "BOILER"
    }
    try:
        r = requests.get(url=url, headers=headers, timeout=10)
        if r.headers.get("Content-Type") == "audio/mpeg":
            with open(path, "wb") as file:
                file.write(r.content)
                file.close()
            log.debug(f"Audio downloaded successfully.")
        else:
            raise AudioFormatError
    except requests.ConnectionError:
        log.error(f"A connection error occurred while attempting to download MP3 audio file.", exc_info=True)
        raise AudioRequestError
    except requests.RequestException:
        log.error(f"An error occurred making the download request for the MP3 audio file.", exc_info=True)
        raise AudioRequestError
    except AudioFormatError:
        log.error(f"The requested URL did not resolve to a file with MPEG audio headers. ({url})", exc_info=False)
        raise AudioRequestError

def convert_mp3_to_wav(path_mp3:str, path_wav:str):
    log.debug(f"Converting '{path_mp3}' to '{path_wav}'.")
    audio = AudioSegment.from_mp3(path_mp3)
    audio.set_frame_rate(16000)
    audio.export(path_wav, format="wav", parameters=["-ar", "16000"])
    log.debug(f"Conversion successful.")

def convert_wav_to_mp3(path_wav: str, path_mp3: str, bitrate="192k"):
    log.debug(f"Converting '{path_wav}' to '{path_mp3}'.")
    audio = AudioSegment.from_wav(path_wav)
    audio.export(path_mp3, format="mp3", bitrate=bitrate)
    log.debug(f"Conversion successful.")

def detect_attention_tone_fft(path_wav: str, threshold=0.2):
    sr, audio = wavfile.read(path_wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    window_size = sr * 1
    step = int(sr * 0.25)
    detected_at_frame = None
    for start in range(0, len(audio) - window_size, step):
        window = audio[start:start+window_size]
        window = window * np.hamming(len(window))
        spectrum = np.abs(fft(window))[:window_size//2]
        freqs = np.fft.fftfreq(window_size, 1/sr)[:window_size//2]
        norm = spectrum / np.max(spectrum)
        peak_853 = norm[np.abs(freqs - 853).argmin()]
        peak_960 = norm[np.abs(freqs - 960).argmin()]
        if peak_853 > threshold and peak_960 > threshold:
            detected_at_frame = start + window_size
            log.info(f"FFT confirmed tone at frame {detected_at_frame}")
            break
    if detected_at_frame:
        cut_point = detected_at_frame / sr
        return True, cut_point
    else:
        return False, 0

def scan_attn(path_wav:str):
    win_s = 4096
    hop_s = 512 
    tone_min_freq = 77
    tone_max_freq = 83
    confirm_threshold = 80000 / hop_s
    samplerate = 16000
    s = aubio.source(path_wav, samplerate, hop_s)
    tolerance = 0.8
    pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    total_frames = 0
    hit_frames = 0
    last_hit_frame = 0
    confirmed_hit = False
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        if tone_min_freq < pitch < tone_max_freq:
            hit_frames += 1
            if hit_frames > confirm_threshold:
                log.debug(f"Confirmed tone at frame {total_frames}")
                confirmed_hit = True
                last_hit_frame = total_frames
        else:
            if hit_frames > 0:
                hit_frames = 0
        total_frames += read
        if read < hop_s: break
    duration = total_frames / samplerate
    log.debug(f"Total frames was {total_frames}. Duration: {duration}")
    cut_point = last_hit_frame / samplerate
    if confirmed_hit:
        log.info(f"Found an attention tone, ending at frame {last_hit_frame} ({cut_point} seconds)!")
        return True, cut_point
    fft_result, fft_cut = detect_attention_tone_fft(path_wav)
    if fft_result:
        log.info(f"Fallback FFT found attention tone ending at {fft_cut} seconds.")
        return True, fft_cut
    log.warning(f"Did not detect an attention tone in this audio.")
    return False, 0

def get_length(path_wav:str):
    audio = AudioSegment.from_wav(path_wav)
    length = len(audio) / 1000
    log.debug(f"Got length of {path_wav}, it's {length} seconds.")
    return length

def cut_tail(path_wav:str, path_new_wav:str, cut_point):
    log.debug(f"Trimming tail of WAV file ({path_wav}) at {cut_point} seconds.")    
    audio = AudioSegment.from_wav(path_wav)
    trimmed_audio = audio[:cut_point * 1000]
    trimmed_audio.export(path_new_wav, format="wav")
    log.debug(f"Exported as {path_new_wav}.")

def cut_lead(path_wav:str, path_new_wav:str, cut_point):
    log.debug(f"Trimming lead of WAV file ({path_wav}) at {cut_point} seconds.")
    audio = AudioSegment.from_wav(path_wav)
    trimmed_audio = audio[cut_point * 1000:]
    trimmed_audio.export(path_new_wav, format="wav")
    log.debug(f"Exported as {path_new_wav}.")

def trim_headers(directory:str, target_file:str):
    path_temp_wav = os.path.join(directory, f"audio-temp.wav")
    convert_mp3_to_wav(target_file, path_temp_wav)
    scanned_ATTN, scanned_ATTN_cut = scan_attn(path_wav=path_temp_wav)
    trimmed_target_file = path_temp_wav
    if scanned_ATTN:
        trimmed_target_file = path_temp_wav
        cut_lead(path_wav=path_temp_wav,path_new_wav=trimmed_target_file,cut_point=scanned_ATTN_cut)
    scanned_EOM = True
    if scanned_EOM:
        trimmed_target_file = path_temp_wav
        audio_length = get_length(path_temp_wav)
        cut_EOM_point = audio_length - 4
        cut_tail(path_wav=path_temp_wav, path_new_wav=trimmed_target_file,cut_point=cut_EOM_point)
    path_final_mp3 = os.path.join(directory, f"eas-audio.mp3")
    convert_wav_to_mp3(path_wav=trimmed_target_file,path_mp3=path_final_mp3)
    log.debug(f"Cleaning up temporary WAV file.")
    os.remove(path=path_temp_wav)

if __name__ == "__main__":
    trim_headers(directory="alerts\\86240", target_file="alerts\\86240\\audio.mp3")
