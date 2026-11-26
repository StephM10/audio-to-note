import json
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from audio_to_note.config import load_config
from audio_to_note.diarize import get_diarizer_provider
from audio_to_note.models import SpeakerTurn


def _make_wav_16k(path: Path) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 16000)
    return path


def test_none_diarizer_returns_empty(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    d = get_diarizer_provider("none", tmp_path, cfg)
    assert d.diarize(tmp_path / "a.wav", tmp_path) == []


def test_command_diarizer_parses_turns_json(tmp_path: Path):
    cfg = load_config(project_root=tmp_path)
    cfg["diarization"]["providers"]["command"]["command"] = "my-diar --in {audio} --out {output_json}"
    d = get_diarizer_provider("command", tmp_path, cfg)
    wav = _make_wav_16k(tmp_path / "a.wav")

    def fake_run(cmd, **kwargs):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps([{"start": 0.0, "end": 3.2, "speaker": 1}]), encoding="utf-8")
        return SimpleNamespace(stdout="", stderr="")

    with patch("audio_to_note.diarize.command.subprocess.run", side_effect=fake_run):
        turns = d.diarize(wav, tmp_path / "w")

    assert turns == [SpeakerTurn(0.0, 3.2, "SPEAKER_1")]


def test_sherpa_onnx_diarizer_with_mocked_module(tmp_path: Path):
    seg_m = tmp_path / "seg.onnx"
    emb_m = tmp_path / "emb.onnx"
    seg_m.write_bytes(b"onnx")
    emb_m.write_bytes(b"onnx")
    wav = _make_wav_16k(tmp_path / "a.wav")

    cfg = load_config(project_root=tmp_path)
    cfg["diarization"]["providers"]["sherpa_onnx"]["segmentation_model"] = str(seg_m)
    cfg["diarization"]["providers"]["sherpa_onnx"]["embedding_model"] = str(emb_m)
    d = get_diarizer_provider("sherpa_onnx", tmp_path, cfg)

    fake_sherpa = ModuleType("sherpa_onnx")
    fake_sherpa.FastClusteringConfig = lambda **kw: kw
    fake_sherpa.OfflineSpeakerSegmentationPyannoteModelConfig = lambda **kw: kw
    fake_sherpa.OfflineSpeakerSegmentationModelConfig = lambda **kw: kw
    fake_sherpa.SpeakerEmbeddingExtractorConfig = lambda **kw: kw
    fake_sherpa.OfflineSpeakerDiarizationConfig = lambda **kw: kw

    class FakeSD:
        sample_rate = 16000

        def __init__(self, config):
            pass

        def process(self, samples):
            class Res:
                def sort_by_start_time(self):
                    return [SimpleNamespace(start=0.0, end=1.0, speaker=0)]

            return Res()

    fake_sherpa.OfflineSpeakerDiarization = FakeSD

    fake_np = ModuleType("numpy")
    fake_np.int16 = "int16"
    fake_np.float32 = "float32"

    class FakeArr:
        def astype(self, _):
            return self

        def __truediv__(self, _):
            return [0.0]

    fake_np.frombuffer = lambda buf, dtype=None: FakeArr()

    with patch.dict(sys.modules, {"sherpa_onnx": fake_sherpa, "numpy": fake_np}):
        turns = d.diarize(wav, tmp_path / "w", num_speakers=2)

    assert turns == [SpeakerTurn(0.0, 1.0, "SPEAKER_0")]


def test_pyannote_diarizer_v4_waveform_and_token_kwarg(tmp_path: Path):
    wav = _make_wav_16k(tmp_path / "a.wav")
    cfg = load_config(project_root=tmp_path)
    cfg["diarization"]["providers"]["pyannote"]["hf_token"] = "hf_dummy"
    d = get_diarizer_provider("pyannote", tmp_path, cfg)

    captured = {}

    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def __truediv__(self, other):
            return self

        def unsqueeze(self, dim):
            return self

    fake_torch = ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.device = lambda name: name
    fake_torch.tensor = lambda data, dtype=None: FakeTensor(data)

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            return [
                (SimpleNamespace(start=0.0, end=2.5), None, "SPEAKER_00"),
                (SimpleNamespace(start=2.5, end=4.0), None, "SPEAKER_01"),
            ]

    class FakePipelineObj:
        def to(self, dev):
            captured["device"] = dev
            return self

        def __call__(self, audio_input, **kwargs):
            captured["audio_input"] = audio_input
            captured["call_kwargs"] = kwargs
            return SimpleNamespace(speaker_diarization=FakeAnnotation())

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, token=None):
            captured["model_id"] = model_id
            captured["token"] = token
            return FakePipelineObj()

    fake_pya = ModuleType("pyannote")
    fake_pya_audio = ModuleType("pyannote.audio")
    fake_pya_audio.Pipeline = FakePipeline
    fake_pya.audio = fake_pya_audio

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "pyannote": fake_pya, "pyannote.audio": fake_pya_audio},
    ):
        turns = d.diarize(wav, tmp_path / "w", num_speakers=2)

    assert captured["token"] == "hf_dummy"
    assert captured["device"] == "cpu"
    assert captured["audio_input"]["sample_rate"] == 16000
    assert captured["call_kwargs"] == {"num_speakers": 2}
    assert turns == [
        SpeakerTurn(0.0, 2.5, "SPEAKER_00"),
        SpeakerTurn(2.5, 4.0, "SPEAKER_01"),
    ]
