#!/usr/bin/env python3
"""
Fine-Tune Nemotron 3.5 ASR Streaming Model with Custom Dataset & Speech Hints
==============================================================================

Combines:
  - Dataset preparation (audio conversion + manifest building) from the
    customize workflow (traintestset/  ->  custom_asr_data/)
  - Fine-tuning of nemotron-3.5-asr-streaming-0.6b using NeMo Python API
    (from asr-finetune-nemotron-3.5-asr-streaming-prompt notebook)
  - Optional speech-hint grammar post-processing (from
    asr-customize-speechhints) for inverse text normalization on transcripts

Uses the installed NeMo package directly -- no cloned repo needed.

Usage:
    python asr_finetune_with_speechhints.py                     # full pipeline
    python asr_finetune_with_speechhints.py --convert-only       # step 1 only
    python asr_finetune_with_speechhints.py --manifest-only      # steps 1-2
    python asr_finetune_with_speechhints.py --train-only         # step 3 only
    python asr_finetune_with_speechhints.py --evaluate           # step 4 only
    python asr_finetune_with_speechhints.py --apply-speechhints  # post-process transcripts

Prerequisites:
    - GPU with CUDA
    - NeMo toolkit installed (nemo_toolkit[asr])
    - ffmpeg, sox installed
    
pip uninstall nemo_toolkit -y
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # tutorials/

# Raw speaker data (p1/, p2/, ..., each with *.wav + transcript.csv)
RAW_DATA_DIR = os.path.join(DATA_DIR, "traintestset")

# Output: converted WAVs + JSON manifests
CUSTOM_DATA_DIR = os.path.join(DATA_DIR, "custom_asr_data")
CONVERTED_WAVS_DIR = os.path.join(CUSTOM_DATA_DIR, "wavs")

# Pretrained model from HuggingFace
PRETRAINED_MODEL = os.path.join(
    DATA_DIR, "pretrained_model", "nemotron-3.5-asr-streaming-0.6b.nemo"
)

# Checkpoint output
CHECKPOINT_DIR = os.path.join(
    DATA_DIR, "checkpoints", "FastConformer-Transducer-BPE-Prompt-Streaming", "test"
)

# Dataset split
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Speech Hints (from asr-customize-speechhints.ipynb)
# ---------------------------------------------------------------------------
# Available grammar symbols for inverse text normalization:
#   $OOV_NUMERIC_SEQUENCE, $OOV_ALPHA_SEQUENCE, $OOV_ALPHA_NUMERIC_SEQUENCE
#   $FULLPHONENUM, $POSTALCODE, $OOV_CLASS_ORDINAL, $OOV_CLASS_NUMERIC
#   $PERCENT, $TIME, $MONEY, $MONTH, $DAY
#
# When --apply-speechhints is used, we try to normalize transcript text
# using the speech_hint library.  If the library is not installed, we fall
# back gracefully (transcripts are kept as-is).

SPEECH_HINT_RULES = [
    # Phone numbers like "one eight hundred five five five four oh oh one"
    (r"\d[\d\s]{9,}\d", "$FULLPHONENUM"),
    # Percentages
    (r"\d+\s*percent", "$PERCENT"),
    # Time expressions
    (r"\d{1,2}:\d{2}", "$TIME"),
    # Money
    (r"\$\d+[\.,]?\d*", "$MONEY"),
]

try:
    from speech_hint import apply_hint  # type: ignore
    SPEECH_HINT_AVAILABLE = True
except ImportError:
    SPEECH_HINT_AVAILABLE = False


def normalize_with_speech_hints(text: str) -> str:
    """Apply speech-hint grammars to normalize a transcript string.

    This is the Python-side equivalent of what the asr-customize-speechhints
    notebook demonstrates with FST-based grammars.  Each rule attempts an
    inverse-text-normalization pass; if apply_hint raises, we skip that rule.
    """
    if not SPEECH_HINT_AVAILABLE:
        return text

    for _, grammar in SPEECH_HINT_RULES:
        try:
            text = apply_hint(text, grammar)
        except Exception:
            pass  # Grammar did not match; move on
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)
    os.makedirs(CONVERTED_WAVS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[INFO] {msg}", flush=True)


def warn(msg: str):
    print(f"[WARN] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1  -  Convert audio (stereo 48kHz -> mono 16kHz)
# ---------------------------------------------------------------------------
def convert_audio():
    """Convert all WAV files from raw speaker dirs to mono 16kHz.

    Reads *.wav from traintestset/p{N}/, produces p{N}_{file}.wav in
    custom_asr_data/wavs/.
    """
    ensure_dirs()
    log(f"Scanning for WAV files in {RAW_DATA_DIR} ...")

    converted = 0
    skipped = 0
    speaker_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "p*")))

    for spk_dir in speaker_dirs:
        if not os.path.isdir(spk_dir):
            continue
        spk_name = os.path.basename(spk_dir)
        for wav_path in glob.glob(os.path.join(spk_dir, "*.wav")):
            basename = os.path.basename(wav_path)
            out_name = f"{spk_name}_{basename}"
            out_path = os.path.join(CONVERTED_WAVS_DIR, out_name)

            if os.path.exists(out_path):
                skipped += 1
                continue

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-ac", "1",        # mono
                    "-ar", "16000",    # 16 kHz (ASR model expectation)
                    "-loglevel", "error",
                    out_path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                warn(f"ffmpeg failed for {wav_path}: {result.stderr.strip()}")
                continue

            converted += 1

    log(f"Converted: {converted} | Skipped (already done): {skipped}")
    log(f"Output directory: {CONVERTED_WAVS_DIR}")


# ---------------------------------------------------------------------------
# Step 2  -  Build NeMo JSON manifests
# ---------------------------------------------------------------------------
def build_manifests(apply_speech_hints: bool = False):
    """Read transcript.csv from each speaker dir and produce train/test
    JSON manifests in the format expected by NeMo's data loader.

    Each manifest line is a JSON object:
        {"audio_filepath": "...", "duration": ..., "text": "...",
         "lang": "en-US", "target_lang": "en-US"}

    Optionally applies speech-hint grammars to normalize transcript text.
    """
    ensure_dirs()
    random.seed(RANDOM_SEED)

    log("Building manifests ...")
    all_entries = []
    speaker_dirs = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "p*")))

    for spk_dir in speaker_dirs:
        if not os.path.isdir(spk_dir):
            continue
        csv_path = os.path.join(spk_dir, "transcript.csv")
        if not os.path.exists(csv_path):
            warn(f"No transcript.csv in {spk_dir}, skipping.")
            continue

        spk_name = os.path.basename(spk_dir)

        with open(csv_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or "|" not in line:
                    continue

                wav_name, text = line.split("|", maxsplit=1)
                wav_name = wav_name.strip()
                text = text.strip()

                # Optional speech-hint normalization on transcripts
                if apply_speech_hints:
                    text = normalize_with_speech_hints(text)

                converted_name = f"{spk_name}_{wav_name}"
                wav_path = os.path.join(CONVERTED_WAVS_DIR, converted_name)

                if not os.path.exists(wav_path):
                    warn(f"Converted file {converted_name} not found (line {line_num})")
                    continue

                # Duration via ffprobe
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", wav_path,
                    ],
                    capture_output=True, text=True,
                )
                duration = float(result.stdout.strip())

                all_entries.append({
                    "audio_filepath": os.path.abspath(wav_path),
                    "duration": round(duration, 4),
                    "text": text,
                    "language": "en-US",       # required by lhotse Cut.supervisions[0].language
                    "lang": "en-US",
                    "target_lang": "en-US",
                })

    # Shuffle and split
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * TRAIN_SPLIT)
    train_entries = all_entries[:split_idx]
    test_entries = all_entries[split_idx:]

    # Write manifests (JSONL format)
    train_manifest = os.path.join(CUSTOM_DATA_DIR, "train_manifest.json")
    test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")

    for path, entries in [(train_manifest, train_entries),
                          (test_manifest, test_entries)]:
        with open(path, "w") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")

    total_duration = sum(e["duration"] for e in all_entries)
    log(f"Total samples : {len(all_entries)}")
    log(f"Total duration: {total_duration:.1f}s ({total_duration / 3600:.2f} hrs)")
    log(f"Train samples : {len(train_entries)} -> {train_manifest}")
    log(f"Test  samples : {len(test_entries)} -> {test_manifest}")

    # Show sample entries
    log("\n--- Sample manifest entries ---")
    for entry in all_entries[:3]:
        name = os.path.basename(entry["audio_filepath"])
        txt = entry["text"][:80]
        if len(entry["text"]) > 80:
            txt += "..."
        log(f"  {name} | dur={entry['duration']:.1f}s | {txt}")

    return train_manifest, test_manifest


# ---------------------------------------------------------------------------
# Step 3  -  Fine-tune with NeMo Python API (no cloned repo needed)
# ---------------------------------------------------------------------------
def run_training(train_manifest: str, test_manifest: str, epochs: int = 200,
                 lr: float = 0.1):
    """Fine-tune the pretrained model using NeMo's Python API directly.

    Loads EncDecRNNTBPEModelWithPrompt from the .nemo file, updates data
    and optimizer configs, then trains with pytorch_lightning.Trainer.
    """
    import warnings
    warnings.filterwarnings("ignore")

    if not os.path.exists(PRETRAINED_MODEL):
        print(f"\n[ERROR] Pretrained model not found: {PRETRAINED_MODEL}")
        print("Download with:")
        print(
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('nvidia/nemotron-3.5-asr-streaming-0.6b', "
            f"local_dir='{os.path.dirname(PRETRAINED_MODEL)}')\""
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Import NeMo model class and Lightning Trainer
    # ------------------------------------------------------------------
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger

    # ------------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------------
    model_size = os.path.getsize(PRETRAINED_MODEL) / 1024 ** 3
    log(f"Loading pretrained model: {PRETRAINED_MODEL} ({model_size:.1f} GB)")

    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=PRETRAINED_MODEL,
        map_location="cpu",
    )
    log("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Update data configs
    # ------------------------------------------------------------------
    # Keep lhotse (required for prompt indices in this model).
    OmegaConf.set_struct(model.cfg.train_ds, False)
    model.cfg.train_ds.manifest_filepath = train_manifest
    model.cfg.train_ds.is_tarred = False
    model.cfg.train_ds.shuffle = True
    model.cfg.train_ds.num_workers = 4
    model.cfg.train_ds.max_duration = 40      # default=20 drops clips >20s; raise to use all data
    model.cfg.train_ds.batch_duration = 100   # smaller for our dataset
    OmegaConf.set_struct(model.cfg.train_ds, True)

    OmegaConf.set_struct(model.cfg.validation_ds, False)
    model.cfg.validation_ds.manifest_filepath = test_manifest
    model.cfg.validation_ds.is_tarred = False
    model.cfg.validation_ds.num_workers = 2
    model.cfg.validation_ds.batch_size = 8
    OmegaConf.set_struct(model.cfg.validation_ds, True)

    # ------------------------------------------------------------------
    # Update optimizer config (from notebook recommendations)
    # ------------------------------------------------------------------
    OmegaConf.set_struct(model.cfg.optim, False)
    model.cfg.optim.name = "adamw"
    model.cfg.optim.lr = lr
    model.cfg.optim.weight_decay = 0.001
    model.cfg.optim.sched.warmup_steps = 100
    OmegaConf.set_struct(model.cfg.optim, True)

   # ------------------------------------------------------------------
    # Set up data loaders on the model
    # ------------------------------------------------------------------
    log("Setting up training and validation data ...")
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)

    train_dl = model.train_dataloader()
    val_dl = model.val_dataloader()
    try:
        log(f"Training batches: {len(train_dl)}")
    except TypeError:
        log("Training batches: dynamic (bucketing sampler)")
    try:
        log(f"Validation batches: {len(val_dl)}")
    except TypeError:
        log("Validation batches: dynamic (bucketing sampler)")

    # ------------------------------------------------------------------
    # Override configure_optimizers to avoid NeMo's prepare_lr_scheduler,
    # which crashes because lhotse samplers don't expose batch_size.
    # We build the optimizer directly from model.cfg.optim.
    # ------------------------------------------------------------------
    import torch as _torch

    def _custom_configure_optimizers(self):
        optim_cfg = self.cfg.optim
        optimizer_cls = getattr(_torch.optim, optim_cfg.name.capitalize(), _torch.optim.AdamW)
        optimizer = optimizer_cls(
            self.parameters(),
            lr=optim_cfg.lr,
            betas=list(optim_cfg.betas) if hasattr(optim_cfg, "betas") else [0.9, 0.98],
            weight_decay=optim_cfg.weight_decay,
        )
        # NeMo expects _optimizer to be set for training_step logging
        self._optimizer = optimizer
        log(f"Optimizer: {optim_cfg.name}, lr={optim_cfg.lr}, wd={optim_cfg.weight_decay}")
        return optimizer

    model.configure_optimizers = _custom_configure_optimizers.__get__(
        model, type(model)
    )
    log("Overrode configure_optimizers (bypasses NeMo LR scheduler)")
    log("Optimizer configured.")

    # ------------------------------------------------------------------
    # Training callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="nemotron-asr-finetuned-{epoch:02d}-{global_step}",
        save_top_k=3,
        monitor="val_wer",
        mode="min",
        every_n_epochs=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(DATA_DIR, "checkpoints", "tb_logs"),
        name="nemotron-asr-finetune",
    )

    # ------------------------------------------------------------------
    # PyTorch Lightning Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        devices=1,
        max_epochs=epochs,
        precision="bf16-mixed",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=tb_logger,
        accumulate_grad_batches=1,
        gradient_clip_val=5.0,
        log_every_n_steps=10,
        val_check_interval=1.0,  # validate once per epoch
        enable_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Train!
    # ------------------------------------------------------------------
    log(f"\nStarting fine-tuning ({epochs} epochs, lr={lr}) ...\n")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # ------------------------------------------------------------------
    # Convert top-3 best checkpoints (lowest WER) to .nemo
    # ------------------------------------------------------------------
    # ModelCheckpoint.best_k_models is a dict {ckpt_path: monitor_score}
    best_k = checkpoint_cb.best_k_models     # type: dict[str, float]

    if best_k:
        import torch as _torch

        # Sort ascending (mode="min" → lowest WER first)
        sorted_best = sorted(best_k.items(), key=lambda kv: kv[1])
        log(f"\nConverting {len(sorted_best)} best checkpoint(s) to .nemo ...")

        # Save the training model's final state so we can restore it after
        # swapping in each best-ckpt's weights.  We reuse `model` because
        # building a fresh EncDecRNNTBPEModelWithPrompt from cfg fails
        # (tokenizer artifact registration needs nemo_file_folder).
        final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        for rank, (ckpt_path, wer_val) in enumerate(sorted_best, start=1):
            wer_str = f"{wer_val:.4f}" if wer_val is not None else "wer-unknown"
            nemo_name = f"nemotron-asr-best{rank}-wer-{wer_str}.nemo"
            nemo_path = os.path.join(CHECKPOINT_DIR, nemo_name)

            log(f"  Best #{rank} (WER={wer_str}): loading {os.path.basename(ckpt_path)} ...")

            # .ckpt files are Lightning checkpoints (ZIP/PK), not NeMo archives
            # (tar.gz).  Load the state_dict and swap it into the existing model.
            ckpt_data = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt_data["state_dict"]

            # Lightning prefixes keys with "model." — strip if present
            if any(k.startswith("model.") for k in sd):
                sd = {k[len("model."):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            model.save_to(nemo_path)
            log(f"    -> Saved: {nemo_path}")

        # Restore the final-epoch weights back into `model`
        model.load_state_dict(final_state)
    else:
        log("\nNo best checkpoints found to convert.")

    # ------------------------------------------------------------------
    # Save final .nemo checkpoint (last epoch, regardless of WER)
    # ------------------------------------------------------------------
    final_nemo = os.path.join(CHECKPOINT_DIR, "nemotron-asr-finetuned.nemo")
    model.save_to(final_nemo)
    log(f"\nTraining complete!")
    log(f"Final model saved to: {final_nemo}")
    log(f"Best checkpoints in:  {CHECKPOINT_DIR}")


# ---------------------------------------------------------------------------
# Step 4  -  Evaluate (CER / WER) using NeMo Python API
# ---------------------------------------------------------------------------
def find_nemo_checkpoint() -> str | None:
    """Search common output locations for the latest .nemo file."""
    search_bases = [CHECKPOINT_DIR, os.path.join(DATA_DIR, "checkpoints")]

    for base in search_bases:
        matches = glob.glob(os.path.join(base, "**", "*.nemo"), recursive=True)
        # Exclude the pretrained model
        matches = [m for m in matches if "pretrained_model" not in m]
        if matches:
            return max(matches, key=os.path.getmtime)  # latest
    return None


def run_evaluation(test_manifest: str):
    """Run streaming inference on the test manifest to compute CER/WER.

    Loads each WAV as a tensor and calls the model's forward pass directly,
    avoiding NeMo's .transcribe() which requires lhotse Cuts with language fields.
    """
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    import librosa
    from nemo.collections.asr.models import EncDecRNNTBPEModelWithPrompt
    from lightning.pytorch import Trainer

    nemo_file = find_nemo_checkpoint()
    if not nemo_file:
        print(f"\n[ERROR] No .nemo checkpoint found.")
        print(f"Searched under: {CHECKPOINT_DIR} and {DATA_DIR}/checkpoints/")
        sys.exit(1)

    log(f"Evaluating checkpoint: {nemo_file}")

    dummy_trainer = Trainer(devices=1, accelerator="gpu")
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=nemo_file,
        map_location="cpu",
        trainer=dummy_trainer,
    )
    model.trainer = dummy_trainer
    model.eval()
    model.cuda()

    # Read test manifest
    entries = []
    with open(test_manifest) as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    log(f"Transcribing {len(entries)} samples ...")
    hyps = []
    refs = []

    for i, entry in enumerate(entries):
        audio_path = entry["audio_filepath"]
        ref_text = entry["text"]
        refs.append(ref_text)

        # Load with librosa (no CUDA dependency issues like torchaudio)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(waveform).unsqueeze(0).float().cuda()
        audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).cuda()

        with torch.no_grad():
            logits, logit_len = model(audio_samples=audio_tensor,
                                      length=audio_len)

        # Greedy argmax decode
        decoded_ids = logits.argmax(dim=-1)[0, :logit_len.item()]
        hyp_text = model.tokenizer.ids_to_tokens(decoded_ids.cpu().numpy())
        hyp_text = " ".join(hyp_text).strip()
        hyps.append(hyp_text)

        if (i + 1) % 5 == 0 or i == len(entries) - 1:
            log(f"  Transcribed {i+1}/{len(entries)}")

    # Compute WER using NeMo's built-in metric
    from nemo.collections.asr.metrics.wer import WER
    wer_metric = WER()
    ref_tokens = [[w for w in r.lower().split()] for r in refs]
    hyp_tokens = [[w for w in h.lower().split()] for h in hyps]
    wer = wer_metric(hyp_tokens, ref_tokens)
    log(f"\n{'=' * 50}")
    log(f"Evaluation Results")
    log(f"{'=' * 50}")
    log(f"Samples : {len(refs)}")
    log(f"WER     : {wer:.2%}")

    # Show a few examples
    log(f"\n--- Sample transcriptions ---")
    for i in range(min(5, len(refs))):
        log(f"  Ref : {refs[i]}")
        log(f"  Hyp : {hyps[i]}")
        log()


# ---------------------------------------------------------------------------
# Extra  -  Apply speech hints to existing manifests
# ---------------------------------------------------------------------------
def apply_speechhints_to_manifests():
    """Post-process already-built manifests by normalizing the text field
    with speech-hint grammars.

    Reads train_manifest.json and test_manifest.json from custom_asr_data/,
    rewrites them in-place with normalized text.
    """
    if not SPEECH_HINT_AVAILABLE:
        log("speech_hint library not installed; installing it now ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "speech-hints"],
            check=True,
        )
        from speech_hint import apply_hint  # type: ignore
        SPEECH_HINT_AVAILABLE = True

    for manifest_name in ["train_manifest.json", "test_manifest.json"]:
        manifest_path = os.path.join(CUSTOM_DATA_DIR, manifest_name)
        if not os.path.exists(manifest_path):
            warn(f"{manifest_name} not found; skipping.")
            continue

        entries = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entry["text"] = normalize_with_speech_hints(entry["text"])
                    entries.append(entry)

        with open(manifest_path, "w") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")

        log(f"Normalized {len(entries)} entries in {manifest_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Nemotron 3.5 ASR Streaming on custom dataset "
            "with optional speech-hint normalization"
        )
    )
    parser.add_argument("--convert-only", action="store_true",
                        help="Only convert audio files (step 1)")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Convert audio + build manifests (steps 1-2)")
    parser.add_argument("--train-only", action="store_true",
                        help="Only run fine-tuning (step 3)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Only evaluate trained model (step 4)")
    parser.add_argument("--apply-speechhints", action="store_true",
                        help="Post-process manifests with speech-hint grammars")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate (default: 0.1, per notebook)")
    args = parser.parse_args()

    if args.apply_speechhints:
        apply_speechhints_to_manifests()
        return

    if not any([args.convert_only, args.manifest_only,
                args.train_only, args.evaluate]):
        # ---- Full pipeline ----
        log("=" * 70)
        log("Nemotron 3.5 ASR Streaming — Full Fine-Tuning Pipeline")
        log("=" * 70)

        log("\n>>> Step 1: Converting audio ...")
        convert_audio()

        log("\n>>> Step 2: Building manifests ...")
        train_manifest, test_manifest = build_manifests()

        log("\n>>> Step 3: Fine-tuning model ...")
        run_training(train_manifest, test_manifest, epochs=args.epochs, lr=args.lr)

        log("\n>>> Step 4: Evaluating model ...")
        run_evaluation(test_manifest)

        log("\n" + "=" * 70)
        log("Pipeline complete!")
        log("=" * 70)
        return

    if args.convert_only:
        convert_audio()
        return

    if args.manifest_only:
        convert_audio()
        build_manifests()
        return

    if args.train_only:
        train_manifest = os.path.join(CUSTOM_DATA_DIR, "train_manifest.json")
        test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")
        for m in [train_manifest, test_manifest]:
            if not os.path.exists(m):
                print(f"[ERROR] {m} not found. Run --manifest-only first.")
                sys.exit(1)
        run_training(train_manifest, test_manifest, epochs=args.epochs, lr=args.lr)
        return

    if args.evaluate:
        test_manifest = os.path.join(CUSTOM_DATA_DIR, "test_manifest.json")
        if not os.path.exists(test_manifest):
            print("[ERROR] test_manifest.json not found. Run training first.")
            sys.exit(1)
        run_evaluation(test_manifest)
        return


if __name__ == "__main__":
    main()
