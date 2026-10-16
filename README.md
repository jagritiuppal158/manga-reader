# manga-reader
transcribes 3-page manga sequences with speaker labels.
# Manga Task: story text and speakers from 3-page manga sequences

## The task
Given three consecutive English manga pages, output the story text in reading order and who says each line. Speaker labels like `character1` are fine, but the same character must keep the same label on all three pages. Sound effects, titles, recap boxes and similar text are left out, and pages without story text stay empty.

## Data and how I split it
There are 80 labelled development sequences and 15 test sequences, three pages each. I split the 80 development sequences **by sequence** (seed 42) into 64 for training and 16 for validation, so pages from one sequence are never on both sides. The split is saved in `results/split.json`.

The test labels are not released, so I have **no test score**. Every number below is on my 16 validation sequences, scored with the official `score.py`.

## Approach
- **Model:** Qwen2-VL-7B-Instruct (open weights), loaded in 4-bit with Unsloth so it fits on a free Colab T4 GPU.
- **Fine-tuning:** LoRA only (rank 16, alpha 16, no dropout, language layers only, about 40.4M trainable parameters). The 7B base weights stay frozen.
- **Output format:** the model writes JSON, which a small parser converts to the scorer's format. The parser also repairs JSON that was cut off.
- **Post-processing:** repeated lines on a page are removed (the model sometimes loops).
- No hosted AI APIs were used, and no prediction was edited by hand.

## Experiments
Models A, B and C read one page at a time. Model D reads all three pages in one call.

| Experiment | What changed | text_order | joint_f1 | balanced_joint_f1 | speaker_acc (matched) | coverage |
|---|---|---|---|---|---|---|
| A | frozen model, basic prompt | 0.5506 | 0.3246 | 0.1795 | 0.4960 | 0.6571 |
| B | stricter prompt | 0.5674 | 0.3587 | 0.2202 | 0.5586 | 0.6789 |
| C | B's prompt + LoRA (page by page) | 0.6047 | 0.4259 | 0.2467 | 0.5972 | 0.7275 |
| C + dedupe | C, repeated lines removed | 0.6130 | 0.4305 | 0.2504 | 0.5956 | 0.7149 |
| D | LoRA on 3 pages at once, labels shared across pages | 0.7517 | 0.4978 | 0.2881 | 0.6194 | 0.8250 |
| **D + dedupe (final)** | D, repeated lines removed | **0.7607** | **0.5157** | **0.3142** | **0.6214** | **0.8214** |

**A to B.** B's prompt excludes sound effects and recap boxes, uses right-to-left order, and limits `NARRATION` to caption boxes. It helped a little. I wrote that prompt after looking at validation examples, so B's gain is slightly optimistic.

**B to C.** LoRA beat the frozen model with the same prompt on joint F1 in 12 of 16 sequences. On balanced F1 it was 8 to 8, so I don't claim that one improved.

**Why I built D.** Page-by-page labelling cannot carry a character's label across pages. If the text and speakers were perfect but labels were renamed on every page, the best possible joint F1 would be 0.743 and balanced F1 0.590 on the validation set. Model C could never get above that.

**C to D.** D scored higher on every metric, and its joint F1 was higher on 13 of 16 sequences (a one-sided sign test gives about 0.011, so this is suggestive, not proof). The biggest change is in text order and coverage. Speaker accuracy only rose from 0.60 to 0.62, and balanced F1 is still 0.31, far below the 0.59 ceiling, so I would not say cross-page consistency is solved. D was worse than C on 3 of 16 sequences, and one sequence scored near zero for both models (I did not investigate it). In D's validation run, 15 outputs parsed cleanly, 1 needed its cut-off JSON repaired, and none failed. [Add the parse status counts of the test run here.]

**D changed five things at once** compared with C: three pages per call, labels numbered across the sequence, images resized the same way at training and test time (longest side 768 px), loss on the answer only, and 4 epochs instead of 2. I did not run ablations, so I cannot say which change caused the gain.


## Limitations
- Only 16 validation sequences, one training run and one seed, so small differences are noise.
- I did not tune the settings (rank, learning rate, epochs). They are common defaults.

## Reproducing the results
data files(`development-*.zip`, `test-*.zip`, `sequences.json`, `sample_submission.jsonl`, `score.py`) in one folder. In the notebooks this is the `SRC` path (default `My Drive/manga_task`). Outputs go to `SRC/outputs`. Versions are in `results/requirements_d.txt` and `results/run_info_d.json` (Colab, Tesla T4).

**Without training (final results).** Open `notebooks/02_sequence_level_model_D.ipynb`. Run the install cell, the setup cells, then **load the saved adapter** (`lora_adapter_d`, download link below) in place of the training cells. Then run the validation experiment and the test prediction cell. Greedy decoding is used (`max_new_tokens=2048`), so the test text should be the same as my submitted file, up to small GPU differences

**With training.** Run the same notebook from the top, including the LoRA and training cells (batch size 1, gradient accumulation 4, 4 epochs, 64 steps, learning rate 1e-4, fp16, seed 3407). A retrained adapter can differ slightly from mine. Training peaked at 6.7 GB of GPU memory.

**Earlier experiments.** `notebooks/01_page_level_experiments_A_to_C.ipynb` contains experiments A, B and C and the analyses behind them. I tidied both notebooks after running them (removed duplicate cells, merged and renamed helpers) without changing the logic.

## Files
- `notebooks/`: the two notebooks above
- `results/`: `experiments.csv` (all runs), `split.json`, `prompts/`, score files, `per_sequence_joint_f1.csv`, run info
- `predictions/final_predictions_d.jsonl`: final test predictions (15 lines). `final_predictions_v1.jsonl` is the earlier page-by-page model's output, kept as a backup
- Adapter (final model):https://drive.google.com/drive/folders/15AUVmKL_dMrj-4CaMC1YZPF4-DvGLkjw?usp=sharing
-  It is not committed to the repo because of file size limits.

## Credits
Qwen2-VL-7B-Instruct, Unsloth, TRL, PEFT, Hugging Face Transformers, bitsandbytes, and the organisers' `score.py`.

## AI assistance
I used an AI assistant (Claude,chatgpt) for planning the experiments, debugging, and drafting code and a first README draft. I ran all experiments myself and rewrote the README in my own words.

