# HumDial-EIBench: A Multi-Turn Emotional Intelligence Benchmark for Audio Language Models

> **Official repository of "HumDial-EIBench: A Multi-Turn Emotional Intelligence Benchmark for Audio Language Models".**

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Arxiv%20Link-blue)](#) 
[![Dataset](https://img.shields.io/badge/Dataset-HumDial--EIBench-green)](#) 

</div>

<div align="center">
  <img src="assets/humdial-bench.png" alt="HumDial-EIBench Pipeline" width="90%">
  <p><em>Figure 1: Data construction and task overview of HumDial-EIBench. Left: The three-stage construction pipeline encompassing dialogue script design, human performance and quality control, and objective rewriting. Right: Representative examples of the four evaluation tasks.</em></p>
</div>

This is the official repository for the paper **"HumDial-EIBench: A Multi-Turn Emotional Intelligence Benchmark for Audio Language Models"**.

**Dataset and evaluation codes can be accessed anonymously at:** [https://anonymous.4open.science/r/HumDial-EIBench/](#)

## 💡 Abstract
Comprehensive assessment of emotional intelligence in end-to-end spoken dialogue models is important. Traditional evaluation of open-ended generation relies on subjective scoring, which hinders stable quantification of context understanding capabilities, such as tracking emotional changes or inferring underlying causes. To address this issue, this paper proposes HumDial-EIBench, an objective benchmark to evaluate the emotional intelligence of audio language models. Based on the human-recorded test set from the ICASSP 2026 HumDial Challenge, the proposed benchmark transforms emotional trajectory detection and emotional causal reasoning tasks into multiple-choice formats with adversarial distractors, eliminating subjective scoring variants. Furthermore, HumDial-EIBench retains the original empathetic response generation task and introduces an acoustic-semantic conflict recognition task to assess cross-modal perception and empathetic expression. Extensive evaluation results across eight audio language models validate the effectiveness of the proposed benchmark in diagnosing complex interaction capabilities. The results indicate significant room for improvement in multi-turn dialogues and cross-modal alignment among current models.

## 🚀 News
* **[2026.03]** Paper submitted and repository initialized.
* **[2026.03]** HumDial-EIBench dataset uploaded.

## 📊 Dataset: HumDial-EIBench

To address the limitations of existing speech benchmarks, which mostly rely on synthesized speech and lack continuous multi-turn emotional evolution, this paper constructs HumDial-EIBench, a real-recorded, bilingual multi-turn dataset.

The dataset includes 1,077 samples across four tasks. It uses scenarios from the ICASSP 2026 HumDial Challenge, re-annotated for objective evaluation to separate cognitive reasoning from expressive generation.

| Task | Format | CN / EN | Level | Primary Metric |
| :--- | :---: | :---: | :---: | :---: |
| **Task 1: Emotional Trajectory** | Multiple Choice | 150 / 150 | Multi-turn | Accuracy |
| **Task 2: Causal Reasoning** | Multiple Choice | 134 / 149 | Multi-turn | Accuracy |
| **Task 3: Empathetic Generation** | Open Generation | 144 / 150 | Multi-turn | LLM + Human |
| **Task 4: Acoustic-Semantic Conflict** | Multiple Choice | 100 / 100 | Single-turn | Accuracy |
| **Total** | | **528 / 549** | | |

### Dataset Access & Explanation

#### 1. Task 1 & 2 (Objective Multi-Turn Reasoning)
These tasks transform open-ended emotional trajectory tracking and causal reasoning into multiple-choice formats. Adversarial distractors are designed to ensure models differentiate factual developments from genuine emotional drivers and temporal state shifts.
* 🔒 **Availability:** The dataset is currently undergoing anonymous peer review and will be released publicly upon paper acceptance.

#### 2. Task 3 (Empathetic Generation)
Retained from the original ICASSP 2026 challenge, this task requires audio language models to generate empathetic responses. The evaluation separates text-level cognitive empathy (D1) and acoustic-level expressive empathy (D2/D3).

#### 3. Task 4 (Acoustic-Semantic Conflict)
A newly introduced single-turn task. Models must identify true emotional states when the literal semantic content contradicts the acoustic emotional tone (e.g., sarcasm). This task specifically evaluates text-dominance bias.

## 💻 Usage

### Evaluation Overview

| Task | Input | Evaluation |
| :--- | :--- | :---: |
| Task 1 – Emotional Trajectory | Multi-turn audio + question + 4 options | Accuracy |
| Task 2 – Causal Reasoning | Multi-turn audio + question + 4 options | Accuracy |
| Task 3 – Empathetic Generation | Multi-turn audio + question | LLM judge (D1) + Human (D2/D3) |
| Task 4 – Acoustic-Semantic Conflict | Single-turn audio + question + 4 options | Accuracy |

---

### Tasks 1, 2 & 4 — Multiple-Choice Evaluation

For Tasks 1, 2, and 4, the model receives sequential audio turns followed by a four-option multiple-choice question. The evaluation script extracts the selected option from the model output and computes accuracy against the ground truth label.

---

### Task 3 — Empathetic Generation Evaluation

- **D1 (Textual Empathy & Insight):** Automatically scored by an LLM judge (Qwen3-Omni or Gemini-2.5-Pro) on a 1–5 scale. The judge assesses whether the response validates the user's emotional state and provides deeper insight, rather than restating the emotion superficially.
- **D2 (Vocal Empathy & Congruence):** Assessed by human evaluators. Measures whether the generated speech exhibits appropriate tonal warmth and paralinguistic alignment with the user's emotional state.
- **D3 (Audio Quality & Naturalness):** Assessed by human evaluators. Measures technical clarity, fluency, and human-like naturalness of the synthesized audio.

**Input format.** The input JSONL file should contain one dialogue object per line, structured as follows:

```jsonc
{
  "dialogue_id": "sample_001",
  "turns": [
    {
      "input_emotion": "sad",
      "input_text": "I've been feeling really overwhelmed lately...",
      "response_text": "It sounds like you're carrying a lot right now.",
      "response_audio": "outputs/sample_001_turn1.wav"
    }
    // ... additional turns
  ]
}
```

The script automatically identifies the target evaluation turn (the second non-neutral turn) and builds the conversation context from preceding turns.

**Step 1: Run automated D1 scoring with the local Qwen3-Omni judge.**

```bash
python eval/eval_task3.py \
  --model Qwen3-Omni-30B-A3B-Instruct \
  --input_file results/task3_outputs.jsonl \
  --output_file results/task3_scores.jsonl
```

The script loads Qwen3-Omni locally via vLLM, scores each response across D1, D2, and D3 dimensions, and writes per-sample results to `--output_file`. A summary of average scores is appended at the end of the file and printed to stdout.

> **Note:** The script requires a GPU environment with vLLM installed. Update the `model_path` variable at line 231 of `eval/eval_task3.py` to point to your local Qwen3-Omni checkpoint before running.

**Step 2: Human evaluation for D2 and D3.**

Human evaluation instructions and the full scoring rubric (1–5 scale for vocal empathy and audio naturalness) are embedded in `eval/eval_task3.py`. Evaluators listen to the `response_audio` files and rate each sample independently under a blind protocol.

## 📖 Citation

If you use this benchmark, please consider citing the paper:

```bibtex
@article{HumDialEIBench2026,
  title={HumDial-EIBench: A Multi-Turn Emotional Intelligence Benchmark for Audio Language Models},
  author={Anonymous Authors},
  journal={Anonymous submission},
  year={2026}
}
```
