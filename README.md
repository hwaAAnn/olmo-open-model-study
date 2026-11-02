# Open Weights as a Research Instrument: A Small-Scale Study with OLMo 2

> A mini-project examining whether the openness claims of *OLMo: Accelerating the Science of Language Models* (Groeneveld et al., ACL 2024) translate into concrete, reproducible experiments at small scale (1B parameters, single GPU).

## Motivation

OLMo argues that releasing weights, data, code, and intermediate checkpoints enables scientific study of language models that closed models do not allow. This project takes that claim as a hypothesis and tests it with four small experiments, each probing one capability that openness is supposed to provide:

| # | Question | Capability tested |
|---|---|---|
| 1 | Can we enumerate and access training-time snapshots? | Checkpoint accessibility |
| 2 | How do outputs change across Base → SFT → Instruct? | Stage-wise behavioral analysis |
| 3 | Can the public SFT recipe be re-run with public data? | Reproducibility |
| 4 | Can the open model be adapted to a new domain with SFT + DPO? | Extensibility |

## Setup

- **Models**: `allenai/OLMo-2-0425-1B`, `-1B-SFT`, `-1B-Instruct`
- **Adaptation**: LoRA (r=8, α=16, dropout 0.05, targets `q_proj`/`v_proj`), ≈0.07% of parameters trainable
- **SFT data (Exp. 3)**: first 10k examples of [TÜLU v2 SFT mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture), 1 epoch, lr 2e-4 (cosine), effective batch size 16
- **Domain data (Exp. 4)**: 500 English instruction–response examples on workplace communication support for workers with developmental disabilities ([data/domain_sft_500_en.jsonl](data/domain_sft_500_en.jsonl))
- **Hardware**: single RTX 4090 for training; inference fits on smaller GPUs
- **Software**: Python 3.11, PyTorch ≥ 2.6, transformers, peft, trl 1.7.1, accelerate, datasets

## Experiments and Findings

### Exp. 1 — Checkpoint accessibility
Enumerated the publicly released OLMo training checkpoints: **268** snapshots, whose names encode stage, step, and tokens seen (e.g. `stage2-ingredient3-step23852-tokens51B`).

This makes it possible, in principle, to locate when a behavior emerges, compare before/after data-mixture changes, and resume training from an intermediate state. This experiment only verifies *access*; no checkpoint-level analysis was performed.

### Exp. 2 — Behavior across post-training stages
The same English prompts (knowledge QA, commonsense, summarization, etc.; [data/sample_cases.jsonl](data/sample_cases.jsonl)) were given to Base, SFT, and Instruct models. Outputs were compared qualitatively (response produced, question repetition, intent recognition, directness, specificity, format fit) and by response length ([src/evaluate.py](src/evaluate.py)).

| Stage | Observed behavior |
|---|---|
| Base | Often empty output or repetition of the prompt |
| SFT | Recognizes intent; short, direct answers |
| Instruct | Well-formatted, more structured explanations |

*Caveat*: this is an observational comparison of released weights, not a causal ablation. Differences between stages may also reflect data and hyperparameter changes beyond the named stage, and the prompt set is small.

### Exp. 3 — Re-running the SFT recipe with LoRA
Applied LoRA SFT to the Base model using a subset of TÜLU v2 and compared it against the official OLMo 2 SFT model on a shared prompt set ([results/olmo2_sft_comparison.csv](results/olmo2_sft_comparison.csv)).

Observed after tuning: less prompt repetition, sentence-form answers, and basic instruction-following behavior emerging from the Base model.

*Caveat*: this does **not** reproduce the official SFT's performance. It shows that the pipeline (public weights + public data + public tooling) can be executed end-to-end at small scale. No quantitative benchmark was run, and outputs from the tuned model still show format artifacts (e.g. continuing past the answer into a new `### Instruction:` block).

### Exp. 4 — Domain adaptation with SFT + DPO
Fine-tuned the model on workplace-communication guidance data, then applied DPO with preference pairs (`chosen` / `rejected`) encoding the following preferences:

- acknowledge what the user did well first
- avoid declaring the user's phrasing simply "wrong"
- use short, easy sentences
- offer several alternative phrasings instead of a single "correct" one
- avoid blaming or discouraging tone

Models compared on shared prompts: Base, TÜLU-SFT (ours), Domain-SFT, Domain-DPO, Official SFT ([results/olmo2_domain_comparison.csv](results/olmo2_domain_comparison.csv)). Qualitatively, domain-tuned models produced more domain-specific responses than Base, and preference tuning shifted response style in the intended direction.

*Caveat*: evaluation is qualitative on a small prompt set with no human or LLM-judge scoring; the domain data is small (500 examples) and was converted from guideline-style text, so overfitting and style leakage are likely. The DPO preference pairs and domain-training code are not included in this repository; the notebook loads the resulting adapters.

## Conclusion

At the 1B scale, openness did make all four experiments possible: stage-wise comparison (Exp. 2), recipe re-execution (Exp. 3), and domain adaptation (Exp. 4) each depended directly on public weights, data, and training code. The results are preliminary and qualitative; they demonstrate feasibility, not performance parity or causal attribution.

## Limitations and Future Work

- Qualitative evaluation only → add rubric-based scoring, LLM-as-judge with human spot-checks, and standard benchmarks
- 1B model, 10k-example subset → test scaling with 7B and larger data fractions
- Checkpoint use limited to enumeration → track emergence of instruction-following across intermediate checkpoints
- Domain data is English and converted from guidelines → collect natural-language data, evaluate in Korean (tokenizer efficiency is a known concern)
- Safety-relevant domains require expert review before any real-world use

## Repository Structure

```
├── README.md
├── notebooks/demo.ipynb        # end-to-end walkthrough of all four experiments
├── src/
│   ├── baseline.py             # Base / SFT / Instruct inference helpers
│   └── evaluate.py             # response-length and stage-wise summaries
├── data/
│   ├── sample_cases.jsonl      # prompts for Exp. 2
│   └── domain_sft_500_en.jsonl # domain data for Exp. 4
├── results/                    # raw model outputs per experiment (CSV)
└── docs/paper_notes.md         # notes on the OLMo paper
```

## Reproducing

```bash
pip install transformers peft trl==1.7.1 accelerate datasets bitsandbytes huggingface_hub pandas
jupyter notebook notebooks/demo.ipynb
```

Exp. 3 training needs a GPU of roughly RTX 4090 class; Exps. 1, 2, and inference for Exp. 4 run on smaller GPUs.

## Reference

Groeneveld, D., et al. *OLMo: Accelerating the Science of Language Models.* ACL 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)

## Note

This repository is a personal re-organization of a team mini-project. Its original structure and experiments were developed collaboratively.
