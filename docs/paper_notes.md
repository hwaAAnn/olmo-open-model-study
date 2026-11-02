# Paper Notes: OLMo — Accelerating the Science of Language Models

- **Authors**: Groeneveld et al. (Allen Institute for AI)
- **Venue**: ACL 2024 / [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)
- **Models**: OLMo-1B, OLMo-7B

## Resources
- Models: [OLMo-7B](https://huggingface.co/allenai/OLMo-7B), [OLMo-1B](https://huggingface.co/allenai/OLMo-1B), [OLMo-7B-SFT](https://huggingface.co/allenai/OLMo-7B-SFT), [OLMo-7B-Instruct](https://huggingface.co/allenai/OLMo-7B-Instruct)
- Data / code: [Dolma](https://huggingface.co/datasets/allenai/dolma), [Open-Instruct](https://github.com/allenai/open-instruct), [TÜLU v2 SFT mixture](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture)

## Problem
Closed language models make it hard to study their strengths, weaknesses, biases, and risks scientifically.

## Key idea
Release the whole pipeline — training data (Dolma, ~3T tokens), code, intermediate checkpoints, and evaluation — so that every stage can be inspected, reproduced, and extended. Post-training follows SFT → DPO.

## Post-training recap
**SFT** — supervised fine-tuning on (instruction, output) pairs.

```json
{"instruction": "How much parental leave can I take?", "output": "Up to one year. Please check your company's policy."}
```

**DPO** — direct preference optimization on (chosen, rejected) pairs for the same instruction.

```json
{
  "instruction": "How much parental leave can I take?",
  "chosen": "Both parents can take up to one year. Please check your company's policy for details.",
  "rejected": "One year. Ask HR yourself."
}
```

## Limitations relevant to this study
- English-centric
- Bias and hallucination remain
- Scale gap: the paper uses 7B, this study uses 1B
- Full fine-tuning replaced by LoRA (~0.07% trainable parameters)
