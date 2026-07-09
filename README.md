## AI Paper Reviewer Agent

This project is being updated to avoid reusing old generated template data and
old PeerRead review-dataset examples.

### Dataset Choice

Do not use the previous generated template dataset.

Do not use the previous PeerRead review datasets, including:

- `iclr_2017`
- `acl_2017`

The current `prepare_llama_finetune.py` script is disabled because it generates
data from those review datasets.

### Workflow

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add only approved fine-tuning data to:

- `finetune_data/train.jsonl`
- `finetune_data/validation.jsonl`
- `finetune_data/test.jsonl`

To build filtered data from approved Hugging Face OpenReview sources:

```bash
python3 prepare_hf_openreview_finetune.py
```

The converter uses real `review` fields, rejects synthetic/prompt-like text, and
does not use `thinking_trace` or chat-formatted fields.

3. Validate the generated data before training:

```bash
python3 validate_finetune_data.py
```

This fails if the files contain old template phrases or old review-dataset
markers such as `iclr_2017`, `acl_2017`, `### Review:`, or the previous
scientific peer-review instruction.

4. Fine-tune a base model with LoRA:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python train_model.py --model-name distilgpt2 --max-train-rows 100 --max-validation-rows 20
```

Remove the row limits for a full run. Use a stronger local/Hugging Face model
with a longer context window if your machine can support it.

5. Test local Llama 3.2 on approved held-out examples:

```bash
python3 test_model.py --model llama3.2 --limit 3
```

To test the Hugging Face base model with the trained LoRA adapter:

```bash
.venv/bin/python test_lora_model.py --base-model distilgpt2 --adapter-dir models/peer_review_lora --limit 3 --local-files-only
```

If you create a custom Ollama model, pass its name:

```bash
python3 test_model.py --model your-custom-model-name
```

### Remaining Scripts

- `prepare_llama_finetune.py`: disabled because it creates forbidden PeerRead review-dataset examples.
- `prepare_hf_openreview_finetune.py`: creates filtered JSONL splits from approved Hugging Face OpenReview review datasets.
- `train_model.py`: LoRA fine-tunes a causal language model on the filtered JSONL splits.
- `test_lora_model.py`: tests a Hugging Face base model plus trained LoRA adapter.
- `test_model.py`: tests local Ollama/Llama models on held-out examples.
- `pdf_to_llama.py`: sends a local PDF to Llama for a preliminary review.
- `rag_novelty_review.py`: retrieves related arXiv papers and asks Llama for novelty comparison.
