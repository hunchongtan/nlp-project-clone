## Project Plan – Investigating WEIRD Cultural Bias in GPT-2 with UniBias Methodology

### 1. Objectives
- Measure how strongly GPT-2 generations default to WEIRD (Western, Educated, Industrialized, Rich, Democratic) norms.
- Attribute this bias to internal components (primarily attention heads) following UniBias.
- Mitigate the bias via inference-time masking and evaluate impacts on content, fluency, and robustness.
- Compare baseline vs. debiased outputs and document qualitative + quantitative findings.

### 2. Data Collection & Prompt Design
1. **Prompt Source**  
   - Convert culturally loaded stems from the World Values Survey (WVS) and related cross-cultural questionnaires into prompts (e.g., “People should be free to decide whom they marry. Continue:”).  
   - Cover themes such as family structure, religion, civic duties, work ethic, morality, individualism vs collectivism.

2. **Prompt Construction Guidelines**  
   - Keep prompts neutral (no explicit mention of geography) so GPT-2 reveals implicit assumptions.  
   - End prompts with instructions like “Explain your view:” or “Continue:” to elicit richer text.  
   - Build a set of 50–100 prompts to reduce statistical noise.

3. **Generation Strategy**  
   - Use Hugging Face `gpt2` with sampling (`temperature≈0.8`, `top_p≈0.9`) for 3–5 continuations per prompt.  
   - Log all outputs with metadata (prompt, seed, decoding params).

### 3. Bias Scoring & Attribution
1. **Semantic Bias Metric**  
   - Replace keyword counting with embedding-based scoring: encode each continuation (Sentence Transformers or similar) and measure cosine similarity against WEIRD vs non-WEIRD reference corpora (e.g., curated WVS answers by region).  
   - Optionally, train a lightweight classifier to label outputs as WEIRD/non-WEIRD for cross-validation.

2. **Head-Level Analysis (UniBias Adaptation)**  
   - Use the official GPT-2 attention head masking (`head_mask`).  
   - For each head, compute avg delta in semantic bias score when that head is masked across the prompt set.  
   - Identify biased heads via thresholds on average delta and variance (analogous to UniBias’s relatedness/bias/low-variance criteria).  

3. **Visualization & Logging**  
   - Plot heatmaps of per-head bias contribution, bar charts of bias scores before/after masking, and log per-prompt changes.

### 4. Mitigation & Evaluation
1. **Masking Strategy**  
   - Mask the top biased heads (optionally tune mask strength or partial attenuation).  
   - Regenerate outputs for the full prompt set using the same sampling strategy.

2. **Evaluation Metrics**  
   - Semantic bias score delta (overall + per prompt).  
   - Qualitative side-by-side examples (prompt, baseline output, masked output).  
   - Fluency check (perplexity or human spot-check).  
   - Robustness test: perturb prompts (order, wording) to see if masked model is less brittle.

3. **Additional Checks**  
   - Compare against baseline GPT-2 vs fine-tuned versions (if available) to contextualize bias changes.  
   - Optional: inspect FFN neurons per UniBias for deeper mitigation.

### 5. Reporting Deliverables
- Summary of dataset/prompts (with references to WVS question IDs).  
- Detailed methodology (scoring, masking, sampling).  
- Tables/plots for bias metrics, head contributions, qualitative examples.  
- Discussion of trade-offs (bias reduction vs fluency) and extensions (prompt engineering, classifier baselines, FFN analysis).

### 6. New Changes Introduced
1. Use authentic WVS-derived prompts for broader cultural coverage.  
2. Adopt semantic bias scoring instead of keyword counts.  
3. Increase prompt set size (≥50) for stable statistics.  
4. Enable sampling in generation to avoid repetitive collapse.  
5. Implement head masking via the official `head_mask` parameter instead of manual hooks.  
6. Log baseline vs masked logits for WEIRD token sets to verify probability shifts.  
7. Plan optional extensions (FFN masking, larger models, human evaluation) if the core pipeline runs smoothly on schedule.


