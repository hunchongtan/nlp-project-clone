# Adapting UniBias Methodology for Subgroup Bias Detection in GPT-2

## Executive Summary

This report documents the adaptation of the UniBias methodology (Zhou et al., 2024) to detect and mitigate representational bias across identifiable subgroups in GPT-2 language model predictions. We investigate bias in stance classification across different pillars (military, civil, economic, social, psychological, digital, others) using the Total Defence Meme dataset. Our implementation successfully identified biased attention heads and demonstrated measurable bias reduction across subgroups, with varying effectiveness depending on the pillar category.

---

## 1. Introduction to UniBias

### 1.1 What is UniBias?

**UniBias** (Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation) is a method developed by Zhou et al. (2024) for detecting and reducing bias in Large Language Models (LLMs). Unlike traditional approaches that treat models as black boxes, UniBias operates by directly manipulating the model's internal components.

### 1.2 Core Principle

UniBias is based on the hypothesis that bias in LLMs is encoded in specific internal components:
- **Attention Heads**: Components that determine which parts of the input the model focuses on
- **FFN (Feed-Forward Network) Neurons**: Components that process and transform information

By identifying and manipulating these biased components, UniBias can reduce bias without retraining the model.

### 1.3 Original UniBias Methodology

The original UniBias method was designed for **tasks with discrete outputs** (e.g., sentiment analysis, natural language inference, question answering, text classification) and uses a three-criterion approach to identify biased components. According to the original implementation, UniBias supports:
- Sentiment Analysis (SST-2, SST-5, CR, MR)
- Natural Language Inference (MNLI, RTE)
- Question Answering (COPA, ARC, MMLU)
- Text Classification (AG News, TREC, WiC)

1. **Bias Criterion**: The component shows disproportionate preference for certain labels
2. **Relatedness Criterion**: The component is meaningfully related to the task
3. **Low-Variance Criterion**: The component exhibits consistent behavior across samples

The method works by:
1. Analyzing model outputs on validation data
2. Computing how each attention head or FFN neuron contributes to biased predictions
3. Identifying components that consistently favor certain labels
4. Masking or adjusting these components during inference
5. Evaluating the reduction in bias

### 1.4 Key Innovation

UniBias's key innovation is that it doesn't require:
- Model retraining
- Additional labeled data
- External debiasing datasets

Instead, it works **inference-time** by manipulating the model's internal representations.

---

## 2. Adapting UniBias for Subgroup Bias Detection

### 2.1 Challenge: Detecting Bias Across Identifiable Subgroups

The original UniBias was designed for tasks where bias is measured across **all samples** without considering subgroup membership. Our adaptation focuses on **representational bias**—detecting whether the model shows different prediction patterns for different subgroups.

**Representational Bias**: When a model's predictions systematically differ across identifiable subgroups (e.g., demographics, categories, regions), even when controlling for other factors.

**Our Task**: Detect if GPT-2 shows bias in stance classification (against, neutral, supportive) that varies across different pillars (military, civil, economic, social, psychological, digital, others).

### 2.2 Our Adaptation Strategy

We adapted UniBias for subgroup bias detection in three key ways:

#### 2.2.1 Dataset: Total Defence Meme Dataset

**Dataset**: SEACrowd/total_defense_meme
- **Schema**: `seacrowd_imtext` (image-text multimodal schema)
- **Text Source**: OCR-extracted text from memes about Singapore's Total Defence policy
- **Subgroups (Pillars)**: military, civil, economic, social, psychological, digital, others
- **Task**: Stance classification (against, neutral, supportive)
- **Total Samples**: 2,512 samples with text and labels

**Dataset Distribution**:
- **Stances**: neutral (1,137), against (1,079), supportive (296)
- **Pillars**: others (946), military (724), psychological (377), economic (209), civil (158), social (59), digital (39)

**Why This Dataset**:
- Contains naturally occurring text with cultural/political content
- Has clear subgroup labels (pillars) for representational bias analysis
- Includes stance labels for classification task (compatible with UniBias)
- Culturally relevant (Singaporean context)

#### 2.2.2 Task Design: Stance Classification

**Challenge**: Need a discrete-output task compatible with UniBias methodology.

**Solution**: Frame as stance classification task with few-shot prompting.

**Prompt Format**:
```
Classify the stance toward Total Defence:
Text: Total Defence is essential for Singapore's security
Stance: supportive

Text: I disagree with Total Defence policies
Stance: against

Text: Total Defence has both pros and cons
Stance: neutral

Text: [meme text]
Stance:
```

**Answer Tokens**: 
- `against`: Token ID 32826
- `neutral`: Token ID 29797
- `supportive`: Token IDs [11284, 425]

This format allows UniBias to measure bias by examining logit differences between stance tokens, exactly as in the original methodology.

#### 2.2.3 Bias Measurement: Subgroup Analysis

**Challenge**: Need to measure if model predictions differ across subgroups (pillars).

**Our Method**:
1. **Baseline Evaluation**: For each pillar, compute:
   - Average logits for each stance (against, neutral, supportive)
   - Prediction distribution (which stance is most favored)
   - Bias magnitude: `max_logit - min_logit` (difference between highest and lowest stance logits)

2. **Head Identification**: For each attention head:
   - Mask that head (set to 0)
   - Recompute bias scores across all prompts
   - Calculate delta: `baseline_bias - masked_bias`
   - Identify heads with largest positive deltas (masking reduces bias)

3. **Subgroup Comparison**: After masking biased heads:
   - Re-evaluate predictions by pillar
   - Compare bias magnitude before/after for each pillar
   - Measure bias reduction: `baseline_bias - masked_bias`

**Key Innovation**: We analyze bias **within each subgroup** and **across subgroups**, revealing if certain pillars show more bias or different bias patterns.

---

## 3. Methodology Details

### 3.1 Experimental Setup

**Model**: GPT-2 (124M parameters) from Hugging Face
- **Architecture**: 12 layers, 12 attention heads per layer (144 total heads)
- **Attention Mode**: `attn_implementation="eager"` (required for `head_mask` support)
- **Device**: CUDA (GPU) when available, CPU otherwise

**Dataset**:
- **Source**: SEACrowd/total_defense_meme
- **Schema**: `seacrowd_imtext`
- **Samples Used**: 50 samples (subset for faster processing)
- **Distribution**: Representative sample across all pillars

**Evaluation**:
- **Task**: Stance classification (3-way: against, neutral, supportive)
- **Metric**: Logit-based bias measurement
- **Subgroup Analysis**: Bias measured separately for each pillar

### 3.2 Implementation Steps

#### Step 1: Data Preparation
1. Load dataset using `seacrowd` library
2. Extract text from `texts` field (OCR text from memes)
3. Extract pillar labels from `metadata.pillar_stances.category`
4. Extract stance labels from `metadata.pillar_stances.stance`
5. Create few-shot prompts for stance classification

#### Step 2: Baseline Evaluation
1. For each prompt, compute logits for stance tokens (against, neutral, supportive)
2. Predict stance: highest logit wins
3. Group predictions by pillar
4. Calculate average logits per stance per pillar
5. Compute bias magnitude: `max_logit - min_logit` for each pillar

#### Step 3: Head Identification
1. Compute baseline bias score: standard deviation of average logits across stances
2. For each attention head (sampled: first 2 layers = 24 heads):
   - Create head mask (mask this specific head)
   - Recompute bias score with masked head
   - Calculate delta: `baseline_bias - masked_bias`
3. Select top 3 heads with largest positive deltas

#### Step 4: Bias Mitigation
1. Create head mask for top 3 biased heads
2. Re-evaluate all prompts with masked heads
3. Recompute predictions and bias scores by pillar
4. Compare before/after results

#### Step 5: Subgroup Analysis
1. For each pillar, compare:
   - Baseline vs masked prediction distribution
   - Baseline vs masked bias magnitude
   - Bias reduction: `baseline_bias - masked_bias`

---

## 4. Results

### 4.1 Baseline Predictions

**Overall Predictions** (50 samples):
- **Against**: 19 (38.0%)
- **Neutral**: 29 (58.0%)
- **Supportive**: 2 (4.0%)

**Average Logits by Stance**:
- **against**: -109.06
- **neutral**: -108.77 (highest, most favored)
- **supportive**: -109.88 (lowest, least favored)

**Baseline Bias Score**: 0.4694 (standard deviation of average logits)

**Key Observation**: Model shows strong preference for "neutral" stance, with "supportive" being least favored.

### 4.2 Subgroup Analysis: Predictions by Pillar

**Baseline Predictions by Pillar**:

| Pillar | Samples | Predictions | True Labels | Avg Logits | Favored Stance | Bias Magnitude |
|--------|---------|-------------|-------------|------------|----------------|----------------|
| **Civil** | 4 | neutral: 2, against: 2 | neutral: 2, against: 1, supportive: 1 | against: -111.74, neutral: -111.31, supportive: -112.72 | neutral | 1.4035 |
| **Digital** | 5 | neutral: 4, against: 1 | against: 5 | against: -108.34, neutral: -108.32, supportive: -109.68 | neutral | 1.3522 |
| **Economic** | 16 | against: 8, neutral: 6, supportive: 2 | neutral: 3, against: 11, supportive: 2 | against: -110.33, neutral: -110.53, supportive: -110.97 | against | 0.6489 |
| **Military** | 15 | neutral: 11, against: 4 | neutral: 9, against: 5, supportive: 1 | against: -107.59, neutral: -106.78, supportive: -108.40 | neutral | 1.6267 |
| **Others** | 3 | neutral: 3 | against: 2, neutral: 1 | against: -106.24, neutral: -105.64, supportive: -106.81 | neutral | 1.1681 |
| **Psychological** | 5 | against: 4, neutral: 1 | supportive: 2, against: 2, neutral: 1 | against: -110.49, neutral: -111.29, supportive: -111.29 | against | 0.7996 |
| **Social** | 2 | neutral: 2 | neutral: 1, supportive: 1 | against: -109.50, neutral: -107.33, supportive: -109.50 | neutral | 2.1715 |

**Key Findings**:
1. **Different pillars show different bias patterns**:
   - **Economic & Psychological**: Favor "against" stance
   - **All other pillars**: Favor "neutral" stance
   - **Social**: Highest bias magnitude (2.1715)

2. **Representational bias detected**: Model predictions vary systematically across pillars, indicating the model treats different subgroups differently.

3. **Accuracy varies by pillar**: 
   - **Digital**: 0% accuracy (all predicted neutral, but all true labels are against)
   - **Economic**: Better alignment (8 against predictions vs 11 true against labels)

### 4.3 Head Identification Results

**Top 10 Biased Attention Heads**:

| Rank | Layer | Head | Delta | Masked Bias |
|------|-------|------|-------|-------------|
| 1 | 0 | 5 | 0.158836 | 0.3105 |
| 2 | 0 | 11 | 0.038258 | 0.4311 |
| 3 | 1 | 2 | 0.035952 | 0.4334 |
| 4 | 1 | 10 | 0.029752 | 0.4396 |
| 5 | 1 | 4 | 0.012714 | 0.4567 |
| 6 | 1 | 3 | -0.005042 | 0.4744 |
| 7 | 1 | 0 | -0.015707 | 0.4851 |
| 8 | 1 | 9 | -0.025497 | 0.4949 |
| 9 | 0 | 7 | -0.028404 | 0.4978 |
| 10 | 1 | 11 | -0.040065 | 0.5094 |

**Selected for Masking**: Top 3 heads
- Layer 0, Head 5 (delta: 0.1588)
- Layer 0, Head 11 (delta: 0.0383)
- Layer 1, Head 2 (delta: 0.0360)

**Key Observations**:
- **Layer 0, Head 5** shows the strongest bias contribution (delta: 0.1588)
- **Early layers** (0-1) contain the most biased heads
- **Negative deltas** indicate some heads actually reduce bias when masked (heads 6-10)

### 4.4 After Masking Results

**Overall Predictions After Masking** (50 samples):
- **Against**: 18 (36.0%) ← decreased from 38%
- **Neutral**: 24 (48.0%) ← decreased from 58%
- **Supportive**: 8 (16.0%) ← **increased from 4%** (4x increase!)

**Key Finding**: Masking biased heads **significantly increased supportive predictions**, reducing the model's bias toward neutral stance.

### 4.5 Bias Reduction by Pillar

**Comparison: Baseline vs After Masking**:

| Pillar | Samples | Baseline Favored | Baseline Bias | Masked Favored | Masked Bias | Bias Reduction |
|--------|---------|------------------|---------------|----------------|-------------|----------------|
| **Civil** | 4 | neutral | 1.4035 | neutral | 0.6006 | **+0.8029** ✅ |
| **Digital** | 5 | neutral | 1.3522 | neutral | 1.6411 | **-0.2889** ❌ |
| **Economic** | 16 | against | 0.6489 | against | 0.2547 | **+0.3942** ✅ |
| **Military** | 15 | neutral | 1.6267 | neutral | 1.5039 | **+0.1228** ✅ |
| **Others** | 3 | neutral | 1.1681 | neutral | 0.8433 | **+0.3248** ✅ |
| **Psychological** | 5 | against | 0.7996 | supportive | 0.8341 | **-0.0345** ❌ |
| **Social** | 2 | neutral | 2.1715 | neutral | 1.8514 | **+0.3200** ✅ |

**Summary**:
- **5 out of 7 pillars** showed bias reduction (positive values)
- **2 pillars** showed increased bias (negative values): Digital, Psychological
- **Largest reduction**: Civil pillar (+0.8029)
- **Largest increase**: Digital pillar (-0.2889)

**Interpretation**:
- Masking is **effective for most pillars** but not universal
- Some pillars (Digital, Psychological) may have different bias mechanisms
- The heads we masked may have been reducing bias for those specific pillars

### 4.6 Detailed Pillar Analysis

#### Civil Pillar (Best Improvement)
- **Baseline**: Favored neutral (bias: 1.4035)
- **After Masking**: Still favors neutral, but bias reduced to 0.6006
- **Reduction**: +0.8029 (57% reduction)
- **Interpretation**: Head masking successfully reduced bias in civil-related content

#### Economic Pillar (Good Improvement)
- **Baseline**: Favored against (bias: 0.6489)
- **After Masking**: Still favors against, but bias reduced to 0.2547
- **Reduction**: +0.3942 (61% reduction)
- **Interpretation**: Model's bias toward "against" stance for economic content was reduced

#### Digital Pillar (Increased Bias)
- **Baseline**: Favored neutral (bias: 1.3522)
- **After Masking**: Still favors neutral, but bias increased to 1.6411
- **Change**: -0.2889 (21% increase)
- **Interpretation**: The masked heads may have been reducing bias for digital content; masking them increased bias

#### Psychological Pillar (Slight Increase)
- **Baseline**: Favored against (bias: 0.7996)
- **After Masking**: **Switched to supportive** (bias: 0.8341)
- **Change**: -0.0345 (4% increase)
- **Interpretation**: Masking changed the favored stance but didn't reduce overall bias magnitude

---

## 5. Discussion

### 5.1 Key Findings

1. **Subgroup Bias Detected**: Model predictions systematically differ across pillars, confirming representational bias exists.

2. **Head Identification Works**: Successfully identified 3 biased attention heads (Layer 0 Head 5, Layer 0 Head 11, Layer 1 Head 2) that contribute to overall bias.

3. **Mitigation is Partially Effective**: Masking biased heads reduces bias for 5 out of 7 pillars, but increases bias for 2 pillars.

4. **Stance Distribution Changed**: Masking significantly increased "supportive" predictions (from 4% to 16%), reducing model's bias toward neutral stance.

5. **Pillar-Specific Effects**: Different pillars show different bias patterns and respond differently to head masking, suggesting complex bias mechanisms.

### 5.2 Why Some Pillars Show Increased Bias

**Hypothesis 1**: The masked heads may have been **reducing bias** for Digital and Psychological pillars. When we mask them, we remove a debiasing mechanism, causing bias to increase.

**Hypothesis 2**: Different pillars may have **different bias mechanisms**. The heads we identified reduce bias for most pillars but may not be relevant for Digital/Psychological content.

**Hypothesis 3**: **Small sample sizes** (Digital: 5, Psychological: 5) may lead to unstable estimates, making results less reliable.

### 5.3 Comparison with Original UniBias

| Aspect | Original UniBias | Our Adaptation |
|--------|------------------|----------------|
| **Task** | Classification, QA | Stance Classification |
| **Bias Type** | Label bias | Representational bias (subgroup) |
| **Analysis** | Overall bias | Subgroup-specific bias |
| **Evaluation** | Accuracy + Calibration | Bias reduction by subgroup |
| **Head Identification** | Three-criterion approach | Delta-based (simplified) |
| **Mitigation** | Masking/Scaling heads | Complete head masking |
| **Results** | Consistent bias reduction | Mixed (pillar-dependent) |

**Key Differences**:
- Original UniBias measures **overall bias** across all samples
- Our adaptation measures **representational bias** across subgroups
- Original UniBias has **ground truth** (correct labels) for validation
- Our adaptation reveals **differential treatment** of subgroups

### 5.4 Methodological Insights

1. **Early Layers Matter**: Most biased heads are in layers 0-1, suggesting bias is encoded early in processing.

2. **Subgroup Analysis is Essential**: Overall bias reduction doesn't guarantee reduction for all subgroups. Subgroup-specific analysis reveals differential effects.

3. **Complete Masking May Be Too Aggressive**: Some heads may have mixed effects (reducing bias for some subgroups, increasing for others). Partial masking (scaling) might be more effective.

4. **Sample Size Matters**: Pillars with few samples (Social: 2, Others: 3) show less reliable results. Larger samples needed for robust subgroup analysis.

5. **Representational Bias is Complex**: Different subgroups may have different bias mechanisms, requiring subgroup-specific mitigation strategies.

### 5.5 Limitations

1. **Small Sample Size**: Only 50 samples used (subset of 2,512), limiting statistical power.

2. **Limited Head Analysis**: Only analyzed first 2 layers (24 heads) instead of all 144 heads.

3. **Complete Masking**: Used complete masking (0.0) instead of partial masking (scaling), which may be too aggressive.

4. **No FFN Analysis**: Only analyzed attention heads, not FFN neurons (original UniBias analyzes both).

5. **Single Model**: Only tested GPT-2 (124M), results may not generalize to larger models.

6. **No Human Evaluation**: Bias measured only through logits, not validated by human evaluators.

---

## 6. Conclusions

### 6.1 Main Findings

1. **UniBias Can Be Adapted for Subgroup Analysis**: The methodology successfully adapts to detect representational bias across identifiable subgroups (pillars).

2. **Representational Bias Exists**: Model predictions systematically differ across pillars, confirming the presence of representational bias.

3. **Head Identification Works**: We identified 3 attention heads that contribute to bias, with Layer 0 Head 5 showing the strongest effect.

4. **Mitigation is Partially Effective**: Masking biased heads reduces bias for most pillars (5/7) but increases bias for some (2/7), indicating complex bias mechanisms.

5. **Subgroup-Specific Effects**: Different pillars show different bias patterns and respond differently to mitigation, highlighting the importance of subgroup analysis.

### 6.2 Implications

1. **Subgroup Analysis is Essential**: Overall bias metrics can mask differential treatment of subgroups. Subgroup-specific analysis is necessary for fair AI systems.

2. **Mitigation Strategies Should Be Subgroup-Aware**: One-size-fits-all mitigation may not work. Different subgroups may require different mitigation strategies.

3. **Complete Masking May Be Too Aggressive**: Partial masking (scaling heads by 0.5x instead of 0x) might preserve beneficial effects while reducing bias.

4. **Larger Samples Needed**: Subgroup analysis requires sufficient samples per subgroup for reliable estimates.

5. **Multi-Component Analysis**: Analyzing both attention heads and FFN neurons (as in original UniBias) may provide more comprehensive bias mitigation.

### 6.3 Future Work

1. **Full Head Analysis**: Analyze all 144 attention heads, not just first 2 layers.

2. **FFN Neuron Analysis**: Extend analysis to FFN neurons as in original UniBias.

3. **Partial Masking**: Experiment with scaling heads (0.5x, 0.25x) instead of complete masking.

4. **Larger Sample Size**: Use full dataset (2,512 samples) for more robust subgroup analysis.

5. **Multi-Model Evaluation**: Test on larger models (GPT-2 medium, GPT-2 large) to assess generalizability.

6. **Human Evaluation**: Validate bias reduction through human evaluators rating outputs.

7. **Subgroup-Specific Mitigation**: Develop pillar-specific mitigation strategies based on subgroup analysis.

8. **Longitudinal Analysis**: Track bias changes over multiple mitigation iterations.

---

## 7. References

1. Zhou, H., Feng, Z., Zhu, Z., Qian, J., & Mao, K. (2024). UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation. *Advances in Neural Information Processing Systems*.

2. Prakash, N., Hee, M. S., & Lee, R. K. (2023). TotalDefMeme: A Multi-Attribute Meme dataset on Total Defence in Singapore. *Proceedings of the 14th Conference on ACM Multimedia Systems*.

3. SEACrowd Dataset Hub: https://github.com/SEACrowd/seacrowd-datahub

---

## Appendix: Technical Details

### A.1 Dataset Schema

**Schema**: `seacrowd_imtext` (image-text multimodal)
- **Fields**: `id`, `image_paths`, `texts`, `metadata`
- **Metadata**: `tags`, `pillar_stances`
- **Pillar Stances**: `category` (pillar), `stance` (against/neutral/supportive)

### A.2 Model Configuration

```python
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    attn_implementation="eager"  # Required for head_mask support
)
```

### A.3 Head Masking Implementation

```python
head_mask = torch.ones(num_layers, num_heads, device=DEVICE)
head_mask[layer_idx, head_idx] = 0.0  # Mask specific head
outputs = model(**encoded, head_mask=head_mask)
```

### A.4 Bias Calculation

```python
# For each pillar
avg_logits = {
    'against': np.mean(logits['against']),
    'neutral': np.mean(logits['neutral']),
    'supportive': np.mean(logits['supportive'])
}
bias_magnitude = max(avg_logits.values()) - min(avg_logits.values())
favored_stance = max(avg_logits.items(), key=lambda x: x[1])[0]
```

---

**Report Generated**: Based on analysis from `task4b.ipynb`  
**Dataset**: SEACrowd/total_defense_meme  
**Model**: GPT-2 (124M parameters)  
**Analysis Date**: 2024

