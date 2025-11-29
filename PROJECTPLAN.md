## Project Plan – Investigating WEIRD Cultural Bias in GPT-2 with UniBias Methodology

**Status**: ✅ Core Implementation Complete | ⚠️ Mixed Results | 📋 Report Generated

### 1. Objectives
- ✅ **Measure WEIRD bias**: Successfully measured cultural bias in GPT-2 generations using semantic scoring
- ✅ **Attribute to components**: Identified 10 biased attention heads using UniBias adaptation
- ⚠️ **Mitigate bias**: Head masking shows logit-level changes but mixed semantic results
- ✅ **Document findings**: Comprehensive report generated with qualitative and quantitative analysis

### 2. Data Collection & Prompt Design ✅ COMPLETED

1. **Prompt Source** ✅  
   - **Implemented**: Created 71 WVS-inspired prompts covering family structure, religion, civic duties, work ethic, morality, individualism vs collectivism, gender roles, community obligations, authority, materialism, and trust.
   - **Examples**: "People should be free to decide whom they marry. My view:", "Children should be taught obedience versus independence. Explain your view:"
   - **Coverage**: 11 thematic categories with 5-6 prompts each

2. **Prompt Construction Guidelines** ✅  
   - **Implemented**: All prompts are neutral (no explicit geography), designed to reveal implicit cultural assumptions
   - **Endings**: Used "Explain your view:", "Continue:", "Discuss:", "My perspective:" to elicit richer text
   - **Size**: 71 prompts (within target range of 50-100)

3. **Generation Strategy** ✅  
   - **Implemented**: Hugging Face `gpt2` with sampling (`temperature=0.8`, `top_p=0.9`)
   - **Generations**: 3 continuations per prompt (213 total baseline outputs)
   - **Metadata**: All outputs logged with prompt, seed, and generation parameters
   - **Model**: GPT-2 (124M) with eager attention mode for head_mask support

### 3. Bias Scoring & Attribution ✅ COMPLETED

1. **Semantic Bias Metric** ✅  
   - **Implemented**: Sentence Transformers (`all-MiniLM-L6-v2`) for embedding-based scoring
   - **Reference Passages**: Created 10 WEIRD and 10 non-WEIRD reference passages representing cultural orientations
   - **Scoring Method**: Cosine similarity to reference embeddings, bias score = `WEIRD_similarity - non_WEIRD_similarity`
   - **Results**: Computed semantic bias scores for all 213 baseline and 213 masked outputs
   - ⏸️ **Classifier**: Not implemented (future work)

2. **Head-Level Analysis (UniBias Adaptation)** ✅  
   - **Implemented**: Official GPT-2 `head_mask` parameter with eager attention mode
   - **Method**: For each of 144 heads, computed logit-based bias delta when head is masked
   - **Analysis**: Analyzed all 71 prompts across all 144 attention heads (10,224 head-prompt combinations)
   - **Identification**: Used top-k selection (top 10 heads) based on average delta
   - **Results**: Identified 10 biased heads with deltas ranging from 0.05 to 0.10
   - **Key Finding**: Layer 2 contains 4 of the top 10 biased heads; early layers (0-2) show strongest bias signals

3. **Visualization & Logging** ✅  
   - **Implemented**: 
     - Heatmaps of per-head bias contribution (avg delta across prompts)
     - Bar charts comparing baseline vs masked semantic bias scores
     - Logit inspection showing baseline vs masked logits for WEIRD tokens
   - **Results**: All visualizations generated and included in report

### 4. Mitigation & Evaluation ⚠️ MIXED RESULTS

1. **Masking Strategy** ✅  
   - **Implemented**: Masked top 10 biased heads (complete masking, set to 0)
   - **Regeneration**: Generated 213 masked outputs (71 prompts × 3 continuations) with same sampling parameters
   - ⚠️ **Note**: Complete masking may be too aggressive; partial masking (scaling) not yet tested

2. **Evaluation Metrics** ✅  
   - ✅ **Semantic bias scores**: Computed for all prompts (baseline avg: -0.0034, masked avg: +0.0081)
   - ✅ **Qualitative examples**: Side-by-side comparisons documented in report
   - ⚠️ **Fluency**: Observed text quality degradation in masked outputs (fragmentation, reduced coherence)
   - ⏸️ **Robustness test**: Not implemented (future work)

3. **Additional Checks** ⚠️  
   - ✅ **Logit inspection**: Verified probability shifts for WEIRD tokens (bias reduction confirmed at logit level)
   - ⚠️ **Discrepancy**: Logit-level shows bias reduction, but semantic scores show bias increase
   - ⏸️ **FFN neurons**: Not implemented (future work)
   - ⏸️ **Fine-tuned comparison**: Not applicable (used base GPT-2)

### 5. Reporting Deliverables ✅ COMPLETED

- ✅ **Summary of dataset/prompts**: Complete list of 71 prompts organized by 11 thematic categories (see report Appendix B)
- ✅ **Detailed methodology**: Comprehensive documentation of UniBias adaptation, semantic scoring, and head identification (see report Sections 2-3)
- ✅ **Tables/plots**: 
  - Table of 10 identified biased heads with statistics
  - Logit inspection results with before/after comparisons
  - Semantic bias score results (overall and per-prompt)
  - Heatmaps and bar charts (visualizations generated)
- ✅ **Discussion**: 
  - Trade-offs between bias reduction and fluency documented
  - Analysis of discrepancy between logit and semantic results
  - Future extensions outlined (FFN analysis, partial masking, classifier-based scoring)
- ✅ **Report**: `UNIBIAS_CULTURAL_BIAS_REPORT.md` generated with full analysis

### 6. Implementation Status & Key Findings

#### ✅ Completed Improvements
1. ✅ **WVS-derived prompts**: 71 prompts covering 11 cultural themes
2. ✅ **Semantic bias scoring**: Sentence Transformers with WEIRD/non-WEIRD reference passages
3. ✅ **Prompt set size**: 71 prompts (exceeds 50 minimum target)
4. ✅ **Sampling enabled**: Temperature 0.8, top-p 0.9 for diverse generations
5. ✅ **Official head_mask**: Implemented with eager attention mode
6. ✅ **Logit inspection**: Verified probability shifts for WEIRD tokens

#### 📊 Key Results
- **Head Identification**: 10 biased heads identified (Layer 0 Head 3 strongest, delta=0.104)
- **Logit-Level**: Confirmed bias reduction when heads are masked
- **Semantic Scores**: Mixed results (38% prompts show reduction, 62% show increase)
- **Text Quality**: Degradation observed (fragmentation, reduced coherence)

#### ⚠️ Challenges Encountered
1. **Discrepancy**: Logit-level shows bias reduction, semantic scores show increase
2. **Text Quality**: Complete head masking degrades output coherence
3. **Mixed Effectiveness**: Only 38% of prompts show semantic bias reduction

#### 🔄 Future Work / Optional Extensions
1. ⏸️ **Partial masking**: Test scaling heads (0.5x, 0.7x) instead of complete masking
2. ⏸️ **FFN neuron analysis**: Extend to FFN neurons per original UniBias
3. ⏸️ **Classifier-based scoring**: Train WEIRD/non-WEIRD classifier for validation
4. ⏸️ **Human evaluation**: Validate semantic scores with human raters
5. ⏸️ **Larger models**: Test on GPT-2 medium/large or other models
6. ⏸️ **Robustness testing**: Test with perturbed prompts
7. ⏸️ **Perplexity metrics**: Quantitative fluency measurement

### 7. Lessons Learned

1. **Adaptation Success**: UniBias methodology successfully adapted from classification to generation tasks
2. **Head Identification Works**: Logit-based analysis effectively identifies biased components
3. **Complete Masking Too Aggressive**: May need partial masking to preserve text quality
4. **Semantic Scoring Limitations**: Reference passages may not fully capture nuanced cultural changes
5. **Early Layers Matter**: Most biased heads in layers 0-2, suggesting bias encoded early
6. **Layer 2 Concentration**: Layer 2 contains 40% of top biased heads, indicating importance

### 8. Deliverables Summary

- ✅ **Notebook**: `task4.ipynb` with complete implementation
- ✅ **Report**: `UNIBIAS_CULTURAL_BIAS_REPORT.md` (579 lines, comprehensive analysis)
- ✅ **Results**: 10 biased heads identified, 426 outputs analyzed (213 baseline + 213 masked)
- ✅ **Visualizations**: Heatmaps, bar charts, logit comparisons
- ✅ **Documentation**: Complete methodology, results, and discussion


