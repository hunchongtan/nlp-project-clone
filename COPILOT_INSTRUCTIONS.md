## Copilot Instructions for the Next Developer

1. **Get Familiar with the Codebase**
   - Ask Copilot: “Summarize the current architecture and key components based on `README.md`.”  
   - Explore the Python modules (`main.py`, `attention_manipulate.py`, `FFN_manipulate.py`, `evaluation.py`, `utils.py`) to understand how UniBias is implemented.  
   - Example prompt: “Copilot, explain how `attention_manipulate.py` identifies biased heads.”

2. **Review the Updated Project Plan**
   - After understanding the codebase, open `PROJECTPLAN.md`.  
   - Prompt Copilot: “List the concrete steps from `PROJECTPLAN.md` so I can map them to code changes.”  
   - Focus on new requirements: WVS-based prompts, sampling-based generation, semantic bias scoring, head masking via `head_mask`.

3. **Implement the Plan in `task4.ipynb`**
   - Use Copilot to scaffold code cells that follow the plan.  
   - Example prompts:
     - “Generate Python code to load WVS-inspired prompts and apply sampling with GPT-2.”
     - “Show how to compute semantic bias scores using sentence embeddings.”  
     - “Demonstrate masking specific GPT-2 heads using the `head_mask` argument.”
   - Make sure each implementation step aligns with the Project Plan goals before running the cell.

4. **Iterate and Validate**
   - After each major change, ask Copilot for verification suggestions (e.g., “How can I log baseline vs masked logits for WEIRD tokens?”).  
   - Keep outputs and notes organized inside `task4.ipynb` so future teammates can follow the workflow.

> Reminder: Copilot should reference `README.md` for the overall UniBias context, inspect the core Python files for how the original method works, consult `PROJECTPLAN.md` for the updated experimental design, and then help you implement the workflow inside `task4.ipynb`.


