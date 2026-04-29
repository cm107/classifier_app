# Progress Tracking & Development Workflow Protocol

## 1. The "State-Sync" Strategy
To ensure development follows the intended **Core-Outward** roadmap, GitHub Copilot must maintain a synchronized state between the actual code and the project's documentation. 

### Mandatory Files for Context:
- **`WORKING_PROGRESS.md`**: The dynamic "State Tracker" located at the project root. This is the source of truth for the *current* task.
- **`docs/roadmap/milestone_X.md`**: The technical specifications and granular TODO lists for the current phase of development.

## 2. Pre-Task Protocol
Before generating or modifying any code, GitHub Copilot must:
1.  **Read `WORKING_PROGRESS.md`** to identify the `Current Milestone` and `Current Todo`.
2.  **Reference the corresponding roadmap file** in `docs/roadmap/` to understand the technical requirements (e.g., specific submodule names, mathematical constraints like **He Initialization**, or PySide6 Signal requirements).
3.  **Confirm Alignment**: If a prompt contradicts the roadmap, Copilot should ask for clarification before proceeding.

## 3. Post-Task Update Protocol
Upon successful implementation of a coding task:
- **Checkbox Update**: Automatically suggest an update to the relevant `[ ]` checkboxes in `WORKING_PROGRESS.md`.
- **State Advancement**: If all TODOs for a sub-section are complete, suggest moving the `Current Todo` to the next item in the roadmap.
- **Commit Summary**: Provide a brief summary of what was implemented and which architectural standards (e.g., Submodule Pattern, Separation of Concerns) were followed.

## 4. Milestone Verification
Every milestone in this project concludes with a **Verification Checklist**. 
- **Validation Scripting**: Before declaring a milestone "Complete," Copilot should reference the "Verification Checklist" in the roadmap and suggest a standalone test script (e.g., `test_data.py` or `test_model.py`) to verify the logic.
- **Success Criteria**: Do not advance to the next milestone in `WORKING_PROGRESS.md` until the verification steps (like the 80/20 split check or the model shape test) have been satisfied.

## 5. Development Order (Core-Outward)
Copilot should prioritize tasks in the following architectural order to ensure the "math" is solid before building the UI:
1.  **Milestone 1**: Data Foundation (`core/data_manager.py`).
2.  **Milestone 2**: Model & Training Engine (`core/model.py`, `core/trainer.py`).
3.  **Milestone 3**: Multithreading Bridge (`worker/`).
4.  **Milestone 4-6**: UI Shell, Monitoring, and Inference (`ui/`).

## 6. Interaction Guidelines
When asked specific status questions, respond as follows:
- **"What are we working on?"**: Summarize the current Milestone, the active TODO, and the immediate technical goal.
- **"Complete the current todo"**: Implement the code following the submodule pattern and then provide the updated `WORKING_PROGRESS.md` markdown.
