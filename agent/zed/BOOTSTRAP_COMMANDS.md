# Zed Bootstrap Commands

Use the corresponding one-line instruction when opening a new Zed thread. The role file remains the canonical prompt.

## Persistent scientific threads

### SCI-DIRECTOR
```text
Read AGENTS.md and agent/prompts/scientific/01_scientific_director.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-NOVELTY
```text
Read AGENTS.md and agent/prompts/scientific/02_novelty_literature_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-DATA
```text
Read AGENTS.md and agent/prompts/scientific/03_data_validity_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-ML
```text
Read AGENTS.md and agent/prompts/scientific/04_ml_methodology_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-STATS
```text
Read AGENTS.md and agent/prompts/scientific/05_evaluation_statistics_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-XAI
```text
Read AGENTS.md and agent/prompts/scientific/06_xai_responsible_ai_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### SCI-PUB
```text
Read AGENTS.md and agent/prompts/scientific/07_publication_reviewer_advisor.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

## Persistent execution/control threads

### IMP-LEAD
```text
Read AGENTS.md and agent/prompts/implementation/01_implementation_lead.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### VER-TECH
```text
Read AGENTS.md and agent/prompts/review/01_technical_verifier.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

### MGT-SECRETARY
```text
Read AGENTS.md and agent/prompts/management/01_project_secretary.md. Adopt that file as your persistent operating contract. Execute its startup sequence before doing any task.
```

## Specialist threads

Specialist prompt paths and IDs are listed in `agent/registry.yaml`. Create these threads only when the Scientific Director or a domain advisor identifies a concrete need. Do not keep every specialist permanently active.

## Local worker threads

Create IMP-DATA, IMP-ML, IMP-STATS, IMP-XAI, IMP-RUN and IMP-REPRO only against a specific READY Work Package. Include the WP path in the bootstrap message after the role contract.
