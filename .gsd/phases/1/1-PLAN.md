---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Project Structure & Dependencies

## Objective
Set up the BeamFinder project directory structure, Python dependencies, and git configuration. This is the foundation everything else builds on.

## Context
- .gsd/SPEC.md
- .gsd/ROADMAP.md

## Tasks

<task type="auto">
  <name>Create project directory structure</name>
  <files>
    src/
    src/__init__.py
    data/images/ (empty, for user's drone images)
    output/ (empty, for CSV results)
  </files>
  <action>
    Create the following directory layout:
    ```
    BeamFinder/
    ├── src/                  # Source code
    │   ├── __init__.py       # Package marker
    ├── data/
    │   └── images/           # User places drone images here
    │       └── .gitkeep
    ├── output/               # CSV detection results go here
    │   └── .gitkeep
    ```
    - Use .gitkeep files so empty dirs are tracked by git
    - src/__init__.py should be empty (just a package marker)
  </action>
  <verify>
    ```powershell
    Test-Path "src/__init__.py"; Test-Path "data/images/.gitkeep"; Test-Path "output/.gitkeep"
    ```
    All should return True.
  </verify>
  <done>Directory structure exists with src/, data/images/, output/ and all gitkeep files</done>
</task>

<task type="auto">
  <name>Create requirements.txt and .gitignore</name>
  <files>
    requirements.txt
    .gitignore
  </files>
  <action>
    Create requirements.txt with:
    ```
    ultralytics>=8.4.0
    ```
    Ultralytics pulls in torch, opencv, numpy, etc. as transitive dependencies.
    Do NOT pin torch separately — let Ultralytics manage it.

    Create .gitignore with standard Python ignores:
    - __pycache__/, *.pyc, .env, venv/
    - data/images/* (but keep .gitkeep)
    - output/* (but keep .gitkeep)
    - *.pt (model weight files — large, shouldn't be in git)
    - IDE files (.vscode/, .idea/)
  </action>
  <verify>
    ```powershell
    Get-Content "requirements.txt"; Get-Content ".gitignore"
    ```
    Both files exist with correct content.
  </verify>
  <done>requirements.txt lists ultralytics>=8.4.0; .gitignore covers Python artifacts, model weights, and data files</done>
</task>

## Success Criteria
- [ ] `src/`, `data/images/`, `output/` directories exist
- [ ] `src/__init__.py` exists
- [ ] `requirements.txt` contains ultralytics dependency
- [ ] `.gitignore` covers Python artifacts, model weights, and data directories
