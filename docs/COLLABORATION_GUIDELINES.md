# Collaboration Guidelines

Current phase: beta.

This project may be edited by multiple coding roles in parallel. Every role should treat the working tree as a shared collaboration space.

## Commit Discipline

Before every commit:

1. Run `git status --short`.
2. Inspect the relevant staged and unstaged diffs.
3. Separate current-task changes from unrelated generated data, temporary files, or local-only files.
4. Check whether other non-conflicting changes are already present and should be included in the same latest-state commit.
5. If another role changed a file that the current task also touches, read the latest file content and adapt to it instead of overwriting or reverting it.
6. Stage only intentional files.
7. Commit from the latest verified working-tree state.
8. Re-check `git status --short` after committing.

Do not assume that only your own edits matter. If a non-conflicting change is part of the current project state and is ready to ship, include it deliberately. If a file is unrelated, generated, large, or local-only, leave it unstaged and mention that decision when relevant.

## Conflict Handling

If changes appear to conflict:

- stop and inspect the exact files
- preserve other roles' work unless the user explicitly asks for a revert
- prefer a small integration patch over replacing the whole file
- validate the changed surface after integration

## Generated Data

Large generated datasets and local reports should not be committed unless explicitly requested. For this project, generated `test_data_v*` directories and CSV reports are normally local evaluation artifacts.
