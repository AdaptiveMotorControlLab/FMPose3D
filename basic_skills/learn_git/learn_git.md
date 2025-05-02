# Git Commands Reference

# Repository Operations

### Clone a Specific Branch
When you only need a specific branch from a repository, you can use `--single-branch` to save bandwidth and disk space:

```bash
# Syntax
git clone -b <branch_name> --single-branch <repository_url>

# Example
git clone -b pytorch_dlc --single-branch https://github.com/DeepLabCut/DeepLabCut.git
```

### Basic Git Workflow

1. Check status of your repository:
```bash
git status
```

2. Create and switch to a new branch:
```bash
git checkout -b <new-branch-name>
```

3. Stage changes:
```bash
git add <file>      # Stage specific file
git add .           # Stage all changes
```

4. Commit changes:
```bash
git commit -m "Your descriptive commit message"
```

5. Push changes:
```bash
git push origin <branch-name>
```

# Useful Tips

- Undo last commit (keeping changes):
```bash
git reset --soft HEAD~1
```

- Discard local changes:
```bash
git checkout -- <file>    # For specific file
git reset --hard HEAD     # For all files
```

## To view the commit history in a tree structure

**Command:** `git log --graph --oneline --all --decorate`

**Explanation of Options:**

- `--graph`: Displays the commit history as a tree with branches and merges.
- `--oneline`: Shows each commit on a single line for better readability.
- `--all`: Includes all branches in the tree.
- `--decorate`: Displays branch names and tags alongside commits.