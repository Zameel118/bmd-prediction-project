# GitHub Repository Setup Guide

Complete step-by-step instructions for creating and publishing your BMD Prediction Project on GitHub.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Create GitHub Repository](#create-github-repository)
3. [Initialize Local Repository](#initialize-local-repository)
4. [Push to GitHub](#push-to-github)
5. [Repository Settings](#repository-settings)
6. [Creating Releases](#creating-releases)
7. [Repository Maintenance](#repository-maintenance)

---

## 🔧 Prerequisites

### Required

- ✅ GitHub account ([Sign up here](https://github.com/join))
- ✅ Git installed on your computer
  ```bash
  # Check if Git is installed
  git --version
  
  # If not installed:
  # Windows: Download from https://git-scm.com/
  # Mac: brew install git
  # Linux: sudo apt-get install git
  ```
- ✅ Project files ready (already prepared in `bmd-prediction-project/`)

### Optional

- Git credential helper configured (saves password)
  ```bash
  git config --global credential.helper store
  ```

---

## 🌟 Create GitHub Repository

### Step 1: Create New Repository

1. Go to [GitHub](https://github.com)
2. Click the **"+"** icon in top-right corner
3. Select **"New repository"**

### Step 2: Repository Settings

Fill in the form:

| Field | Value | Notes |
|-------|-------|-------|
| **Repository name** | `bmd-prediction-project` | Use this exact name |
| **Description** | `Machine Learning & Deep Learning for Bone Mineral Density Prediction from X-ray Images` | Short summary |
| **Visibility** | ⚪ Public or 🔒 Private | Your choice |
| **Initialize repository** | ❌ **Do NOT check** | We'll push existing code |
| **Add .gitignore** | ❌ **Do NOT select** | Already included |
| **Choose a license** | ❌ **Do NOT select** | Already included (MIT) |

### Step 3: Create Repository

- Click **"Create repository"** button
- You'll see a page with setup instructions
- **Keep this page open** - we'll use the URL

The URL will be:
```
https://github.com/YOUR-USERNAME/bmd-prediction-project.git
```

---

## 💻 Initialize Local Repository

### Step 1: Open Terminal/Command Prompt

Navigate to your project folder:

```bash
# Windows (PowerShell or CMD)
cd C:\Users\HP\Desktop\bmd-prediction-project

# Mac/Linux
cd /path/to/bmd-prediction-project
```

### Step 2: Initialize Git Repository

```bash
# Initialize Git in the project folder
git init

# Add all files to staging
git add .

# Create first commit
git commit -m "Initial commit: BMD Prediction Project v1.0.0"
```

**Expected output**:
```
Initialized empty Git repository in /path/to/bmd-prediction-project/.git/
[master (root-commit) abc1234] Initial commit: BMD Prediction Project v1.0.0
 XX files changed, XXXX insertions(+)
 create mode 100644 README.md
 ...
```

### Step 3: Verify Commit

```bash
# Check commit was successful
git log --oneline

# Should show:
# abc1234 (HEAD -> master) Initial commit: BMD Prediction Project v1.0.0
```

---

## 🚀 Push to GitHub

### Step 1: Add Remote Origin

Replace `YOUR-USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR-USERNAME/bmd-prediction-project.git
```

**Example**:
```bash
git remote add origin https://github.com/johnsmith/bmd-prediction-project.git
```

### Step 2: Verify Remote

```bash
git remote -v

# Should show:
# origin  https://github.com/YOUR-USERNAME/bmd-prediction-project.git (fetch)
# origin  https://github.com/YOUR-USERNAME/bmd-prediction-project.git (push)
```

### Step 3: Push to GitHub

```bash
# Push to GitHub (main branch)
git branch -M main
git push -u origin main
```

**You'll be prompted for credentials**:
```
Username: YOUR-USERNAME
Password: your-personal-access-token
```

**Note**: GitHub no longer accepts passwords. You need a **Personal Access Token**:

#### Creating a Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Set:
   - Note: "BMD Project"
   - Expiration: 90 days (or No expiration)
   - Scopes: ✅ `repo` (full control)
4. Click "Generate token"
5. **Copy the token immediately** (you won't see it again)
6. Use this token as your password when pushing

### Step 4: Verify Upload

Go to your GitHub repository URL:
```
https://github.com/YOUR-USERNAME/bmd-prediction-project
```

You should see all your files!

---

## ⚙️ Repository Settings

### Step 1: Update Repository Details

1. Go to repository page
2. Click **"⚙️ Settings"** tab
3. Update:
   - **Description**: Add description
   - **Website**: (optional) Link to your portfolio
   - **Topics**: Add tags like:
     - `machine-learning`
     - `deep-learning`
     - `medical-imaging`
     - `bone-density`
     - `pytorch`
     - `computer-vision`
     - `healthcare`

### Step 2: Add README Badges

Your README already has badges! They'll automatically work once published.

### Step 3: Enable GitHub Pages (Optional)

To create a project website:

1. Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main`, folder: `/docs` (if you want to add docs)
4. Click Save

### Step 4: Set Up Issues Templates (Optional)

Create `.github/ISSUE_TEMPLATE/` folder with templates:

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

Create bug report template:

```yaml
# .github/ISSUE_TEMPLATE/bug_report.md
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g., Windows 10]
 - Python version: [e.g., 3.8.10]
 - PyTorch version: [e.g., 2.0.0]

**Additional context**
Add any other context about the problem here.
```

---

## 📦 Creating Releases

### Step 1: Create First Release

1. Go to repository → **Releases**
2. Click **"Create a new release"**
3. Fill in:
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - Initial Release`
   - **Description**:
     ```markdown
     ## BMD Prediction Project - Initial Release
     
     First public release of the Bone Mineral Density Prediction project.
     
     ### Features
     - CNN model (ResNet-50) with 96.3% classification accuracy
     - SVM model with 90.7% classification accuracy
     - Comprehensive evaluation and visualization
     - Complete documentation
     
     ### Models Performance
     - CNN: MAE 0.111, Accuracy 96.3%
     - SVM: MAE 0.099, Accuracy 90.7% (Best)
     
     ### Downloads
     - Source code (zip)
     - Source code (tar.gz)
     
     ### Installation
     ```bash
     git clone https://github.com/YOUR-USERNAME/bmd-prediction-project.git
     cd bmd-prediction-project
     pip install -r requirements.txt
     ```
     
     See [QUICKSTART.md](QUICKSTART.md) for usage instructions.
     ```

4. Attach files (optional):
   - Trained models (if you want to share)
   - Sample predictions

5. Click **"Publish release"**

### Step 2: Future Releases

For future updates:
```bash
# Update version in code
# Make changes
git add .
git commit -m "Update: description of changes"
git push

# Create new tag
git tag v1.1.0
git push origin v1.1.0

# Create release on GitHub
```

---

## 🔄 Repository Maintenance

### Regular Updates

#### Adding New Files

```bash
# Add specific files
git add new_file.py

# Or add all changes
git add .

# Commit with message
git commit -m "Add: new feature description"

# Push to GitHub
git push
```

#### Updating Existing Files

```bash
# Make changes to files
# Then:
git add .
git commit -m "Update: what you changed"
git push
```

#### Viewing Changes

```bash
# See what changed
git status

# See differences
git diff

# See commit history
git log --oneline
```

### Branch Management

#### Create a Development Branch

```bash
# Create and switch to dev branch
git checkout -b dev

# Make changes
# Commit changes
git add .
git commit -m "Dev: experimental feature"

# Push dev branch
git push -u origin dev

# Switch back to main
git checkout main
```

#### Merge Dev into Main

```bash
# Switch to main
git checkout main

# Merge dev
git merge dev

# Push to GitHub
git push
```

### Syncing Fork (If Collaborating)

```bash
# Add upstream remote (original repo)
git remote add upstream https://github.com/ORIGINAL-OWNER/bmd-prediction-project.git

# Fetch updates
git fetch upstream

# Merge updates
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

---

## 🎨 Customization

### Update README for Your GitHub

Replace placeholders in README.md:

```markdown
# Before
[![GitHub](https://img.shields.io/badge/GitHub-yourusername-blue)]
git clone https://github.com/yourusername/bmd-prediction-project.git

# After
[![GitHub](https://img.shields.io/badge/GitHub-johnsmith-blue)]
git clone https://github.com/johnsmith/bmd-prediction-project.git
```

Search and replace:
```bash
# Find all instances of "yourusername"
grep -r "yourusername" .

# Replace (Mac/Linux)
find . -type f -name "*.md" -exec sed -i 's/yourusername/YOUR-GITHUB-USERNAME/g' {} +

# Replace (Windows with PowerShell)
Get-ChildItem -Recurse -Filter *.md | ForEach-Object {
    (Get-Content $_.FullName) -replace 'yourusername', 'YOUR-GITHUB-USERNAME' | 
    Set-Content $_.FullName
}
```

After replacing:
```bash
git add .
git commit -m "Update: personalize repository URLs"
git push
```

---

## 🔒 Privacy & Security

### What NOT to Push

Already handled by `.gitignore`:
- ❌ Dataset images (privacy + size)
- ❌ Trained models (size)
- ❌ Personal API keys
- ❌ Kaggle credentials
- ❌ Virtual environment files

### What TO Push

- ✅ Source code
- ✅ Documentation
- ✅ Requirements
- ✅ License
- ✅ Visualizations (small PNGs)

### Removing Sensitive Data

If you accidentally committed sensitive data:

```bash
# Remove file from Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (be careful!)
git push origin --force --all
```

**Better**: Use `.gitignore` before committing!

---

## 📊 Repository Analytics

### Enable Analytics

1. Settings → Insights → Traffic
2. View:
   - Visitors
   - Page views
   - Clones
   - Referrers

### Star Your Repo

Ask classmates to star your repository to increase visibility!

---

## ✅ Final Checklist

Before announcing your repository:

- [ ] All files pushed successfully
- [ ] README displays correctly
- [ ] Links in README work
- [ ] Badges show correct information
- [ ] LICENSE file present
- [ ] .gitignore working (no large files pushed)
- [ ] Repository description set
- [ ] Topics/tags added
- [ ] First release created
- [ ] All markdown files render correctly
- [ ] Images display properly
- [ ] No sensitive data included

---

## 🎓 Submitting for Assignment

### What to Submit

Submit the GitHub repository URL:
```
https://github.com/YOUR-USERNAME/bmd-prediction-project
```

### Repository Should Include

1. **Source Code**:
   - `src/BMD_Prediction.py`
   - Supporting modules

2. **Documentation**:
   - README.md
   - USAGE.md
   - DATASET.md
   - RESULTS.md

3. **Visualizations**:
   - All 8 result plots

4. **Assignment Documents**:
   - docs/ folder with PDFs

5. **Commit History**:
   - Shows your work progression
   - Meaningful commit messages

### Making Repository Professional

Add these finishing touches:

1. **Complete README**
   - Clear installation instructions
   - Usage examples
   - Results summary
   - Citations

2. **Clean Commit History**
   - Meaningful messages
   - Logical progression
   - No "test" or "fix" without context

3. **Documentation**
   - All .md files complete
   - No broken links
   - Proper formatting

4. **Code Quality**
   - Comments where needed
   - Consistent style
   - No debug prints

---

## 🆘 Troubleshooting

### Issue: "Permission denied (publickey)"

**Solution**: Use HTTPS instead of SSH
```bash
git remote set-url origin https://github.com/YOUR-USERNAME/bmd-prediction-project.git
```

### Issue: "Large files not allowed"

**Solution**: Already handled by `.gitignore`, but if needed:
```bash
# Install Git LFS for large files
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### Issue: "Authentication failed"

**Solution**: Use Personal Access Token (see Step 3 above)

### Issue: "rejected non-fast-forward"

**Solution**: Pull first, then push
```bash
git pull origin main --rebase
git push origin main
```

### Issue: "fatal: remote origin already exists"

**Solution**: Update remote URL
```bash
git remote rm origin
git remote add origin https://github.com/YOUR-USERNAME/bmd-prediction-project.git
```

---

## 📚 Resources

### Git Tutorials
- [GitHub Guides](https://guides.github.com/)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [Learn Git Branching](https://learngitbranching.js.org/)

### GitHub Features
- [GitHub Actions](https://docs.github.com/en/actions) - CI/CD
- [GitHub Pages](https://pages.github.com/) - Free hosting
- [GitHub Discussions](https://docs.github.com/en/discussions) - Community

### Best Practices
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)

---

## 🎉 Success!

Once completed:
1. Your repository is public/private on GitHub
2. Anyone (with permissions) can clone it
3. You have a professional portfolio piece
4. Ready for assignment submission

**Share your repository**:
```
🔗 https://github.com/YOUR-USERNAME/bmd-prediction-project
```

---

**Questions?** Open an issue in your repository or contact via GitHub Discussions.

**Good luck with your assignment!** 🚀
