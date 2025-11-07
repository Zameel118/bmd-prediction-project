# Contributing to BMD Prediction Project

Thank you for your interest in contributing to this project! This guide will help you get started.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of PyTorch and scikit-learn
- Familiarity with medical imaging (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/bmd-prediction-project.git
   cd bmd-prediction-project
   ```

3. Add the original repository as upstream:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/bmd-prediction-project.git
   ```

## 💻 Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔧 Making Changes

### Project Structure

Please follow the existing project structure:

```
src/
├── models/          # Model architectures
├── training/        # Training utilities
└── utils/           # Helper functions
```

### Adding New Features

1. **New Models**: Add to `src/models/`
2. **New Metrics**: Add to `src/utils/metrics.py`
3. **New Visualizations**: Add to `src/utils/visualization.py`

### Best Practices

- Write clear, descriptive commit messages
- Add comments for complex logic
- Update documentation if adding new features
- Include docstrings for functions and classes
- Test your changes before submitting

## 📝 Code Style

We follow PEP 8 style guidelines:

### Python Code Style

```python
# Good
def calculate_t_score(bmd_value, reference_bmd=0.86, std_dev=0.12):
    """
    Calculate T-score from BMD value.
    
    Args:
        bmd_value (float): Patient's BMD value
        reference_bmd (float): Reference BMD (default: 0.86)
        std_dev (float): Standard deviation (default: 0.12)
    
    Returns:
        float: Calculated T-score
    """
    return (bmd_value - reference_bmd) / std_dev
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Variables**: `snake_case`

### Documentation

- Use docstrings for all functions and classes
- Update README.md if adding major features
- Add inline comments for complex logic

## 🧪 Testing

Before submitting:

1. Test with a small subset of data
2. Verify outputs are generated correctly
3. Check that visualizations render properly
4. Ensure no breaking changes to existing functionality

### Manual Testing

```bash
# Test with small epoch count
python src/BMD_Prediction.py --epochs 2 --batch-size 8
```

## 📤 Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**:
   - Go to GitHub and create a PR from your fork
   - Fill in the PR template
   - Link any related issues

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Comments added for complex sections
- [ ] Documentation updated (if needed)
- [ ] Changes tested locally
- [ ] No merge conflicts
- [ ] Descriptive commit messages

### PR Title Format

Use conventional commits format:

- `feat: Add new model architecture`
- `fix: Correct BMD calculation bug`
- `docs: Update installation instructions`
- `refactor: Improve data loading efficiency`
- `test: Add unit tests for metrics`

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Verify the bug with latest code
3. Collect relevant information

### Bug Report Template

```markdown
## Bug Description
A clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 2.0.0]
- CUDA version (if GPU): [e.g., 11.7]

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

We welcome feature suggestions! Please:

1. Check if the feature already exists or is planned
2. Clearly describe the feature and its benefits
3. Provide examples or use cases

## ❓ Questions

For questions:
- Open a GitHub Discussion
- Check existing documentation
- Review closed issues

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Unprofessional conduct

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Acknowledged in the README

## 📚 Resources

### Helpful Links

- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [PEP 8 Style Guide](https://pep8.org/)
- [Git Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows)

### Learning Resources

- [Deep Learning for Medical Imaging](https://arxiv.org/abs/1702.05747)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📞 Contact

For urgent matters or security issues:
- Email: your.email@university.edu
- GitHub Issues: [Create Issue](https://github.com/yourusername/bmd-prediction-project/issues)

---

Thank you for contributing! 🎉
