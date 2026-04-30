# Contributing to Advanced Sign Language Studio

Thank you for your interest in improving this project! Contributions are welcome as issues, bug reports, documentation fixes, and pull requests.

## How to contribute

1. Fork the repository on GitHub.
2. Create a new branch for your work:
   ```bash
git checkout -b feature/my-new-feature
```
3. Make your changes in a clean, focused commit.
4. Run the project locally and verify your changes.
5. Open a pull request with a descriptive title and summary.

## Reporting issues

Please open an issue if you find a bug, have a feature request, or want to improve the documentation.

- Use the provided issue templates for bugs and feature requests.
- Provide steps to reproduce the problem if applicable.
- Include system information if the issue is environment-specific.

## Code style

- Keep Python code readable and idiomatic.
- Use consistent indentation and line lengths.
- Prefer clear variable names and smaller helper functions.

## Development

To run the app locally:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Pull request checklist

- [ ] Code compiles and runs successfully
- [ ] Changes are documented in the README if needed
- [ ] No sensitive data is included
- [ ] Commits are atomic and descriptive
