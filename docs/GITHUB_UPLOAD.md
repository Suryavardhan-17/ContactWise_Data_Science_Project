# Publish ContactWise to GitHub

## Option A — Upload through the browser (easiest)

1. Download and unzip the project to your computer.
2. In GitHub, click **New repository**.
3. Name it `ContactWise-Data-Science`, set **Public**, and click **Create repository**.
4. On the empty repository page, click **uploading an existing file**.
5. Upload the **contents** of the unzipped folder, preserving its directory structure.
   If browser drag-and-drop skips nested folders, use Option B instead.
6. Commit with message `Add ContactWise data science project`.
7. Verify README, `contactwise/`, `sql/`, `tests/`, and `docs/` appear.

## Option B — Push with Git in Terminal or VS Code (best for nested files)

Open a terminal **inside** the extracted `ContactWise_Data_Science_Project` folder.

```bash
git init
git add .
git commit -m "Initial ContactWise data science portfolio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ContactWise-Data-Science.git
git push -u origin main
```

Create the **empty** public GitHub repository first, without an automatically
generated README or .gitignore. Replace `YOUR_USERNAME` with your own username.
If Git asks for login, use your GitHub sign-in through credential manager; do not
paste passwords or tokens in chat or commit them to the repository.

## Complete the real-data analysis before sharing it with recruiters

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m contactwise.download
python -m pytest -q
python -m contactwise.run
```

Review `reports/generated/executive_summary.md` and the generated plots.
Do not invent benchmark numbers. If you decide to publish figures, copy a
carefully selected, *real-data-generated* figure to `assets/` and commit that
copy, along with an honest description in the README.
