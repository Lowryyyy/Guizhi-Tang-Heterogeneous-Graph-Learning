"""Change the default GitHub repository URL in the submission DOCX files and project docs.

Default repository name: Guizhi-Tang-Heterogeneous-Graph-Learning
If you create the GitHub project under a different name, run:

    cd D:\\harness\\guizhi_revision\\SUBMISSION_PACKAGE_FINAL_2026-10-23
    python set_github_repo.py https://github.com/Lowryyyy/YOUR_REPO_NAME

The script updates:
    01_Manuscript\\Revised_Manuscript_Clean.docx
    01_Manuscript\\Revised_Manuscript_Marked.docx
    02_Response\\Response_to_Reviewers.docx
    GitHub_Project_Guizhi-Tang-Heterogeneous-Graph-Learning\\README.md
    GitHub_Project_Guizhi-Tang-Heterogeneous-Graph-Learning\\CITATION.cff
    GitHub_Project_Guizhi-Tang-Heterogeneous-Graph-Learning\\zenodo_metadata.json
and creates *.bak backups.
"""
from __future__ import annotations

import os
import shutil
import sys

import docx

OLD_URL = "https://github.com/Lowryyyy/Guizhi-Tang-Heterogeneous-Graph-Learning"


def replace_docx(path, new_url):
    doc = docx.Document(path)
    changed = 0
    for para in doc.paragraphs:
        if OLD_URL in para.text:
            template = para.runs[0] if para.runs else None
            new_text = para.text.replace(OLD_URL, new_url)
            for run in list(para.runs):
                run.text = ""
            run = para.add_run(new_text)
            if template is not None:
                run.bold = template.bold
                run.italic = template.italic
                run.underline = template.underline
                run.font.name = template.font.name
                run.font.size = template.font.size
                try:
                    run.font.color.rgb = template.font.color.rgb
                except Exception:
                    pass
            changed += 1
    if changed:
        shutil.copy2(path, path + ".bak")
        doc.save(path)
    return changed


def replace_text(path, new_url):
    if not os.path.exists(path):
        return 0
    s = open(path, encoding="utf-8").read()
    if OLD_URL not in s:
        return 0
    shutil.copy2(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(s.replace(OLD_URL, new_url))
    return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python set_github_repo.py https://github.com/<user>/<repo>")
        return 1
    new_url = sys.argv[1].strip().rstrip("/")
    if not new_url.startswith("https://github.com/"):
        print("ERROR: URL must start with https://github.com/")
        return 1
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
    project = os.path.join(root, "GitHub_Project_Guizhi-Tang-Heterogeneous-Graph-Learning")
    total = 0
    for rel in ["01_Manuscript/Revised_Manuscript_Clean.docx",
                "01_Manuscript/Revised_Manuscript_Marked.docx",
                "02_Response/Response_to_Reviewers.docx"]:
        path = os.path.join(root, rel.replace("/", os.sep))
        if os.path.exists(path):
            n = replace_docx(path, new_url)
            total += n
            print(f"[{'ok' if n else 'no match'}] {path}")
    for rel in ["README.md", "CITATION.cff", "zenodo_metadata.json", "docs/UPLOAD_TO_GITHUB.md"]:
        path = os.path.join(project, rel.replace("/", os.sep))
        n = replace_text(path, new_url)
        total += n
        print(f"[{'ok' if n else 'no match' if os.path.exists(path) else 'skip'}] {path}")
    print("Done. Backups are *.bak files." if total else "No default URL found; nothing changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
