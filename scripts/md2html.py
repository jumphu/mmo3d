"""Convert markdown files to self-contained HTML with embedded images."""
import base64
import re
import sys
from pathlib import Path

import markdown


def embed_images(html: str, base_dir: Path) -> str:
    """Replace img src with base64 data URIs."""
    def replace_img(match):
        full_tag = match.group(0)
        src = match.group(1)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            return full_tag
        suffix = img_path.suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix.lstrip("."), "image/png")
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        print(f"  Embedded: {src} ({img_path.stat().st_size // 1024}KB)")
        return full_tag.replace(src, f"data:{mime};base64,{b64}")
    return re.sub(r'<img[^>]+src="([^"]+)"', replace_img, html)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }
h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; }
h2 { border-bottom: 1px solid #eee; padding-bottom: 6px; margin-top: 2em; }
h3 { margin-top: 1.5em; }
h4 { margin-top: 1.2em; color: #555; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
th { background: #f6f8fa; }
tr:nth-child(even) { background: #fafbfc; }
code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
img { max-width: 100%; height: auto; margin: 10px 0; }
blockquote { border-left: 4px solid #ddd; margin: 0; padding: 0 16px; color: #666; }
a { color: #0366d6; text-decoration: none; }
"""


def convert(md_path: str, html_path: str):
    md_file = Path(md_path)
    print(f"Converting: {md_file}")
    text = md_file.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    # Add id attributes to headings for anchor navigation
    def add_heading_id(match):
        tag = match.group(1)
        content = match.group(2)
        # Strip HTML tags from content for id generation
        plain = re.sub(r'<[^>]+>', '', content)
        slug = re.sub(r'[^\w\u4e00-\u9fff-]', '', plain.replace(' ', '-').replace('/', '')).lower()
        return f'<h{tag} id="{slug}">{content}</h{tag}>'
    html_body = re.sub(r'<h([1-6])>(.*?)</h\1>', add_heading_id, html_body)
    html_body = embed_images(html_body, md_file.parent)
    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{md_file.stem}</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""
    out = Path(html_path)
    out.write_text(html_doc, encoding="utf-8")
    print(f"Output: {out} ({out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python md2html.py input.md output.html")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
