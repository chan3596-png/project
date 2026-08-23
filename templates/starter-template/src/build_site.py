import os

def build_site():
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)

    index_path = os.path.join(docs_dir, "index.html")
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semiconductor AI Project</title>
</head>
<body>
    <h1>Welcome to Semiconductor AI Project</h1>
    <p>This is the generated documentation site.</p>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Successfully generated {index_path}")

if __name__ == "__main__":
    build_site()
