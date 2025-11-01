with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

cleaned = code.replace('\u00A0', ' ')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(cleaned)

print("✅ Cleaned non-breaking spaces successfully. Try running again.")
