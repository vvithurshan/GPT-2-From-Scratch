
def load_data(path = "discipline.txt"):
    with open(path, "r", encoding = "utf-8") as f:
        text = f.read()
    return text