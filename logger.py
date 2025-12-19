def log(message, file_title):
    with open(f"{file_title}.log", "a") as f:
        f.write(message + "\n")
        