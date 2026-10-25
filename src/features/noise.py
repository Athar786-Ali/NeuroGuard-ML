import random


def add_typo(word):

    if len(word) < 3:
        return word

    idx = random.randint(0, len(word) - 2)

    chars = list(word)

    chars[idx], chars[idx + 1] = (
        chars[idx + 1],
        chars[idx]
    )

    return "".join(chars)


def add_noise(text):

    words = text.split()

    if len(words) == 0:
        return text


    noise_type = random.choice([
        "uppercase",
        "repeat",
        "exclaim",
        "typo",
        "negation"
    ])


    # ---------------- Uppercase ----------------

    if noise_type == "uppercase":

        idx = random.randint(0, len(words) - 1)

        words[idx] = words[idx].upper()


    # ---------------- Repeat letters ----------------

    elif noise_type == "repeat":

        idx = random.randint(0, len(words) - 1)

        words[idx] = words[idx] + words[idx][-1] * 3


    # ---------------- Exclamation spam ----------------

    elif noise_type == "exclaim":

        text += "!!!"

        return text


    # ---------------- Typo attack ----------------

    elif noise_type == "typo":

        idx = random.randint(0, len(words) - 1)

        words[idx] = add_typo(words[idx])


    # ---------------- Negation attack ----------------

    elif noise_type == "negation":

        idx = random.randint(0, len(words))

        words.insert(idx, "not")


    return " ".join(words)