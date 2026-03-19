def interleave(first_input_text:str, second_input_text:str) -> str:
    first_input = [
        line.strip() for line in first_input_text.splitlines() if line.strip()
    ]
    second_input = [
        line.strip() for line in second_input_text.splitlines() if line.strip()
    ]

    result = "\n\n".join(f"{x}\n{y}" for x, y in zip(first_input, second_input))

    return result