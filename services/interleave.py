from config import model

# def interleave_text(first_input_text:str, second_input_text:str) -> str:
#     first_input = [
#         line.strip() for line in first_input_text.splitlines() if line.strip()
#     ]
#     second_input = [
#         line.strip() for line in second_input_text.splitlines() if line.strip()
#     ]

#     result = "\n\n".join(f"{x}\n{y}" for x, y in zip(first_input, second_input))

#     return result


def interleave_text(first_input_text:str, second_input_text:str) -> str:

    messages = [
        (
            "system", "You are a helpful assistant that interleave text between to inputs. Interleave the user inputs."
        ),
        (
            "human", f"{first_input_text} {second_input_text}"
        )
    ]

    response = model.invoke(messages)

    return response.content
