from openai import OpenAI

client = OpenAI()  # usa OPENAI_API_KEY del entorno


def main():
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "Decime si estás vivo en una sola frase."}
        ],
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

    