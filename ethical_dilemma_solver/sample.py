from groq import Groq

client = Groq(api_key="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")

models = client.models.list()
for model in models.data:
    print(model.id)
