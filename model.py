from transformers import pipeline

def load_granite_model():
    pipe = pipeline(
        "text-generation",
        model="ibm-granite/granite-3.3-2b-instruct",
        max_new_tokens=300,
        temperature=0.2
    )
    return pipe

def ask_model(pipe, question, context):
    messages = [
        {"role": "system", "content": "You are StudyMate, an academic assistant. Always answer using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    output = pipe(messages)
    return output[0]["generated_text"]
