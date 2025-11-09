template_1 = """You are given a task description. Determine if it is a **classification task** (i.e., the output involves choosing from a finite set of labels like yes/no, true/false, A/B/C/D, positive/negative, support/unsupport, etc.).

Answer ONLY with "Yes" or "No".

Here are some examples:

Task: Given my personality and the job, tell me if I would be suitable.
Is it classification? Yes

Task: Give me an example of a time when you had to use your sense of humor.
Is it classification? No

Task: Fact checking - tell me if the statement is true, false, or unknown, based on your knowledge and common sense.
Is it classification? Yes

Task: Detect if the Reddit thread contains hate speech.
Is it classification? Yes

Task: Replace the placeholders in the given text with appropriate named entities.
Is it classification? No

Task: You are provided with a news article, and you need to identify all the categories that this article belongs to.
Is it classification? Yes

Task: Write a detailed budget for a trip.
Is it classification? No

Task: Answer the following multiple choice question. Select A, B, C, or D for the final answer.
Is it classification? Yes

Now analyze the next task carefully and answer only with "Yes" or "No".
Task:"""
