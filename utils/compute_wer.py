from jiwer import wer

reference = "hello world"
hypothesis = "hello duck"

error = wer(reference, hypothesis)

print(f"Word Error Rate: {error:.2f}")