from jiwer import wer

reference = """The stale smell of old beer lingers.

It takes heat to bring out the odor.

A cold dip restores health and zest.

A salt pickle tastes fine with ham.

Tacos al pastor are my favorite.

A zestful food is the hot cross bun."""
hypothesis = "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pastor are my favorite. A zestful food is the hot cross bun."


error = wer(reference, hypothesis)

print(f"Word Error Rate: {error:.2f}")