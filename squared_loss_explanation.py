#!/usr/bin/env python3
"""
Why do we square the loss? Let's see it step by step!
"""

print("🎯 Why Do We Square the Loss?")
print("=" * 40)

# Example 1: Compare different loss functions
print("\n📚 Example 1: Comparing Loss Functions")
print("Let's see how different loss functions behave:")

mistakes = [0.1, 0.5, 1.0, 2.0, 5.0]
print("\nMistake | No Square | Squared | Ratio")
print("-" * 40)

for mistake in mistakes:
    no_square = abs(mistake)
    squared = mistake ** 2
    ratio = squared / no_square if no_square != 0 else 0
    print(f"  {mistake}    |    {no_square}    |   {squared}   |  {ratio:.1f}x")

print("\n💡 Notice: Squared loss grows much faster!")
print("   - Mistake of 5: No square = 5, Squared = 25")
print("   - This makes big mistakes REALLY expensive")

# Example 2: Why this helps training
print("\n📚 Example 2: Why This Helps Training")
print("Let's see how gradients behave:")

print("\nMistake | Gradient (No Square) | Gradient (Squared)")
print("-" * 50)

for mistake in mistakes:
    # For absolute loss: gradient is always ±1
    grad_no_square = 1 if mistake >= 0 else -1
    
    # For squared loss: gradient = 2 * mistake
    grad_squared = 2 * mistake
    
    print(f"  {mistake}    |         {grad_no_square}         |        {grad_squared}")

print("\n💡 Key insight: Squared loss gives bigger gradients for bigger mistakes!")
print("   - This means: bigger mistakes get bigger corrections")
print("   - This helps the model learn faster")

# Example 3: Training example
print("\n📚 Example 3: Training Example")
print("Let's train a simple model: y = w * x")
print("We want to learn w = 2")

w = 1.0  # Our current guess
x = 3.0  # Input
target = 6.0  # We want y = 2 * 3 = 6

print(f"\nInput x: {x}")
print(f"Target y: {target}")
print(f"Our current w: {w}")

# Forward pass
y = w * x
print(f"Our prediction y: {y}")

# Loss with squared difference
loss = (y - target) ** 2
print(f"Loss: {loss}")

# Gradient calculation
gradient = 2 * (y - target) * x
print(f"Gradient: {gradient}")

# Update w
learning_rate = 0.1
new_w = w - learning_rate * gradient
print(f"New w: {new_w}")

# Check improvement
new_y = new_w * x
new_loss = (new_y - target) ** 2
print(f"New prediction: {new_y}")
print(f"New loss: {new_loss}")
print(f"Improvement: {loss - new_loss}")

# Example 4: Why squared loss is smooth
print("\n📚 Example 4: Why Squared Loss is Smooth")
print("Squared loss has a nice property: it's smooth everywhere")
print("This makes gradient descent work better")

print("\nLet's see the difference:")
print("Absolute loss: |x| has a sharp point at x=0")
print("Squared loss: x² is smooth everywhere")

# Example 5: Multiple training steps
print("\n📚 Example 5: Multiple Training Steps")
print("Let's see how the model improves over time:")

w = 1.0
learning_rate = 0.1
target = 6.0
x = 3.0

print(f"\nStep | w    | Prediction | Loss   | Gradient")
print("-" * 45)

for step in range(5):
    y = w * x
    loss = (y - target) ** 2
    gradient = 2 * (y - target) * x
    w = w - learning_rate * gradient
    
    print(f"  {step}  | {w:.2f} |    {y:.2f}    | {loss:.2f}  |   {gradient:.2f}")

print(f"\n💡 Notice: Loss gets smaller each step!")
print(f"   - Started with loss: {((1.0 * 3.0) - 6.0) ** 2:.2f}")
print(f"   - Ended with loss: {((w * 3.0) - 6.0) ** 2:.2f}")

print("\n🎉 Key Takeaways:")
print("1. Squaring makes big mistakes cost WAY more")
print("2. This gives bigger gradients for bigger mistakes")
print("3. Bigger gradients = bigger corrections")
print("4. This helps the model learn faster")
print("5. Squared loss is smooth, which helps optimization")

print("\n🚀 Now you understand why we square!")
print("It's not random - it's a clever way to make training work better!")
