#!/usr/bin/env python3
"""
Super simple loss function examples!
This will teach you loss functions by seeing them in action.
"""

print("🎯 Understanding Loss Functions")
print("=" * 40)

# Example 1: Guessing a number
print("\n📚 Example 1: Guessing a Number")
print("I'm thinking of a number between 1 and 10")
print("You guess: 7")
print("My number: 5")

your_guess = 7
real_number = 5

# Method 1: Absolute difference
loss1 = abs(your_guess - real_number)
print(f"\nMethod 1 - Absolute difference: {loss1}")
print("This means: You're off by 2")

# Method 2: Squared difference  
loss2 = (your_guess - real_number) ** 2
print(f"Method 2 - Squared difference: {loss2}")
print("This means: You're off by 4 (squared)")

print("\n💡 Key insight: Both methods measure 'wrongness'")
print("   - Absolute: Simple difference")
print("   - Squared: Penalizes big mistakes more")

# Example 2: Multiple guesses
print("\n📚 Example 2: Multiple Guesses")
print("Let's see how loss changes as we get closer:")

guesses = [1, 3, 5, 7, 9]
real = 5

print(f"\nReal number: {real}")
print("Guess | Absolute Loss | Squared Loss")
print("-" * 35)

for guess in guesses:
    abs_loss = abs(guess - real)
    sq_loss = (guess - real) ** 2
    print(f"  {guess}   |      {abs_loss}      |     {sq_loss}")

print("\n💡 Notice: Loss gets smaller as we get closer to 5!")
print("   - When guess = 5, loss = 0 (perfect!)")
print("   - When guess = 1 or 9, loss is big (far off)")

# Example 3: Why squared loss is popular
print("\n📚 Example 3: Why Squared Loss?")
print("Squared loss penalizes big mistakes more:")

mistakes = [1, 2, 3, 4, 5]
print("\nMistake Size | Absolute | Squared")
print("-" * 35)

for mistake in mistakes:
    abs_loss = mistake
    sq_loss = mistake ** 2
    print(f"     {mistake}       |    {abs_loss}    |   {sq_loss}")

print("\n💡 Squared loss makes big mistakes REALLY expensive!")
print("   - Mistake of 5: Absolute=5, Squared=25")
print("   - This encourages the model to avoid big mistakes")

# Example 4: Training a simple model
print("\n📚 Example 4: Training a Model")
print("Let's train a simple model: y = 2x")
print("We want to learn the coefficient 2")

# Our model: y = w * x (we want w = 2)
w = 1.0  # Our current guess
x = 3.0  # Input
target = 6.0  # We want y = 2 * 3 = 6

print(f"\nInput x: {x}")
print(f"Target y: {target}")
print(f"Our current w: {w}")

# Forward pass
y = w * x
print(f"Our prediction y: {y}")

# Loss
loss = (y - target) ** 2
print(f"Loss: {loss}")

# Gradient (how to improve w)
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

print("\n🎉 Key Takeaways:")
print("1. Loss function = measure of 'wrongness'")
print("2. Goal: make loss as small as possible")
print("3. Squared loss penalizes big mistakes more")
print("4. We use gradients to reduce loss")
print("5. This is how neural networks learn!")

print("\n🚀 Next: We'll build this into our Tensor class!")
print("Then you can do: loss = (prediction - target) ** 2")
print("And the computer will compute gradients automatically!")
