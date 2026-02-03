import secrets

# Generate a secure 32-character API key
api_key = secrets.token_urlsafe(32)
print(f"Your API Key: {api_key}")

# Output example: sk_zs9XYCbTPKvux46UJckflw_A7B3c9D
