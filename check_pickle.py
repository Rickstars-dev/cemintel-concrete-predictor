import pickle
import sys

# Check if BitGenerator is in the pickle file
with open('model/concrete_model.pkl', 'rb') as f:
    content = f.read()
    
has_bitgen = b'BitGenerator' in content or b'MT19937' in content
has_random_state = b'_random_state' in content

print(f"🔍 Pickle file analysis:")
print(f"   File size: {len(content)} bytes")
print(f"   Contains 'BitGenerator': {has_bitgen}")
print(f"   Contains 'MT19937': {b'MT19937' in content}")
print(f"   Contains '_random_state': {has_random_state}")

if has_bitgen or b'MT19937' in content:
    print("   ❌ WARNING: BitGenerator still present in pickle!")
else:
    print("   ✅ SUCCESS: No BitGenerator found in pickle!")

# Try loading to confirm
try:
    model = pickle.load(open('model/concrete_model.pkl', 'rb'))
    print(f"\n✅ Model loads successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Has random_state attr: {hasattr(model, 'random_state')}")
    if hasattr(model, 'random_state'):
        print(f"   random_state value: {model.random_state}")
    print(f"   Has _random_state attr: {hasattr(model, '_random_state')}")
    if hasattr(model, '_random_state'):
        print(f"   _random_state value: {model._random_state}")
except Exception as e:
    print(f"\n❌ Error loading: {e}")
