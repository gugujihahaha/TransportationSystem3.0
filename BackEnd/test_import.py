import sys
import os

print("sys.path:", sys.path)

exp3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp3")
print("exp3_path:", exp3_path)

sys.path.insert(0, exp3_path)

print("sys.path after insert:", sys.path)

try:
    import exp3.src.model_weather
    print("✅ exp3.src.model_weather loaded!")
except Exception as e:
    print("❌", e)
    import traceback
    traceback.print_exc()

try:
    from exp3.src.model_weather import TransportationModeClassifierWithWeather
    print("✅ TransportationModeClassifierWithWeather imported!")
except Exception as e:
    print("❌", e)
    import traceback
    traceback.print_exc()
