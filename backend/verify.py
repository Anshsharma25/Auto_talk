import sounddevice as sd
print("Default input device:", sd.default.device)
print("Available devices:")
print(sd.query_devices())
