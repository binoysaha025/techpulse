import multiprocessing
multiprocessing.set_start_method("fork")

from tribev2 import TribeModel

print("Loading TRIBE...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
print("TRIBE loaded!")

with open("test.txt", "w") as f:
    f.write("OpenAI just released a new model that beats every benchmark and the AI community is going insane over it.")

df = model.get_events_dataframe(text_path="test.txt")
preds, segments = model.predict(events=df)
print(f"Predicted events: {preds.shape}")