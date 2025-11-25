from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import torch
import os

model = InceptionResnetV1(pretrained='vggface2').eval()

embeddings = []

for img_name in os.listdir("data/train_me"):
    img = Image.open(f"data/train_me/{img_name}").resize((160, 160))
    img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255
    img = img.unsqueeze(0)

    with torch.no_grad():
        emb = model(img).numpy()[0]
        embeddings.append(emb)

mean_emb = np.mean(embeddings, axis=0)

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/me_embedding.npy", mean_emb)

print("Training complete! Embedding saved to embeddings/me_embedding.npy")