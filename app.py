from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# -----------------------------------
# FLASK APP
# -----------------------------------
app = Flask(__name__)

# -----------------------------------
# UPLOAD FOLDER
# -----------------------------------
UPLOAD_FOLDER = "static/uploads"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -----------------------------------
# DATASET PATH
# -----------------------------------
dataset_path = r"C:\Users\DELL\Desktop\PlantDataset"

# -----------------------------------
# LOAD CLASS NAMES
# -----------------------------------
dataset = ImageFolder(dataset_path)

class_names = dataset.classes

# -----------------------------------
# DEVICE
# -----------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# LOAD SHUFFLENET MODEL
# -----------------------------------
model = models.shufflenet_v2_x0_5(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "plant_shuffle_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# -----------------------------------
# IMAGE TRANSFORM
# -----------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -----------------------------------
# HOME PAGE
# -----------------------------------
@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(filepath)

            image = Image.open(filepath).convert("RGB")

            image = transform(image)

            image = image.unsqueeze(0)

            image = image.to(device)

            with torch.no_grad():

                outputs = model(image)

                _, predicted = torch.max(outputs, 1)

                prediction = class_names[predicted.item()]

            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

# -----------------------------------
# RUN APP
# -----------------------------------
if __name__ == "__main__":

    app.run(debug=True)