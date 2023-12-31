from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Use the model
model.train(data="C:\\Users\\Honor!\\OneDrive\\Desktop\Man or Woman prject\\dataset\\data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("me.png")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format