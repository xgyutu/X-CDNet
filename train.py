from ultralytics import YOLO

model = YOLO("/root/ultralytics-main/X-CDNet.yaml")  # build a YOLOv8n model from scratch

# YOLO("model.pt")  use pre-trained model if available
model.info()  # display model information
model.train(
    # data="/root/ultralytics-main/ultralytics/cfg/datasets/coco.yaml",
            data="/root/ultralytics-main/CD9K.yaml",
            epochs=200,
            name="X-CDNet",
    verbose=True,
           imgsz=640,
           device='0',
           batch=-1,
           project="/root/tf-logs/runs/train",
           pretrained="/root/ultralytics-main/yolov8s.pt")  # train the model

# model.predict("/root/autodl-tmp/bdd100k/images/val", device='0')


# from ultralytics import YOLO

# model = YOLO("/root/ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml")  # build a YOLOv8n model from scratch

# # YOLO("model.pt")  use pre-trained model if available
# model.info()  # display model information
# model.train(
#     # data="/root/ultralytics-main/ultralytics/cfg/datasets/coco.yaml",
#             data="/root/ultralytics-main/ultralytics/cfg/datasets/coco.yaml",
#             epochs=200,
#             name="yolov8s",
#     verbose=True,
#            imgsz=640,
#            device='1',
#            batch=60,
#            project="/root/tf-logs/runs/train")  # train the model

# model.predict("/root/autodl-tmp/images/val2017", device='0')