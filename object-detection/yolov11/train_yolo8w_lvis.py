from ultralytics import YOLOWorld
import os
import torch

"""
Train the YOLO v8 world model on LVIS
"""
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# model = YOLOWorld('yolov8s-worldv2.pt')

# results = model.train(data='lvis.yaml', epochs=30, imgsz=640, batch=16)

"""
best weight is automatically stored in the runs/detect/train/weights directory as best.pt.
When I retrain the model, the best.pt weight instead
"""
model_copy = YOLOWorld('yolov8s-worldv2.pt')  # Replace with the appropriate YOLO model class

# Load the saved weights into the new instance
lvis_yolo_path = "/home/nguyennguyen/Documents/egotopo/ego-topo/object_detection/yolov11/runs/detect/train15/weights/best.pt"
model_copy.load_state_dict(torch.load(lvis_yolo_path), strict=False)

# Save this copy to a new file to keep it independent
localization_yolo_lvis_path = "/home/nguyennguyen/Documents/egotopo/experiments/yolo-egotopo/build_graph/localization_network"
localization_yolo_lvis_model = "loc_yolo8_lvis.pth"
final_path = os.path.join(localization_yolo_lvis_path, localization_yolo_lvis_model)
torch.save(model_copy.state_dict(), final_path)