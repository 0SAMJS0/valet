import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os

class DamageDetector:
    def __init__(self, model_path='models/damage_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print(f"✅ Damage detection model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            print("Train model first: python3 train_damage_model.py")
            self.model_loaded = False
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict if image contains damage"""
        if not self.model_loaded:
            return 0.0, "Model not loaded"
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                damage_prob = probabilities[0][1].item()
            
            return damage_prob, "Damaged" if damage_prob > 0.5 else "Clean"
        except Exception as e:
            print(f"Error predicting: {e}")
            return 0.0, "Error"

# Global detector instance
detector = None

def detect_damage(image_path):
    """
    Main function called by Flask app
    Returns: (output_path, has_damage)
    """
    global detector
    
    # Initialize detector on first use
    if detector is None:
        detector = DamageDetector()
    
    if not detector.model_loaded:
        # Model not available - return original image
        return image_path, False
    
    # Predict damage
    damage_prob, prediction = detector.predict(image_path)
    confidence_percent = damage_prob * 100
    
    # Threshold: 50% confidence
    has_damage = damage_prob > 0.3
    
    # Annotate image
    img = cv2.imread(image_path)
    if img is None:
        return image_path, False
    
    # Add prediction overlay
    if has_damage:
        # Red banner for damage
        color = (0, 0, 255)
        text = f"DAMAGE DETECTED - {confidence_percent:.1f}% Confidence"
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), color, -1)
        cv2.putText(img, text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        # Green banner for clean
        color = (0, 255, 0)
        text = f"NO DAMAGE - {(100-confidence_percent):.1f}% Confidence"
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), color, -1)
        cv2.putText(img, text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Save annotated image
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, img)
    
    return output_path, has_damage
