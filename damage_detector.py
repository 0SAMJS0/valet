from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("yolov8n")

def detect_damage(image_path):
    """
    Practical damage detector that looks for:
    - Broken/missing parts (gaps in vehicle structure)
    - Crushed/deformed areas (shape irregularities)
    - Exposed primer/metal (color anomalies)
    - Shattered glass
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path, False
    
    # Detect vehicle
    results = model(image_path, classes=[2, 5, 7], conf=0.3)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return image_path, False
    
    # Get vehicle region
    largest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
    x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    vehicle_roi = img[y1:y2, x1:x2]
    if vehicle_roi.size == 0:
        return image_path, False
    
    # Analyze for damage
    damage_score, damage_regions = analyze_structural_damage(vehicle_roi)
    
    # Threshold: score > 25 indicates likely damage
    has_damage = damage_score > 25
    
    if has_damage:
        annotated_img = img.copy()
        
        # Draw vehicle box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # Draw damage regions
        for (dx, dy, dw, dh) in damage_regions:
            abs_x, abs_y = x1 + dx, y1 + dy
            cv2.rectangle(annotated_img, (abs_x, abs_y), (abs_x + dw, abs_y + dh), 
                         (0, 0, 255), 3)
        
        # Add text overlay
        cv2.putText(annotated_img, f"DAMAGE DETECTED - Score: {damage_score:.0f}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.putText(annotated_img, f"Damaged Regions: {len(damage_regions)}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        output_path = os.path.join("static", f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        return output_path, True
    
    return image_path, False


def analyze_structural_damage(vehicle_img):
    """
    Analyzes vehicle for structural damage indicators
    """
    gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
    
    damage_score = 0
    damage_regions = []
    
    # === 1. DETECT WHITE/PRIMER (exposed undercoat after impact) ===
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in white_contours:
        area = cv2.contourArea(contour)
        if 500 < area < 50000:  # Significant white patch
            damage_score += 15
            x, y, w, h = cv2.boundingRect(contour)
            damage_regions.append((x, y, w, h))
    
    # === 2. DETECT SHARP EDGES (broken/bent metal) ===
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 250)
    
    # Dilate to connect broken edges
    kernel_edge = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel_edge, iterations=2)
    
    edge_contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sharp_edges = 0
    for contour in edge_contours:
        area = cv2.contourArea(contour)
        if area > 800:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Compactness: low value = irregular/jagged shape (damage)
                compactness = (4 * np.pi * area) / (perimeter ** 2)
                if compactness < 0.3:  # Very irregular
                    sharp_edges += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:
                        damage_regions.append((x, y, w, h))
    
    damage_score += sharp_edges * 8
    
    # === 3. TEXTURE VARIANCE (rough/uneven surface from impact) ===
    # Split into grid and check variance
    vh, vw = vehicle_img.shape[:2]
    grid_size = 50
    
    high_variance_cells = 0
    for i in range(0, vh - grid_size, grid_size):
        for j in range(0, vw - grid_size, grid_size):
            cell = gray[i:i+grid_size, j:j+grid_size]
            variance = np.var(cell)
            
            if variance > 1500:  # High texture variance
                high_variance_cells += 1
    
    damage_score += high_variance_cells * 2
    
    # === 4. COLOR DEVIATION (mismatched paint from repair/damage) ===
    # Calculate dominant color
    pixels = vehicle_img.reshape(-1, 3)
    pixels = pixels[::10]  # Sample for speed
    
    if len(pixels) > 0:
        avg_color = np.mean(pixels, axis=0)
        color_deviations = np.sum(np.abs(pixels - avg_color), axis=1)
        high_deviation_count = np.sum(color_deviations > 100)
        
        deviation_percentage = (high_deviation_count / len(pixels)) * 100
        if deviation_percentage > 15:
            damage_score += 10
    
    # Remove overlapping regions
    damage_regions = remove_overlapping_boxes(damage_regions)
    
    return damage_score, damage_regions


def remove_overlapping_boxes(boxes):
    """Remove overlapping bounding boxes, keep larger ones"""
    if len(boxes) == 0:
        return []
    
    # Sort by area (largest first)
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    
    keep = []
    for box in boxes_sorted:
        x1, y1, w1, h1 = box
        overlap = False
        
        for kept_box in keep:
            x2, y2, w2, h2 = kept_box
            
            # Check if boxes overlap significantly
            x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = x_overlap * y_overlap
            
            if overlap_area > 0.5 * (w1 * h1):  # 50% overlap
                overlap = True
                break
        
        if not overlap:
            keep.append(box)
    
    return keep
