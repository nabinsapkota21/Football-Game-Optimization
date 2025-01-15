import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
from scipy.spatial.distance import cdist

# Create necessary directories for storing frames and models
def setup_directories():
    os.makedirs('frames', exist_ok=True)
    os.makedirs('models', exist_ok=True)

setup_directories()

# Zone Utility Functions
def assign_zone(player_pos, field_dim, grid_size=(5, 5)):
    zone_x = int(player_pos[0] / (field_dim[0] / grid_size[0]))
    zone_y = int(player_pos[1] / (field_dim[1] / grid_size[1]))
    return zone_y * grid_size[0] + zone_x

def calculate_zone_weights(transition_matrix, opponents_zones):
    zone_weights = transition_matrix.sum(axis=0)  # Sum probabilities to each zone
    for zone in opponents_zones:
        zone_weights[zone] *= 0.5  # Penalize zones with opponents
    return zone_weights

# Kalman Filter Tracker
class KalmanTracker:
    def __init__(self):
        self.tracker = cv2.KalmanFilter(4, 2)
        self.tracker.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.tracker.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.tracker.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, coord):
        self.tracker.correct(np.array([[np.float32(coord[0])], [np.float32(coord[1])]]))

    def predict(self):
        prediction = self.tracker.predict()
        return [int(prediction[0]), int(prediction[1])]

# Football Analyzer
class FootballAnalyzer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.trackers = []

    def detect_objects(self, frame):
        results = self.model.predict(frame, stream=True)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                center = [(x1 + x2) // 2, (y1 + y2) // 2]
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': center,
                    'confidence': conf,
                    'class': cls
                })
        return detections

    def track_players(self, detections):
        centers = [det['center'] for det in detections]
        if len(self.trackers) < len(centers):
            for _ in range(len(centers) - len(self.trackers)):
                self.trackers.append(KalmanTracker())

        predictions = []
        for i, center in enumerate(centers):
            self.trackers[i].update(center)
            predictions.append(self.trackers[i].predict())
        return predictions

# Markov Chain Analysis
class MarkovStrategy:
    def __init__(self, num_zones):
        self.num_zones = num_zones
        self.transition_matrix = np.zeros((num_zones, num_zones))

    def update_transitions(self, current_zone, next_zone):
        self.transition_matrix[current_zone, next_zone] += 1

    def normalize(self):
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = (self.transition_matrix.T / (row_sums + 1e-5)).T

    def recommend_zone(self, current_zone, opponents_zones):
        zone_weights = calculate_zone_weights(self.transition_matrix, opponents_zones)
        best_zone = np.argmax(zone_weights)
        return best_zone

# Flask Server
app = Flask(__name__)
analyzer = FootballAnalyzer('models/yolov8n.pt')
strategy_analyzer = MarkovStrategy(num_zones=25)  # Assuming a 5x5 grid

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    data = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(data.read(), np.uint8), cv2.IMREAD_COLOR)
    field_dim = (frame.shape[1], frame.shape[0])  # Field dimensions (width, height)

    # Object Detection
    detections = analyzer.detect_objects(frame)

    # Player Tracking
    tracked_positions = analyzer.track_players(detections)

    # Zone Assignment
    zones = [assign_zone(pos, field_dim) for pos in tracked_positions]

    # Team Assignment
    positions = np.array(tracked_positions)
    kmeans = KMeans(n_clusters=2).fit(positions)
    team_clusters = kmeans.labels_.tolist()  # Team labels

    # Update Strategy Analysis
    for i, zone in enumerate(zones[:-1]):
        next_zone = zones[i + 1]
        strategy_analyzer.update_transitions(zone, next_zone)

    strategy_analyzer.normalize()

    # Optimization
    current_zone = zones[0]  # Mock current zone
    opponents_zones = [zones[i] for i in range(len(zones)) if team_clusters[i] == 1]
    recommended_zone = strategy_analyzer.recommend_zone(current_zone, opponents_zones)

    return jsonify({
        'detections': detections,
        'tracked_positions': tracked_positions,
        'zones': zones,
        'team_clusters': team_clusters,
        'recommended_zone': recommended_zone,
        'transition_matrix': strategy_analyzer.transition_matrix.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
