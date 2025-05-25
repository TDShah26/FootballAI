import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
from google.colab import userdata
from more_itertools import chunked
import supervision as sv
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans
import umap
import time

# Configure API keys
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

def setup_environment():
    # Install required packages with all dependencies
    os.system('pip install -q gdown')
    os.system('pip install -q inference-gpu[transformers,sam,clip,gaze,grounding-dino,yolo-world]')
    os.system('pip install -q git+https://github.com/roboflow/sports.git')

    # Download video samples if not already present
    if not os.path.exists("121364_0.mp4"):
        os.system('gdown -O "121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"')

    # Set environment variables from user secrets
    api_key = userdata.get('ROBOFLOW_API_KEY')
    if api_key:
        os.environ["ROBOFLOW_API_KEY"] = api_key
    else:
        # Fallback for testing
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    return api_key

# Player tracking and team classification utility functions
def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """Assign goalkeepers to teams based on proximity to team centroids"""
    if len(goalkeepers) == 0 or len(players) == 0:
        return np.array([])

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Check if there are players in each team
    team_0_mask = players.class_id == 0
    team_1_mask = players.class_id == 1

    if not np.any(team_0_mask) or not np.any(team_1_mask):
        # If one team has no players, assign all goalkeepers to the other team
        if np.any(team_0_mask):
            return np.zeros(len(goalkeepers), dtype=int)
        else:
            return np.ones(len(goalkeepers), dtype=int)

    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)

    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

def replace_outliers_based_on_distance(positions: List[np.ndarray], distance_threshold: float) -> List[np.ndarray]:
    """Filter out outlier positions based on distance threshold"""
    last_valid_position: Union[np.ndarray, None] = None
    cleaned_positions: List[np.ndarray] = []

    for position in positions:
        if len(position) == 0:
            cleaned_positions.append(position)
        else:
            if last_valid_position is None:
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                distance = np.linalg.norm(position - last_valid_position)
                if distance > distance_threshold:
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    cleaned_positions.append(position)
                    last_valid_position = position

    return cleaned_positions

def draw_pitch_voronoi_diagram_2(config, team_1_xy, team_2_xy, team_1_color=sv.Color.RED, team_2_color=sv.Color.WHITE,
                               opacity=0.5, padding=50, scale=0.1, pitch=None):
    """Draw Voronoi diagram on pitch with smooth gradient for team control areas"""
    from sports.annotators.soccer import draw_pitch

    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    if len(team_1_xy) == 0 or len(team_2_xy) == 0:
        return pitch  # Return unmodified pitch if either team has no players

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                      (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    # Increase steepness of the blend effect
    steepness = 15
    distance_ratio = min_distances_team_2 / np.clip(min_distances_team_1 + min_distances_team_2, a_min=1e-5, a_max=None)
    blend_factor = np.tanh((distance_ratio - 0.5) * steepness) * 0.5 + 0.5

    # Create the smooth color transition
    for c in range(3):  # Iterate over the B, G, R channels
        voronoi[:, :, c] = (blend_factor * team_1_color_bgr[c] +
                           (1 - blend_factor) * team_2_color_bgr[c]).astype(np.uint8)

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay

class TeamClassifier:
    """Simplified team classifier that clusters players into two teams based on jersey appearance"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = 'google/siglip-base-patch16-224'
        self.embedding_model = SiglipVisionModel.from_pretrained(self.model_path).to(device)
        self.embedding_processor = AutoProcessor.from_pretrained(self.model_path)
        self.kmeans = KMeans(n_clusters=2)
        self.reducer = umap.UMAP(n_components=3)
        self.fitted = False
        print(f"Team classifier initialized on device: {device}")

    def _extract_embeddings(self, crops):
        """Extract visual embeddings from player crops"""
        pil_crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batch_size = 32
        batches = chunked(pil_crops, batch_size)
        embeddings = []

        with torch.no_grad():
            for batch in batches:
                inputs = self.embedding_processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.embedding_model(**inputs)
                batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

        return np.concatenate(embeddings)

    def fit(self, crops):
        """Train the team classifier on player crops"""
        if len(crops) == 0:
            print("Warning: No crops provided for team classification training")
            return self

        print(f"Extracting embeddings from {len(crops)} player crops...")
        embeddings = self._extract_embeddings(crops)

        print("Reducing dimensionality and clustering...")
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        self.kmeans.fit(reduced_embeddings)
        self.fitted = True
        print("Team classifier trained successfully")
        return self

    def predict(self, crops):
        """Predict team assignments for player crops"""
        if not self.fitted:
            print("Warning: Team classifier not fitted yet")
            return np.zeros(len(crops), dtype=int)

        if len(crops) == 0:
            return np.array([], dtype=int)

        embeddings = self._extract_embeddings(crops)
        reduced_embeddings = self.reducer.transform(embeddings)
        return self.kmeans.predict(reduced_embeddings)

class PossessionTracker:
    """Tracks possession statistics based on ball proximity and player tracking"""
    def __init__(self, proximity_threshold=150, smoothing_window=10):
        """
        Initialize the possession tracker

        Args:
            proximity_threshold: Distance in pitch coordinates to consider a player in possession
            smoothing_window: Number of frames to use for smoothing possession calculations
        """
        self.proximity_threshold = proximity_threshold
        self.smoothing_window = smoothing_window

        # Possession stats - Track all history
        self.possession_history = []  # Tracks possession for each frame: 0, 1, 'not_detected', or 'out_of_bounds'
        self.team_possession_time = {0: 0, 1: 0, 'not_detected': 0, 'out_of_bounds': 0}
        self.zone_possession = {
            0: {
                'def': 0, 'mid': 0, 'att': 0  # Team 0 possession by zone
            },
            1: {
                'def': 0, 'mid': 0, 'att': 0  # Team 1 possession by zone
            }
        }

        # For calculating contested possession
        self.min_contest_distance_ratio = 1.5  # If second closest player is within this ratio, possession is contested

        # Track last clear possession for contested frames
        self.last_clear_possession = None  # Stores the last team that had clear possession
        self.contested_frame_threshold = 5  # Number of consecutive contested frames before switching to "last clear"

    def determine_possession(self, ball_position, team_0_positions, team_1_positions, pitch_config):
        """
        Determine which team has possession based on proximity to the ball

        Args:
            ball_position: np.array with [x, y] coordinates of the ball
            team_0_positions: np.array with shape [n, 2] for team 0 player positions
            team_1_positions: np.array with shape [n, 2] for team 1 player positions
            pitch_config: Soccer pitch configuration

        Returns:
            tuple: (team_id, zone, is_contested) where:
                - team_id: 0, 1, 'not_detected', or 'out_of_bounds'
                - zone: 'def', 'mid', 'att', or None
                - is_contested: True if possession was contested and assigned to last clear team
        """
        if len(ball_position) == 0:
            # Ball not detected - this is different from being out of play
            return 'not_detected', None, False

        # Check if ball is in play (within pitch boundaries with some margin)
        margin = 100  # Allow some margin outside the pitch
        if (ball_position[0] < -margin or
            ball_position[0] > pitch_config.length + margin or
            ball_position[1] < -margin or
            ball_position[1] > pitch_config.width + margin):
            return 'out_of_bounds', None, False  # Ball actually out of play

        # Calculate distances from ball to all players
        team_0_distances = []
        if len(team_0_positions) > 0:
            team_0_distances = np.linalg.norm(team_0_positions - ball_position, axis=1)

        team_1_distances = []
        if len(team_1_positions) > 0:
            team_1_distances = np.linalg.norm(team_1_positions - ball_position, axis=1)

        # Find closest player from each team
        team_0_min_dist = np.min(team_0_distances) if len(team_0_distances) > 0 else float('inf')
        team_1_min_dist = np.min(team_1_distances) if len(team_1_distances) > 0 else float('inf')

        # Determine possession based on proximity
        team_with_possession = None
        is_contested = False

        if team_0_min_dist > self.proximity_threshold and team_1_min_dist > self.proximity_threshold:
            # No player close enough to ball - assign to closest if reasonably close
            if min(team_0_min_dist, team_1_min_dist) < self.proximity_threshold * 2:
                team_with_possession = 0 if team_0_min_dist < team_1_min_dist else 1
            else:
                # Ball detected but no one near it - could be a loose ball situation
                # Use last clear possession to maintain continuity
                if self.last_clear_possession is not None:
                    team_with_possession = self.last_clear_possession
                    is_contested = True
                else:
                    team_with_possession = 0 if team_0_min_dist < team_1_min_dist else 1
        elif team_0_min_dist < team_1_min_dist:
            # Team 0 is closer - check for contested possession
            if team_1_min_dist < team_0_min_dist * self.min_contest_distance_ratio:
                # Contested situation - use last clear possession logic
                if self.last_clear_possession is not None:
                    team_with_possession = self.last_clear_possession
                    is_contested = True
                else:
                    # No previous clear possession, assign to closer team
                    team_with_possession = 0
            else:
                # Clear possession for team 0
                team_with_possession = 0
                self.last_clear_possession = 0  # Update last clear possession
        else:
            # Team 1 is closer - check for contested possession
            if team_0_min_dist < team_1_min_dist * self.min_contest_distance_ratio:
                # Contested situation - use last clear possession logic
                if self.last_clear_possession is not None:
                    team_with_possession = self.last_clear_possession
                    is_contested = True
                else:
                    # No previous clear possession, assign to closer team
                    team_with_possession = 1
            else:
                # Clear possession for team 1
                team_with_possession = 1
                self.last_clear_possession = 1  # Update last clear possession

        # Determine zone of possession (defensive, middle, attacking)
        zone = None
        if team_with_possession in [0, 1]:  # Only calculate zone for actual teams
            pitch_thirds = pitch_config.length / 3

            # Team 0 attacks left to right, Team 1 attacks right to left
            if team_with_possession == 0:
                if ball_position[0] < pitch_thirds:
                    zone = 'def'
                elif ball_position[0] < 2 * pitch_thirds:
                    zone = 'mid'
                else:
                    zone = 'att'
            else:  # team_with_possession == 1
                if ball_position[0] > 2 * pitch_thirds:
                    zone = 'def'
                elif ball_position[0] > pitch_thirds:
                    zone = 'mid'
                else:
                    zone = 'att'

        return team_with_possession, zone, is_contested

    def update(self, ball_position, team_0_positions, team_1_positions, pitch_config):
        """
        Update possession statistics based on current frame data

        Args:
            ball_position: np.array with [x, y] coordinates of the ball
            team_0_positions: np.array with shape [n, 2] for team 0 player positions
            team_1_positions: np.array with shape [n, 2] for team 1 player positions
            pitch_config: Soccer pitch configuration
        """
        team_with_possession, zone, is_contested = self.determine_possession(
            ball_position, team_0_positions, team_1_positions, pitch_config
        )

        # Update possession history
        self.possession_history.append(team_with_possession)

        # Update total possession time
        self.team_possession_time[team_with_possession] += 1

        # Update zone possession if applicable (only for actual teams)
        if team_with_possession in [0, 1] and zone is not None:
            self.zone_possession[team_with_possession][zone] += 1

    def get_possession_percentages(self):
        """
        Calculate possession percentages for each team BASED ON FRAMES WHERE BALL WAS DETECTED

        Returns:
            Dictionary with team possession percentages
        """
        # Calculate percentages only for team 0 and 1, excluding detection failures and out of bounds
        in_play_frames = self.team_possession_time[0] + self.team_possession_time[1]
        if in_play_frames == 0:
            return {0: 0, 1: 0}

        return {
            0: round(self.team_possession_time[0] / in_play_frames * 100, 1),
            1: round(self.team_possession_time[1] / in_play_frames * 100, 1)
        }

    def get_current_possession_team(self):
        """Get the team currently in possession based on smoothed recent history"""
        if not self.possession_history:
            return None

        # Use smoothing window for CURRENT possession determination
        recent_possession = self.possession_history[-self.smoothing_window:]
        # Filter out non-team values (not_detected, out_of_bounds)
        valid_possession = [p for p in recent_possession if p in [0, 1]]

        if not valid_possession:
            return self.last_clear_possession

        # Return most common team in possession
        team_0_count = valid_possession.count(0)
        team_1_count = valid_possession.count(1)

        if team_0_count > team_1_count:
            return 0
        elif team_1_count > team_0_count:
            return 1
        else:
            return self.last_clear_possession  # Use last clear possession for ties

    def get_zone_percentages(self):
        """
        Calculate possession percentages by zone for each team

        Returns:
            Dictionary with team zone possession percentages
        """
        result = {0: {}, 1: {}}

        for team in [0, 1]:
            total_team_possession = sum(self.zone_possession[team].values())
            if total_team_possession == 0:
                result[team] = {'def': 0, 'mid': 0, 'att': 0}
                continue

            for zone in ['def', 'mid', 'att']:
                result[team][zone] = round(
                    self.zone_possession[team][zone] / total_team_possession * 100, 1
                )

        return result

    def get_debug_stats(self):
        """Get debug information about possession tracking"""
        total_frames = len(self.possession_history)

        return {
            'total_frames_processed': total_frames,
            'team_0_frames': self.team_possession_time[0],
            'team_1_frames': self.team_possession_time[1],
            'ball_not_detected_frames': self.team_possession_time['not_detected'],
            'ball_out_of_bounds_frames': self.team_possession_time['out_of_bounds'],
            'current_possession': self.get_current_possession_team(),
            'last_clear_possession': self.last_clear_possession,
            'recent_history': self.possession_history[-10:] if len(self.possession_history) >= 10 else self.possession_history,
            'proximity_threshold': self.proximity_threshold,
            'ball_detection_success_rate': round((self.team_possession_time[0] + self.team_possession_time[1]) / max(total_frames, 1) * 100, 1),
            'ball_not_detected_percentage': round(self.team_possession_time['not_detected'] / max(total_frames, 1) * 100, 1),
            'ball_out_of_bounds_percentage': round(self.team_possession_time['out_of_bounds'] / max(total_frames, 1) * 100, 1)
        }


def integrate_possession_tracking(football_analyzer):
    """
    Add possession tracking functionality to the FootballAnalyzer class
    """
    # Add possession tracker to the analyzer
    football_analyzer.possession_tracker = PossessionTracker(
        proximity_threshold=150,  # Increased from 50 - more lenient distance
        smoothing_window=15       # Number of frames for smoothing possession calculation
    )

    # Store the original render_game_view method
    original_render_game_view = football_analyzer.render_game_view

    # Override the render_game_view method to include possession stats
    def render_game_view_with_possession(self, frame_data):
        """Render game view with possession statistics overlay"""
        # Call the original method to get the base rendered frame
        frame = original_render_game_view(frame_data)

        # Update possession tracker if we have valid data
        transformer = self.create_perspective_transformer(frame_data['key_points'])
        if transformer is not None:
            # Get pitch coordinates
            _, pitch_coordinates = self.render_pitch_view(frame_data, transformer)

            # Update possession tracker
            ball_pos = pitch_coordinates['ball'][0] if len(pitch_coordinates['ball']) > 0 else np.array([])
            team_0_pos = pitch_coordinates['team_0']
            team_1_pos = pitch_coordinates['team_1']

            self.possession_tracker.update(ball_pos, team_0_pos, team_1_pos, self.config)

        # Draw possession stats on the frame
        possession_percentages = self.possession_tracker.get_possession_percentages()
        current_possession = self.possession_tracker.get_current_possession_team()

        # Get debug stats for troubleshooting
        debug_stats = self.possession_tracker.get_debug_stats()

        # Draw possession bar
        height, width = frame.shape[:2]
        bar_height = 40
        bar_y = height - bar_height - 10

        # Draw background
        cv2.rectangle(frame, (10, bar_y), (width - 10, bar_y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, bar_y), (width - 10, bar_y + bar_height), (255, 255, 255), 2)

        # Draw possession percentage bars
        team_0_width = int((width - 20) * possession_percentages[0] / 100)
        cv2.rectangle(frame, (10, bar_y), (10 + team_0_width, bar_y + bar_height), (0, 191, 255), -1)  # Blue for team 0

        # Add percentage text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"POSSESSION: {possession_percentages[0]}%", (20, bar_y + 25),
                   font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{possession_percentages[1]}%", (width - 100, bar_y + 25),
                   font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Indicate current possession with a ball icon or text
        if current_possession is not None:
            ball_indicator_y = bar_y - 20
            text = "POSSESSION"
            if current_possession == 0:
                text_x = 20
                color = (0, 191, 255)  # Blue for team 0
            else:
                text_x = width - 150
                color = (255, 20, 147)  # Pink for team 1

            cv2.putText(frame, text, (text_x, ball_indicator_y),
                       font, 0.6, color, 2, cv2.LINE_AA)

        # Updated debug info with better breakdown
        debug_y = 30
        cv2.putText(frame, f"Frames: T0={debug_stats['team_0_frames']} T1={debug_stats['team_1_frames']} NotDetected={debug_stats['ball_not_detected_frames']} ({debug_stats['ball_not_detected_percentage']}%)",
                   (10, debug_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Detection Rate: {debug_stats['ball_detection_success_rate']}% | Last Clear: Team {debug_stats['last_clear_possession']}",
                   (10, debug_y + 15), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # Replace the original method with our modified version
    football_analyzer.render_game_view = render_game_view_with_possession.__get__(football_analyzer)

    return football_analyzer


# Updated process_video function with better debug output
def modified_process_video(self, start_time=0, duration=10, output_prefix="football_analysis"):
    """Process video and generate analysis outputs with possession statistics"""
    import cv2
    import numpy as np
    from collections import deque
    from tqdm import tqdm
    import supervision as sv
    from sports.annotators.soccer import draw_pitch

    # Get video properties
    video_info = sv.VideoInfo.from_video_path(self.video_path)
    fps = video_info.fps
    width = video_info.width
    height = video_info.height

    # Calculate frame range
    start_frame = int(start_time * fps)
    if duration is None:
        end_frame = video_info.total_frames
    else:
        end_frame = min(start_frame + int(duration * fps), video_info.total_frames)
    total_frames = end_frame - start_frame

    # Prepare paths for output videos
    game_view_path = f"{output_prefix}_game_view.mp4"
    pitch_view_path = f"{output_prefix}_pitch_view.mp4"
    voronoi_view_path = f"{output_prefix}_voronoi_view.mp4"
    combined_view_path = f"{output_prefix}_combined_view.mp4"

    # Initialize tracker, smoothing, and ball path
    self.tracker.reset()
    M_buffer = deque(maxlen=5)
    ball_path = []

    # Initialize possession tracker if not already done
    if not hasattr(self, 'possession_tracker'):
        self.possession_tracker = PossessionTracker(
            proximity_threshold=150,  # More lenient threshold
            smoothing_window=15       # About half a second at 30fps
        )

    # Process frames
    frame_generator = sv.get_video_frames_generator(
        source_path=self.video_path, start=start_frame, end=end_frame)

    # Create a blank pitch for fallback
    blank_pitch = draw_pitch(self.config)

    # Create VideoSinks with defined dimensions
    game_sink = sv.VideoSink(game_view_path, video_info=video_info)

    # For pitch view and voronoi view, we'll use the dimensions of blank_pitch
    pitch_h, pitch_w = blank_pitch.shape[:2]
    pitch_video_info = sv.VideoInfo(
        width=pitch_w,
        height=pitch_h,
        fps=fps,
        total_frames=total_frames
    )

    pitch_sink = sv.VideoSink(pitch_view_path, video_info=pitch_video_info)
    voronoi_sink = sv.VideoSink(voronoi_view_path, video_info=pitch_video_info)

    # For combined view, calculate the dimensions
    combined_w = width
    combined_h = height

    combined_video_info = sv.VideoInfo(
        width=combined_w,
        height=combined_h,
        fps=fps,
        total_frames=total_frames
    )
    combined_sink = sv.VideoSink(combined_view_path, video_info=combined_video_info)

    # Open all sinks
    game_sink.__enter__()
    pitch_sink.__enter__()
    voronoi_sink.__enter__()
    combined_sink.__enter__()

    try:
        for i, frame in tqdm(enumerate(frame_generator), total=total_frames, desc="Processing video"):
            # Process current frame
            frame_data = self.process_frame(frame)

            # Create perspective transformer
            transformer = self.create_perspective_transformer(frame_data['key_points'])

            # Default views in case transformer fails
            game_view = self.render_game_view(frame_data)  # This now updates possession
            pitch_view = blank_pitch.copy()
            voronoi_view = blank_pitch.copy()
            current_ball_xy = np.empty((0, 2), dtype=np.float32)

            # Only generate advanced views if transformer is valid
            if transformer is not None:
                # Smooth transformation matrix
                M_buffer.append(transformer.m)
                if len(M_buffer) >= 3:  # Only smooth if we have enough matrices
                    transformer.m = np.mean(np.array(M_buffer), axis=0)

                # Update ball path
                ball_detections = frame_data['ball_detections']
                if len(ball_detections) > 0:
                    try:
                        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
                        if pitch_ball_xy.shape[0] > 0:
                            current_ball_xy = pitch_ball_xy[0]
                        else:
                            current_ball_xy = np.empty((0, 2), dtype=np.float32)
                    except Exception as e:
                        print(f"Error transforming ball position: {e}")
                        current_ball_xy = np.empty((0, 2), dtype=np.float32)
                else:
                    current_ball_xy = np.empty((0, 2), dtype=np.float32)

                ball_path.append(current_ball_xy)

                # Render different views
                try:
                    pitch_result = self.render_pitch_view(frame_data, transformer)
                    if isinstance(pitch_result, tuple) and len(pitch_result) == 2:
                        pitch_view, _ = pitch_result
                    else:
                        pitch_view = pitch_result
                except Exception as e:
                    print(f"Error rendering pitch view: {e}")
                    pitch_view = blank_pitch.copy()

                try:
                    voronoi_view = self.render_voronoi_view(frame_data, transformer)
                    if voronoi_view is None or voronoi_view.size == 0:
                        voronoi_view = blank_pitch.copy()
                except Exception as e:
                    print(f"Error rendering voronoi view: {e}")
                    voronoi_view = blank_pitch.copy()
            else:
                ball_path.append(np.empty((0, 2), dtype=np.float32))

            # Create ball trail view
            try:
                ball_trail_view = self.render_ball_trail(ball_path)
                if ball_trail_view is None or ball_trail_view.size == 0:
                    ball_trail_view = blank_pitch.copy()
            except Exception as e:
                print(f"Error rendering ball trail: {e}")
                ball_trail_view = blank_pitch.copy()

            # Ensure all views have valid dimensions and data types
            game_view = cv2.resize(game_view, (width, height))

            # Make sure pitch_view and voronoi_view have consistent sizes
            pitch_view = cv2.resize(pitch_view, (pitch_w, pitch_h))
            voronoi_view = cv2.resize(voronoi_view, (pitch_w, pitch_h))
            ball_trail_view = cv2.resize(ball_trail_view, (pitch_w, pitch_h))

            # Create combined view with proper sizing
            game_view_resized = cv2.resize(game_view, (width//2, height//2))
            pitch_view_resized = cv2.resize(pitch_view, (width//2, height//2))
            voronoi_view_resized = cv2.resize(voronoi_view, (width//2, height//2))
            ball_trail_view_resized = cv2.resize(ball_trail_view, (width//2, height//2))

            top_row = np.hstack([game_view_resized, pitch_view_resized])
            bottom_row = np.hstack([voronoi_view_resized, ball_trail_view_resized])
            combined_view = np.vstack([top_row, bottom_row])

            # Make sure combined view has the right size
            combined_view = cv2.resize(combined_view, (combined_w, combined_h))

            # Write frames to video files
            game_sink.write_frame(frame=game_view)
            pitch_sink.write_frame(frame=pitch_view)
            voronoi_sink.write_frame(frame=voronoi_view)
            combined_sink.write_frame(frame=combined_view)

            # Print possession stats every 5 seconds for debugging
            if i % (fps * 5) == 0 and i > 0:
                current_stats = self.possession_tracker.get_possession_percentages()
                debug_stats = self.possession_tracker.get_debug_stats()
                print(f"Time {i/fps:.1f}s: Team A: {current_stats[0]}%, Team B: {current_stats[1]}% "
                      f"(Detection rate: {debug_stats['ball_detection_success_rate']}%) "
                      f"Ball not detected: {debug_stats['ball_not_detected_percentage']}% "
                      f"Last clear: Team {debug_stats['last_clear_possession']}")

    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        # Close all sinks
        game_sink.__exit__(None, None, None)
        pitch_sink.__exit__(None, None, None)
        voronoi_sink.__exit__(None, None, None)
        combined_sink.__exit__(None, None, None)

        # Save final possession stats
        final_possession = self.possession_tracker.get_possession_percentages()
        zone_possession = self.possession_tracker.get_zone_percentages()
        debug_info = self.possession_tracker.get_debug_stats()

        possession_results = {
            'team_possession': final_possession,
            'zone_possession': zone_possession,
            'debug_info': debug_info
        }

        print("\nFinal possession stats:")
        print(f"Team A: {final_possession[0]}% ({debug_info['team_0_frames']} frames)")
        print(f"Team B: {final_possession[1]}% ({debug_info['team_1_frames']} frames)")
        print(f"Total frames processed: {debug_info['total_frames_processed']}")
        print(f"Ball detection success rate: {debug_info['ball_detection_success_rate']}%")
        print(f"Ball not detected: {debug_info['ball_not_detected_frames']} frames ({debug_info['ball_not_detected_percentage']}%)")
        print(f"Ball actually out of bounds: {debug_info['ball_out_of_bounds_frames']} frames ({debug_info['ball_out_of_bounds_percentage']}%)")
        print(f"Last clear possession: Team {debug_info['last_clear_possession']}")

    print(f"Videos saved with prefix: {output_prefix}")
    return {
        'game_view': game_view_path,
        'pitch_view': pitch_view_path,
        'voronoi_view': voronoi_view_path,
        'combined_view': combined_view_path,
        'possession_stats': possession_results
    }

class FootballAnalyzer:
    """Main class for football video analysis and visualization"""
    def __init__(self, api_key, video_path="121364_0.mp4"):
        from inference import get_model
        from sports.configs.soccer import SoccerPitchConfiguration

        self.video_path = video_path
        self.api_key = api_key

        # Initialize models
        print("Loading object detection model...")
        self.player_detection_model = get_model(
            model_id="football-players-detection-3zvbc/11",
            api_key=api_key
        )

        print("Loading field detection model...")
        self.field_detection_model = get_model(
            model_id="football-field-detection-f07vi/14",
            api_key=api_key
        )

        # Constants
        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3

        # Initialize pitch config
        self.config = SoccerPitchConfiguration()

        # Initialize team classifier
        print("Training team classifier...")
        self.team_classifier = self._train_team_classifier()

        # Tracking
        self.tracker = sv.ByteTrack()

        # Visualization setup
        self._setup_annotators()

        print("Football analyzer initialized successfully")

    def _train_team_classifier(self, stride=30, num_frames=30):
      """Train the team classifier on player crops from the video"""
      crops = []
      # Get video info to avoid frame bounds error
      video_info = sv.VideoInfo.from_video_path(self.video_path)
      max_frames = video_info.total_frames
      # Adjust stride to ensure we don't go out of bounds
      actual_num_frames = min(num_frames, max_frames // stride)

      frame_generator = sv.get_video_frames_generator(
          source_path=self.video_path,
          stride=stride,
          end=min(stride*actual_num_frames, max_frames-1))

      for frame in tqdm(frame_generator, desc='Collecting player crops'):
          result = self.player_detection_model.infer(frame, confidence=0.3)[0]
          detections = sv.Detections.from_inference(result)
          players_detections = detections[detections.class_id == self.PLAYER_ID]
          players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
          crops += players_crops

      # Initialize and train team classifier
      team_classifier = TeamClassifier()
      return team_classifier.fit(crops)

    def _setup_annotators(self):
        """Set up visualization annotators"""
        # Game visualization
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25, height=21,
            outline_thickness=1
        )

        # Pitch visualization
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8
        )
        self.edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            thickness=2,
            edges=self.config.edges
        )

    def process_frame(self, frame):
        """Process a single frame for object detection and tracking"""
        # Detect players, ball, and referees
        result = self.player_detection_model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        # Extract ball detections
        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # Process players and referees
        all_detections = detections[detections.class_id != self.BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = self.tracker.update_with_detections(detections=all_detections)

        # Split by role
        goalkeepers_detections = all_detections[all_detections.class_id == self.GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == self.PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == self.REFEREE_ID]

        # Team assignment
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = self.team_classifier.predict(players_crops)

        if len(goalkeepers_detections) > 0:
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                players_detections, goalkeepers_detections)

        if len(referees_detections) > 0:
            referees_detections.class_id -= 1

        # Combine all person detections
        all_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections])

        # Detect pitch keypoints
        field_result = self.field_detection_model.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(field_result)

        return {
            'frame': frame,
            'ball_detections': ball_detections,
            'players_detections': players_detections,
            'goalkeepers_detections': goalkeepers_detections,
            'referees_detections': referees_detections,
            'all_detections': all_detections,
            'key_points': key_points
        }

    def create_perspective_transformer(self, key_points):
        """Create a transform from camera view to pitch coordinates"""
        from sports.common.view import ViewTransformer

        # Filter keypoints by confidence
        filter_mask = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter_mask]
        pitch_reference_points = np.array(self.config.vertices)[filter_mask]

        # Need at least 4 points for homography
        if len(frame_reference_points) < 4:
            return None

        # Create transformer from frame to pitch coordinates
        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

        return transformer

    def render_game_view(self, frame_data):
        """Render player tracking and team identification view"""
        frame = frame_data['frame'].copy()
        all_detections = frame_data['all_detections']
        ball_detections = frame_data['ball_detections']

        # Generate labels (track IDs)
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        # Apply annotations
        frame = self.ellipse_annotator.annotate(
            scene=frame,
            detections=all_detections)
        frame = self.label_annotator.annotate(
            scene=frame,
            detections=all_detections,
            labels=labels)
        frame = self.triangle_annotator.annotate(
            scene=frame,
            detections=ball_detections)

        return frame

    def render_pitch_view(self, frame_data, transformer=None):
        """Render a top-down pitch view with player positions"""
        from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

        if transformer is None:
            transformer = self.create_perspective_transformer(frame_data['key_points'])
            if transformer is None:
                # If can't create transformer, render empty pitch
                return draw_pitch(self.config)

        # Get positions for ball and players
        ball_detections = frame_data['ball_detections']
        players_detections = frame_data['players_detections']
        goalkeepers_detections = frame_data['goalkeepers_detections']
        referees_detections = frame_data['referees_detections']

        # Merge player and goalkeeper detections
        players_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections
        ])

        # Transform coordinates from frame to pitch
        pitch_coordinates = {}

        # Transform ball coordinates
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_coordinates['ball'] = transformer.transform_points(points=frame_ball_xy)
        else:
            pitch_coordinates['ball'] = np.empty((0, 2), dtype=np.float32)

        # Transform player coordinates
        if len(players_detections) > 0:
            frame_players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_coordinates['players'] = transformer.transform_points(points=frame_players_xy)
            pitch_coordinates['team_0'] = pitch_coordinates['players'][players_detections.class_id == 0]
            pitch_coordinates['team_1'] = pitch_coordinates['players'][players_detections.class_id == 1]
        else:
            pitch_coordinates['players'] = np.empty((0, 2), dtype=np.float32)
            pitch_coordinates['team_0'] = np.empty((0, 2), dtype=np.float32)
            pitch_coordinates['team_1'] = np.empty((0, 2), dtype=np.float32)

        # Transform referee coordinates
        if len(referees_detections) > 0:
            frame_referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_coordinates['referees'] = transformer.transform_points(points=frame_referees_xy)
        else:
            pitch_coordinates['referees'] = np.empty((0, 2), dtype=np.float32)

        # Draw pitch view
        annotated_pitch = draw_pitch(self.config)

        # Draw players on pitch
        if len(pitch_coordinates['team_0']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['team_0'],
                face_color=sv.Color.from_hex('00BFFF'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_pitch)

        if len(pitch_coordinates['team_1']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['team_1'],
                face_color=sv.Color.from_hex('FF1493'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_pitch)

        # Draw ball on pitch
        if len(pitch_coordinates['ball']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['ball'],
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=annotated_pitch)

        # Draw referees on pitch
        if len(pitch_coordinates['referees']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['referees'],
                face_color=sv.Color.from_hex('FFD700'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_pitch)

        return annotated_pitch, pitch_coordinates

    def render_voronoi_view(self, frame_data, transformer=None):
        """Render voronoi diagram showing team control areas"""
        if transformer is None:
            transformer = self.create_perspective_transformer(frame_data['key_points'])
            if transformer is None:
                # If can't create transformer, render empty pitch
                from sports.annotators.soccer import draw_pitch
                return draw_pitch(self.config)

        # Get pitch coordinates
        _, pitch_coordinates = self.render_pitch_view(frame_data, transformer)

        # Draw voronoi diagram
        annotated_pitch = draw_pitch_voronoi_diagram_2(
            config=self.config,
            team_1_xy=pitch_coordinates['team_0'],
            team_2_xy=pitch_coordinates['team_1'],
            team_1_color=sv.Color.from_hex('00BFFF'),
            team_2_color=sv.Color.from_hex('FF1493'))

        # Draw ball and players on top
        from sports.annotators.soccer import draw_points_on_pitch

        # Draw ball
        if len(pitch_coordinates['ball']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['ball'],
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.WHITE,
                radius=8,
                thickness=1,
                pitch=annotated_pitch)

        # Draw team 0
        if len(pitch_coordinates['team_0']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['team_0'],
                face_color=sv.Color.from_hex('00BFFF'),
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_pitch)

        # Draw team 1
        if len(pitch_coordinates['team_1']) > 0:
            annotated_pitch = draw_points_on_pitch(
                config=self.config,
                xy=pitch_coordinates['team_1'],
                face_color=sv.Color.from_hex('FF1493'),
                edge_color=sv.Color.WHITE,
                radius=16,
                thickness=1,
                pitch=annotated_pitch)

        return annotated_pitch

    def render_ball_trail(self, ball_path):
        """Render ball movement trail on the pitch"""
        from sports.annotators.soccer import draw_pitch, draw_paths_on_pitch

        # Filter out potential outliers in ball path
        filtered_path = replace_outliers_based_on_distance(ball_path, 500)

        # Draw pitch with ball path
        annotated_pitch = draw_pitch(self.config)
        annotated_pitch = draw_paths_on_pitch(
            config=self.config,
            paths=[filtered_path],
            color=sv.Color.WHITE,
            pitch=annotated_pitch)

        return annotated_pitch

    def process_video(self, start_time=0, duration=10, output_prefix="football_analysis"):
      """Process video and generate analysis outputs"""
      from sports.common.view import ViewTransformer
      from sports.annotators.soccer import draw_pitch
      import cv2
      import numpy as np
      from collections import deque
      from tqdm import tqdm
      import supervision as sv

      # Get video properties
      video_info = sv.VideoInfo.from_video_path(self.video_path)
      fps = video_info.fps
      width = video_info.width
      height = video_info.height

      # Calculate frame range
      start_frame = int(start_time * fps)
      if duration is None:
          end_frame = video_info.total_frames
      else:
          end_frame = min(start_frame + int(duration * fps), video_info.total_frames)
      total_frames = end_frame - start_frame

      # Prepare paths for output videos
      game_view_path = f"{output_prefix}_game_view.mp4"
      pitch_view_path = f"{output_prefix}_pitch_view.mp4"
      voronoi_view_path = f"{output_prefix}_voronoi_view.mp4"
      combined_view_path = f"{output_prefix}_combined_view.mp4"

      # Initialize tracker, smoothing, and ball path
      self.tracker.reset()
      M_buffer = deque(maxlen=5)
      ball_path = []

      # Process frames
      frame_generator = sv.get_video_frames_generator(
          source_path=self.video_path, start=start_frame, end=end_frame)

      # Create a blank pitch for fallback
      blank_pitch = draw_pitch(self.config)

      # Create VideoSinks with defined dimensions
      game_sink = sv.VideoSink(game_view_path, video_info=video_info)

      # For pitch view and voronoi view, we'll use the dimensions of blank_pitch
      pitch_h, pitch_w = blank_pitch.shape[:2]
      pitch_video_info = sv.VideoInfo(
          width=pitch_w,
          height=pitch_h,
          fps=fps,
          total_frames=total_frames
      )

      pitch_sink = sv.VideoSink(pitch_view_path, video_info=pitch_video_info)
      voronoi_sink = sv.VideoSink(voronoi_view_path, video_info=pitch_video_info)

      # For combined view, calculate the dimensions
      combined_w = width
      combined_h = height

      combined_video_info = sv.VideoInfo(
          width=combined_w,
          height=combined_h,
          fps=fps,
          total_frames=total_frames
      )
      combined_sink = sv.VideoSink(combined_view_path, video_info=combined_video_info)

      # Open all sinks
      game_sink.__enter__()
      pitch_sink.__enter__()
      voronoi_sink.__enter__()
      combined_sink.__enter__()

      try:
          for i, frame in tqdm(enumerate(frame_generator), total=total_frames, desc="Processing video"):
              # Process current frame
              frame_data = self.process_frame(frame)

              # Create perspective transformer
              transformer = self.create_perspective_transformer(frame_data['key_points'])

              # Default views in case transformer fails
              game_view = self.render_game_view(frame_data)
              pitch_view = blank_pitch.copy()
              voronoi_view = blank_pitch.copy()
              current_ball_xy = np.empty((0, 2), dtype=np.float32)

              # Only generate advanced views if transformer is valid
              if transformer is not None:
                  # Smooth transformation matrix
                  M_buffer.append(transformer.m)
                  if len(M_buffer) >= 3:  # Only smooth if we have enough matrices
                      transformer.m = np.mean(np.array(M_buffer), axis=0)

                  # Update ball path
                  ball_detections = frame_data['ball_detections']
                  if len(ball_detections) > 0:
                      try:
                          frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                          pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
                          current_ball_xy = pitch_ball_xy.flatten() if pitch_ball_xy.shape[0] == 1 else np.empty((0, 2), dtype=np.float32)
                      except Exception as e:
                          print(f"Error transforming ball position: {e}")
                          current_ball_xy = np.empty((0, 2), dtype=np.float32)
                  else:
                      current_ball_xy = np.empty((0, 2), dtype=np.float32)

                  ball_path.append(current_ball_xy)

                  # Render different views
                  try:
                      pitch_result = self.render_pitch_view(frame_data, transformer)
                      if isinstance(pitch_result, tuple) and len(pitch_result) == 2:
                          pitch_view, _ = pitch_result
                      else:
                          pitch_view = pitch_result
                  except Exception as e:
                      print(f"Error rendering pitch view: {e}")
                      pitch_view = blank_pitch.copy()

                  try:
                      voronoi_view = self.render_voronoi_view(frame_data, transformer)
                      if voronoi_view is None or voronoi_view.size == 0:
                          voronoi_view = blank_pitch.copy()
                  except Exception as e:
                      print(f"Error rendering voronoi view: {e}")
                      voronoi_view = blank_pitch.copy()
              else:
                  ball_path.append(np.empty((0, 2), dtype=np.float32))

              # Create ball trail view
              try:
                  ball_trail_view = self.render_ball_trail(ball_path)
                  if ball_trail_view is None or ball_trail_view.size == 0:
                      ball_trail_view = blank_pitch.copy()
              except Exception as e:
                  print(f"Error rendering ball trail: {e}")
                  ball_trail_view = blank_pitch.copy()

              # Ensure all views have valid dimensions and data types
              game_view = cv2.resize(game_view, (width, height))

              # Make sure pitch_view and voronoi_view have consistent sizes
              pitch_view = cv2.resize(pitch_view, (pitch_w, pitch_h))
              voronoi_view = cv2.resize(voronoi_view, (pitch_w, pitch_h))
              ball_trail_view = cv2.resize(ball_trail_view, (pitch_w, pitch_h))

              # Create combined view with proper sizing
              game_view_resized = cv2.resize(game_view, (width//2, height//2))
              pitch_view_resized = cv2.resize(pitch_view, (width//2, height//2))
              voronoi_view_resized = cv2.resize(voronoi_view, (width//2, height//2))
              ball_trail_view_resized = cv2.resize(ball_trail_view, (width//2, height//2))

              top_row = np.hstack([game_view_resized, pitch_view_resized])
              bottom_row = np.hstack([voronoi_view_resized, ball_trail_view_resized])
              combined_view = np.vstack([top_row, bottom_row])

              # Make sure combined view has the right size
              combined_view = cv2.resize(combined_view, (combined_w, combined_h))

              # Write frames to video files
              game_sink.write_frame(frame=game_view)
              pitch_sink.write_frame(frame=pitch_view)
              voronoi_sink.write_frame(frame=voronoi_view)
              combined_sink.write_frame(frame=combined_view)

      except Exception as e:
          print(f"Error during video processing: {e}")
      finally:
          # Close all sinks
          game_sink.__exit__(None, None, None)
          pitch_sink.__exit__(None, None, None)
          voronoi_sink.__exit__(None, None, None)
          combined_sink.__exit__(None, None, None)

      print(f"Videos saved with prefix: {output_prefix}")
      return {
          'game_view': game_view_path,
          'pitch_view': pitch_view_path,
          'voronoi_view': voronoi_view_path,
          'combined_view': combined_view_path,
      }

    # def process_video(self, start_time=0, duration=10, output_prefix="football_analysis"):
    #   """Process video and generate analysis outputs"""
    #   from sports.common.view import ViewTransformer
    #   from sports.annotators.soccer import draw_pitch
    #   import cv2
    #   import numpy as np
    #   from collections import deque
    #   from tqdm import tqdm
    #   import supervision as sv

    #   # Get video properties
    #   video_info = sv.VideoInfo.from_video_path(self.video_path)
    #   fps = video_info.fps
    #   width = video_info.width
    #   height = video_info.height

    #   # Calculate frame range
    #   start_frame = int(start_time * fps)
    #   if duration is None:
    #       end_frame = video_info.total_frames
    #   else:
    #       end_frame = min(start_frame + int(duration * fps), video_info.total_frames)
    #   total_frames = end_frame - start_frame

    #   # Prepare paths for output videos
    #   game_view_path = f"{output_prefix}_game_view.mp4"
    #   pitch_view_path = f"{output_prefix}_pitch_view.mp4"
    #   voronoi_view_path = f"{output_prefix}_voronoi_view.mp4"
    #   combined_view_path = f"{output_prefix}_combined_view.mp4"

    #   # Initialize tracker, smoothing, and ball path
    #   self.tracker.reset()
    #   M_buffer = deque(maxlen=5)
    #   ball_path = []

    #   # Process frames
    #   frame_generator = sv.get_video_frames_generator(
    #       source_path=self.video_path, start=start_frame, end=end_frame)

    #   # Use context managers for all writers except combined_view (which is handled below)
    #   with sv.VideoSink(game_view_path, video_info=video_info) as game_view_writer, \
    #       sv.VideoSink(pitch_view_path, video_info=video_info) as pitch_view_writer, \
    #       sv.VideoSink(voronoi_view_path, video_info=video_info) as voronoi_view_writer:

    #       combined_writer = None
    #       combined_writer_initialized = False

    #       for i, frame in tqdm(enumerate(frame_generator), total=total_frames, desc="Processing video"):
    #           # Process current frame
    #           frame_data = self.process_frame(frame)

    #           # Create perspective transformer
    #           transformer = self.create_perspective_transformer(frame_data['key_points'])

    #           # Skip frame if transformer can't be created
    #           if transformer is None:
    #               continue

    #           # Smooth transformation matrix
    #           M_buffer.append(transformer.m)
    #           transformer.m = np.mean(np.array(M_buffer), axis=0)

    #           # Update ball path
    #           ball_detections = frame_data['ball_detections']
    #           if len(ball_detections) > 0:
    #               frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    #               pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
    #               ball_path.append(pitch_ball_xy.flatten() if pitch_ball_xy.shape[0] == 1 else np.empty((0, 2), dtype=np.float32))
    #           else:
    #               ball_path.append(np.empty((0, 2), dtype=np.float32))

    #           # Render different views
    #           game_view = self.render_game_view(frame_data)
    #           pitch_view, _ = self.render_pitch_view(frame_data, transformer)
    #           voronoi_view = self.render_voronoi_view(frame_data, transformer)

    #           # Create combined view
    #           game_view_resized = cv2.resize(game_view, (width//2, height//2))
    #           pitch_view_resized = cv2.resize(pitch_view, (width//2, height//2))
    #           voronoi_view_resized = cv2.resize(voronoi_view, (width//2, height//2))
    #           ball_trail_view = self.render_ball_trail(ball_path)

    #           # Check if ball_trail_view is valid before resizing
    #           if ball_trail_view is not None and ball_trail_view.size > 0:
    #               ball_trail_view_resized = cv2.resize(ball_trail_view, (width//2, height//2))
    #           else:
    #               # Create an empty black image if ball_trail_view is invalid
    #               ball_trail_view = draw_pitch(self.config)
    #               ball_trail_view_resized = cv2.resize(ball_trail_view, (width//2, height//2))

    #           top_row = np.hstack([game_view_resized, pitch_view_resized])
    #           bottom_row = np.hstack([voronoi_view_resized, ball_trail_view_resized])
    #           combined_view = np.vstack([top_row, bottom_row])

    #           # Initialize or re-initialize the combined view writer with correct dimensions on the first frame
    #           if not combined_writer_initialized:
    #               combined_video_info = sv.VideoInfo(
    #                   width=combined_view.shape[1],
    #                   height=combined_view.shape[0],
    #                   fps=fps,
    #                   total_frames=total_frames
    #               )
    #               combined_writer = sv.VideoSink(combined_view_path, video_info=combined_video_info)
    #               combined_writer.__enter__()
    #               combined_writer_initialized = True

    #           # Write frames to video files
    #           game_view_writer.write_frame(frame=game_view)  # Changed write_image() to write_frame()
    #           pitch_view_writer.write_frame(frame=pitch_view)
    #           voronoi_view_writer.write_frame(frame=voronoi_view)
    #           combined_writer.write_frame(frame=combined_view)


    #       # Close the combined writer if it was initialized
    #       if combined_writer is not None:
    #           combined_writer.__exit__(None, None, None)  # Manually exit context

    #   print(f"Videos saved with prefix: {output_prefix}")
    #   return {
    #       'game_view': game_view_path,
    #       'pitch_view': pitch_view_path,
    #       'voronoi_view': voronoi_view_path,
    #       'combined_view': combined_view_path,
    #   }



def main():
    # Setup environment
    api_key = setup_environment()
    if not api_key:
        print("API key not found. Set ROBOFLOW_API_KEY in secrets or env vars.")
        return

    # Create Football Analyzer and integrate possession tracking
    video_path = "livrma.mp4"  # Replace with the actual path to your video
    analyzer = FootballAnalyzer(api_key=api_key, video_path=video_path)

    # Integrate possession tracking
    analyzer = integrate_possession_tracking(analyzer)

    # Replace process_video method
    analyzer.process_video = modified_process_video.__get__(analyzer)

    # Process video
    output_files = analyzer.process_video(
        start_time=0,
        duration=None,
        output_prefix="football_analysis_with_possession"
    )

    print("Analysis complete! Output files:")
    for view_name, filename in output_files.items():
        if isinstance(filename, str):  # Only print file paths
            print(f"- {view_name}: {filename}")

    print("\nPossession Statistics:")
    possession_stats = output_files['possession_stats']
    print(f"Team 0 possession: {possession_stats['team_possession'][0]}%")
    print(f"Team 1 possession: {possession_stats['team_possession'][1]}%")

    return output_files


def batch_process_videos(video_paths, api_key, output_dir="output_videos", duration=30):
    """Process multiple football videos in batch mode"""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for i, video_path in enumerate(video_paths):
        print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")

        # Extract video name for output
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_prefix = f"{output_dir}/{video_name}"

        # Initialize analyzer with this video
        analyzer = FootballAnalyzer(api_key=api_key, video_path=video_path)

        # Process the video
        output_files = analyzer.process_video(
            start_time=0,
            duration=duration,
            output_prefix=output_prefix
        )

        results[video_name] = output_files

    return results

def extract_analytical_metrics(analyzer, video_path, start_time=0, duration=10):
    """Extract analytical metrics from a football video"""
    # Get video properties
    video_info = sv.VideoInfo.from_video_path(video_path)
    fps = video_info.fps

    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = min(start_frame + int(duration * fps), video_info.total_frames)

    # Metrics storage
    metrics = {
        'team_possession': {0: 0, 1: 0},  # Team possession percentage
        'ball_position_heatmap': np.zeros((analyzer.config.length, analyzer.config.width)),
        'player_coverage': {0: np.zeros((analyzer.config.length, analyzer.config.width)),
                           1: np.zeros((analyzer.config.length, analyzer.config.width))},
        'ball_speed': [],
        'team_dispersion': {0: [], 1: []},  # Team spread/compactness
    }

    # Previous ball position for speed calculation
    prev_ball_pos = None

    # Process frames
    frame_generator = sv.get_video_frames_generator(
        source_path=video_path, start=start_frame, end=end_frame)

    # Reset tracker
    analyzer.tracker.reset()

    # Process each frame
    for frame in tqdm(frame_generator, desc="Extracting metrics"):
        # Process current frame
        frame_data = analyzer.process_frame(frame)

        # Create perspective transformer
        transformer = analyzer.create_perspective_transformer(frame_data['key_points'])
        if transformer is None:
            continue

        # Get pitch coordinates for players and ball
        _, pitch_coords = analyzer.render_pitch_view(frame_data, transformer)

        # Update team possession based on Voronoi cells
        if len(pitch_coords['team_0']) > 0 and len(pitch_coords['team_1']) > 0:
            # Simplified possession calculation based on player count
            metrics['team_possession'][0] += len(pitch_coords['team_0'])
            metrics['team_possession'][1] += len(pitch_coords['team_1'])

        # Update ball position heatmap
        if len(pitch_coords['ball']) > 0:
            ball_pos = pitch_coords['ball'][0]
            x, y = int(ball_pos[0]), int(ball_pos[1])

            # Check boundaries
            if 0 <= x < analyzer.config.length and 0 <= y < analyzer.config.width:
                metrics['ball_position_heatmap'][x, y] += 1

            # Calculate ball speed
            if prev_ball_pos is not None:
                speed = np.linalg.norm(ball_pos - prev_ball_pos) * fps  # units/second
                metrics['ball_speed'].append(speed)

            prev_ball_pos = ball_pos

        # Update team coverage and dispersion
        for team_id in [0, 1]:
            team_key = f'team_{team_id}'
            if len(pitch_coords[team_key]) > 0:
                # Update coverage
                for pos in pitch_coords[team_key]:
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < analyzer.config.length and 0 <= y < analyzer.config.width:
                        metrics['player_coverage'][team_id][x, y] += 1

                # Calculate team dispersion (mean distance between players)
                if len(pitch_coords[team_key]) > 1:
                    distances = []
                    for i in range(len(pitch_coords[team_key])):
                        for j in range(i+1, len(pitch_coords[team_key])):
                            dist = np.linalg.norm(pitch_coords[team_key][i] - pitch_coords[team_key][j])
                            distances.append(dist)

                    metrics['team_dispersion'][team_id].append(np.mean(distances))

    # Normalize possession percentages
    total_possession = sum(metrics['team_possession'].values())
    if total_possession > 0:
        for team_id in metrics['team_possession']:
            metrics['team_possession'][team_id] = metrics['team_possession'][team_id] / total_possession * 100

    # Average ball speed and team dispersion
    if metrics['ball_speed']:
        metrics['avg_ball_speed'] = np.mean(metrics['ball_speed'])
    else:
        metrics['avg_ball_speed'] = 0

    for team_id in metrics['team_dispersion']:
        if metrics['team_dispersion'][team_id]:
            metrics['team_dispersion'][team_id] = np.mean(metrics['team_dispersion'][team_id])
        else:
            metrics['team_dispersion'][team_id] = 0

    return metrics

if __name__ == "__main__":
    main()