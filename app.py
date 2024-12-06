from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Lock
from collections import deque
import statistics

# Initialize Flask app
app = Flask(__name__)

# Mediapipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables for camera and pose
camera = None
pose = None
camera_lock = Lock()
camera_initialized = False

def check_body_alignment(landmarks):
    """Analyze body alignment and return feedback."""
    hip_sway = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - 
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
    
    shoulder_rotation = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z - 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z)
    
    feedback = []
    if hip_sway > 0.2:
        feedback.append("Keep hips stable")
    if shoulder_rotation > 0.1:
        feedback.append("Keep shoulders square")
    return feedback

def enhance_visual_feedback(image, feedback_text, position, is_warning=False):
    """Add enhanced visual feedback to the image."""
    text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    bg_pad = 15
    cv2.rectangle(image,
                 (position[0] - bg_pad, position[1] - text_size[1] - bg_pad),
                 (position[0] + text_size[0] + bg_pad, position[1] + bg_pad),
                 (0, 0, 0),
                 -1)
    
    color = (0, 0, 255) if is_warning else (0, 255, 0)
    cv2.putText(image, feedback_text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                color,
                3)

# For form analysis
class FormAnalyzer:
    def __init__(self, window_size=10):
        self.angles_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.balance_history = deque(maxlen=window_size)
        self.last_rep_time = time.time()
        
        # For summary statistics
        self.total_reps = 0
        self.rep_times = []
        self.rep_quality_scores = []
        
        # Ideal ranges
        self.ideal_rep_time_range = (1.0, 3.0)  # seconds
        self.ideal_balance_threshold = 0.1      # shoulder alignment threshold

        # Exercise-specific ideal ranges
        self.exercise_ranges = {
            "bicep_curl": {
                "ideal_angles": (90, 120),  # Range for good form
                "ideal_tempo": (1.5, 2.5),  # Seconds per rep
                "max_elbow_drift": 0.1      # Maximum allowed elbow drift from body
            },
            "lateral_raise": {
                "ideal_angles": (25, 65),
                "ideal_tempo": (2.0, 3.0),
                "min_arm_straightness": 160  # Minimum angle for straight arms
            },
            "shoulder_press": {
                "ideal_angles": (60, 140),
                "ideal_tempo": (2.0, 3.0),
                "max_shoulder_imbalance": 0.05
            }
        }

    def analyze_fatigue(self):
        """Analyze form degradation over workout."""
        if len(self.rep_quality_scores) < 3:
            return None
            
        recent_scores = self.rep_quality_scores[-3:]
        initial_scores = self.rep_quality_scores[:3]
        
        avg_recent = sum(recent_scores) / 3
        avg_initial = sum(initial_scores) / 3
        
        if avg_recent < avg_initial * 0.85:
            return "Form declining - consider reducing weight or taking a break"
        return None    
        
    def calculate_rep_score(self, exercise_type, angles, rep_duration, landmarks):
        """Calculate a detailed form score for a single rep."""
        score_components = {}
        
        # 1. Range of Motion Score (40% of total)
        ideal_range = self.exercise_ranges[exercise_type]["ideal_angles"]
        actual_range = max(angles) - min(angles)
        expected_range = ideal_range[1] - ideal_range[0]
        rom_score = min(100, (actual_range / expected_range) * 100)
        score_components['rom'] = rom_score * 0.4
        
        # 2. Tempo Score (30% of total)
        ideal_tempo = self.exercise_ranges[exercise_type]["ideal_tempo"]
        if ideal_tempo[0] <= rep_duration <= ideal_tempo[1]:
            tempo_score = 100
        else:
            deviation = min(abs(rep_duration - ideal_tempo[0]), 
                          abs(rep_duration - ideal_tempo[1]))
            tempo_score = max(0, 100 - (deviation * 25))
        score_components['tempo'] = tempo_score * 0.3
        
        # 3. Stability Score (30% of total)
        stability_score = 100
        if exercise_type == "bicep_curl":
            # Check elbow drift
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            drift = abs(l_elbow.x - l_shoulder.x)
            if drift > self.exercise_ranges[exercise_type]["max_elbow_drift"]:
                stability_score -= (drift * 200)  # Reduce score based on drift
                
        elif exercise_type == "lateral_raise":
            # Check arm straightness
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            arm_angle = calculate_angle(
                [l_shoulder.x, l_shoulder.y],
                [l_elbow.x, l_elbow.y],
                [l_wrist.x, l_wrist.y]
            )
            if arm_angle < self.exercise_ranges[exercise_type]["min_arm_straightness"]:
                stability_score -= (160 - arm_angle)
                
        elif exercise_type == "shoulder_press":
            # Check shoulder alignment
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            imbalance = abs(l_shoulder.y - r_shoulder.y)
            if imbalance > self.exercise_ranges[exercise_type]["max_shoulder_imbalance"]:
                stability_score -= (imbalance * 200)
                
        score_components['stability'] = max(0, stability_score) * 0.3
        
        # Calculate total score
        total_score = sum(score_components.values())
        
        return total_score, score_components
        
    def analyze_form(self, current_angles, landmarks, exercise_type, is_rep_complete=False):
        """Analyze form wih specific reasoning and provide updated reasoning."""
        current_time = time.time()
        rep_duration = current_time - self.last_rep_time
        
        if is_rep_complete:
            score, components = self.calculate_rep_score(
                exercise_type, 
                current_angles, 
                rep_duration, 
                landmarks
            )
            self.rep_quality_scores.append(score)
            
            # Generate detailed feedback based on score components
            feedback = []
            if components['rom'] < 32:  # Below 80% of max ROM score
                feedback.append("Increase range of motion")
            if components['tempo'] < 21:  # Below 70% of max tempo score
                feedback.append("Control your tempo")
            if components['stability'] < 21:  # Below 70% of max stability score
                feedback.append("Focus on stability")
                
            self.rep_times.append(rep_duration)
            
            return len(feedback) == 0, " | ".join(feedback) if feedback else "Good form!"
            
        self.last_rep_time = current_time
        return True, ""
        
    def get_workout_summary(self):
        """Generate a comprehensive workout summary with detailed metrics and insights."""
        if self.total_reps == 0:
            return {
                "total_reps": 0,
                "form_percentage": 0,
                "avg_rep_time": 0,
                "feedback": "No workout data available."
            }
            
        # Calculate detailed metrics
        form_percentage = statistics.mean(self.rep_quality_scores)
        avg_rep_time = statistics.mean(self.rep_times)
        
        # Calculate consistency metrics
        time_consistency = statistics.stdev(self.rep_times) if len(self.rep_times) > 1 else 0
        form_consistency = statistics.stdev(self.rep_quality_scores) if len(self.rep_quality_scores) > 1 else 0
        
        # Analyze form progression
        form_trend = []
        window_size = 3
        for i in range(0, len(self.rep_quality_scores) - window_size + 1):
            window_avg = statistics.mean(self.rep_quality_scores[i:i + window_size])
            form_trend.append(window_avg)
        
        # Calculate fatigue indicators
        form_decline = False
        endurance_issue = False
        if len(form_trend) > 2:
            form_decline = form_trend[-1] < form_trend[0] * 0.9  # 10% decline
            endurance_issue = all(t < form_trend[0] * 0.95 for t in form_trend[-3:])  # Consistent decline
        
        summary = {
            "total_reps": self.total_reps,
            "form_percentage": round(form_percentage, 1),
            "avg_rep_time": round(avg_rep_time, 1),
            "time_consistency": round(time_consistency, 2),
            "form_consistency": round(form_consistency, 2)
        }
        
        # Generate comprehensive feedback
        feedback = []
        
        # Form quality analysis
        if form_percentage >= 90:
            feedback.append("Outstanding form maintained throughout your workout!")
        elif form_percentage >= 80:
            feedback.append("Very good form overall - focus on maintaining consistent technique.")
        elif form_percentage >= 70:
            feedback.append("Good foundation with room for improvement in form.")
        else:
            feedback.append("Focus on proper form before increasing intensity.")

        # Consistency insights
        if time_consistency < 0.3 and form_consistency < 5:
            feedback.append("Excellent workout rhythm and consistency!")
        elif time_consistency > 0.5:
            feedback.append("Try to maintain more consistent timing between reps.")
        elif form_consistency > 10:
            feedback.append("Work on maintaining consistent form quality across all reps.")
            
        # Fatigue and endurance insights
        if form_decline and endurance_issue:
            feedback.append("Notable form decline detected - consider shorter sets or lighter weights to maintain form.")
        elif form_decline:
            feedback.append("Slight form decline towards end - focus on maintaining technique throughout.")
        
        # Progress insights
        if len(self.rep_quality_scores) > 5:
            first_half = statistics.mean(self.rep_quality_scores[:len(self.rep_quality_scores)//2])
            second_half = statistics.mean(self.rep_quality_scores[len(self.rep_quality_scores)//2:])
            if second_half > first_half:
                feedback.append("Great improvement in form during your workout!")
            elif second_half < first_half * 0.9:
                feedback.append("Consider shorter sets to maintain form quality throughout.")
        
        # Rep timing insights
        if avg_rep_time < 1.5:
            feedback.append("Consider slowing down your reps for better control and muscle engagement.")
        elif avg_rep_time > 3.0:
            feedback.append("Your rep pace is quite controlled - great for form but consider a slightly faster tempo for optimal training.")
            
        # Join feedback with proper spacing
        summary["feedback"] = " ".join(feedback)
        return summary

form_analyzer = FormAnalyzer()

class ExerciseState:
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.feedback = "Ready to start!"
        self.form_feedback = "Analyzing form..."
        self.debug_info = ""
        self.lock = Lock()
        self.last_update = time.time()
        self.rep_history = []
        self.workout_active = False
        self.workout_summary = None
        self.rep_angles = []  # Store angles throughout rep
        self.movement_smoothness = []  # Track smooth motion
        self.concentric_start = time.time()
        self.eccentric_start = time.time()
        
    def analyze_movement_smoothness(self, current_angle):
        """Analyze if movement is smooth or jerky"""
        if len(self.rep_angles) > 2:
            angle_changes = np.diff(self.rep_angles[-3:])
            if max(abs(angle_changes)) > 15:  # Threshold for sudden movement
                return "Control the movement - keep it smooth"
        self.rep_angles.append(current_angle)
        return None

    def increment_counter(self):
        with self.lock:
            current_time = time.time()
            min_interval = 1.0  # Minimum interval between reps in seconds
            if current_time - self.last_update > min_interval:
                self.counter += 1
                self.last_update = current_time
                self.feedback = f"Good rep! Count: {self.counter}"

    def reset(self):
        with self.lock:
            self.counter = 0
            self.stage = "down"
            self.feedback = "Ready to start!"
            self.form_feedback = "Analyzing form..."
            self.debug_info = ""
            self.rep_history = []
            self.last_update = time.time()
            self.workout_active = False
            self.workout_summary = None

state = ExerciseState()

def initialize_camera():
    """Initialize the camera with proper settings."""
    global camera, camera_initialized
    if camera is not None:
        camera.release()
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)
    
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera_initialized = True
    return camera.isOpened()

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        if np.any(np.isnan([a, b, c])):
            return None
            
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians))
        return angle if angle <= 180 else 360 - angle
    except:
        return None

def process_frame(frame, exercise_type):
    """Process a single frame for pose detection and exercise counting with enhanced feedback."""
    global pose
    
    if pose is None:
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
    
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        form_correct = True
        exercise_feedback = []

        if results.pose_landmarks and state.workout_active:
            landmarks = results.pose_landmarks.landmark

            # Alignment Check
            alignment_feedback = check_body_alignment(landmarks)
            if alignment_feedback:
                exercise_feedback.extend(alignment_feedback)
            
            # Extract common landmarks
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # Track velocity for tempo feedback
            current_time = time.time()
            time_since_last_rep = current_time - state.last_update
            if time_since_last_rep < 1.0:
                exercise_feedback.append("Slow down - aim for 2 seconds per rep")
            elif time_since_last_rep > 3.0:
                exercise_feedback.append("Speed up slightly - maintain control")
            
            angles_to_analyze = []
            is_rep_complete = False
            
            if exercise_type == "bicep_curl":
                angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                if angle is not None:
                    state.debug_info = f"Bicep Curl - Angle: {angle:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.append(angle)
                    
                    # Enhanced range of motion tracking
                    if angle > 120:  # Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Now curl up with control"
                            exercise_feedback.append(feedback)
                            if time_since_last_rep < 1.0:
                                exercise_feedback.append("Slow down slightly")
                    elif angle < 90:  # Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()
                            is_rep_complete = True
                            feedback = "Great curl!"  # Add positive feedback
                            exercise_feedback.append(feedback)  
                            
                    # Check elbow position
                    if abs(l_elbow[0] - l_shoulder[0]) > 0.1:
                        exercise_feedback.append("Keep elbow close to body")
                            
            elif exercise_type == "lateral_raise":
                angle = calculate_angle(l_hip, l_shoulder, l_elbow)
                if angle is not None:
                    state.debug_info = f"Lateral Raise - Angle: {angle:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.append(angle)
                    
                    # Enhanced feedback for lateral raises
                    if angle < 25:  # Changed from 20 - Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Raise arms with control"
                            exercise_feedback.append(feedback)
                    elif angle > 65:  # Changed from 75 - Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()
                            is_rep_complete = True
                            feedback = "Good height!"
                            exercise_feedback.append(feedback)
                            
                    # Check for straight arms
                    elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    if elbow_angle and elbow_angle < 160:
                        exercise_feedback.append("Keep arms straighter")
                            
            elif exercise_type == "shoulder_press":
                angle1 = calculate_angle(l_hip, l_shoulder, l_elbow)
                angle2 = calculate_angle(l_shoulder, l_elbow, l_wrist)
                if angle1 is not None and angle2 is not None:
                    state.debug_info = f"Shoulder Press - Angles: {angle1:.1f}°, {angle2:.1f}°, Stage: {state.stage}"
                    angles_to_analyze.extend([angle1, angle2])

                    # Check for shoulder alignment during the movement
                    shoulder_level = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                    
                    # Adjusted thresholds for better rep detection
                    if angle1 < 80 and angle2 > 110:  # Relaxed from 60/140 - Down position
                        if state.stage != "down":
                            state.stage = "down"
                            feedback = "Press up with control"
                            exercise_feedback.append(feedback)

                    elif angle1 > 60 and angle2 < 100:  # Relaxed from 75/70 - Up position
                        if state.stage == "down":
                            state.stage = "up"
                            state.increment_counter()
                            is_rep_complete = True
                            if angle1 > 160:  # Full lockout
                                exercise_feedback.append("Great lockout!")
                            if shoulder_level > 0.08:
                                exercise_feedback.append("Keep shoulders level")

                    if angle1 > 150:  # Full lockout
                        exercise_feedback.append("Great lockout!")
                    
                    # Check shoulder alignment
                    shoulder_level = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
                    if shoulder_level > 0.08:
                        exercise_feedback.append("Keep shoulders level")
                    
        # Check for fatigue
        fatigue_feedback = form_analyzer.analyze_fatigue()
        if fatigue_feedback:
            exercise_feedback.append(fatigue_feedback)

        # Add semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (20, 150), (400, 550), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
        # Draw rep counter
        cv2.putText(image, f"Reps: {state.counter}", 
                      (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                      3.0, (0, 255, 0), 4)
            
        # Draw stage indicator
        cv2.putText(image, f"Stage: {state.stage.upper()}", 
                      (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                      2.5, (245, 117, 66), 3)
            
        # Draw form feedback if it exists
        if state.form_feedback:
            feedback_lines = state.form_feedback.split('|')
            y_position = 400
            for i, feedback_line in enumerate(feedback_lines):
                is_warning = any(word in feedback_line.lower() for word in 
                                   ['keep', 'adjust', 'maintain', 'slow', 'form declining'])
                enhance_visual_feedback(
                        image,
                        feedback_line.strip(),
                        (40, y_position + (i * 60)),
                        is_warning
                    )

        # Keep form analysis and wireframe in angles_to_analyze check
        if angles_to_analyze:
            form_correct, form_feedback = form_analyzer.analyze_form(angles_to_analyze, landmarks, is_rep_complete)
                
            if form_feedback:
                exercise_feedback.append(form_feedback)
                
            state.form_feedback = " | ".join(exercise_feedback) if exercise_feedback else "Good form!"
                
            wireframe_color = (0, 255, 0) if form_correct else (0, 0, 255)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=wireframe_color, thickness=2, circle_radius=2)
            )

        return image
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return frame

def generate_frames(exercise_type):
    """Generate frames for video streaming."""
    global camera, camera_initialized
    
    if not camera_initialized:
        if not initialize_camera():
            print("Error: Could not initialize camera.")
            return
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                if not initialize_camera():
                    print("Error: Camera disconnected and could not be reinitialized.")
                    break
            
            success, frame = camera.read()
            
            if not success:
                print("Error: Failed to read frame from camera.") 
                continue
            
            try:
                processed_frame = process_frame(frame, exercise_type)
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"Error generating frame: {str(e)}")
                continue

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed/<exercise_type>')
def video_feed(exercise_type):
    """Video streaming route."""
    return Response(generate_frames(exercise_type),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    """Get the current rep count and feedback."""
    if state.workout_summary:
        summary = state.workout_summary
    else:
        summary = {
            "total_reps": 0,
            "form_percentage": 0,
            "avg_rep_time": 0,
            "time_consistency": 0,
            "form_consistency": 0,
            "feedback": "No workout data available."
        }

    return jsonify({
        'count': state.counter,
        'feedback': state.feedback,
        'form_feedback': state.form_feedback,
        'debug_info': state.debug_info,
        'workout_active': state.workout_active,
        'workout_summary': summary
    })

@app.route('/start_workout')
def start_workout():
    """Start a new workout session."""
    state.reset()
    state.workout_active = True
    form_analyzer.__init__()  # Reset form analyzer
    return jsonify({'status': 'success'})

@app.route('/stop_workout')
def stop_workout():
    """Stop the current workout and get summary."""
    state.workout_active = False
    summary = form_analyzer.get_workout_summary()
    state.workout_summary = summary
    return jsonify({
        'status': 'success',
        'summary': summary
    })

@app.route('/reset_count')
def reset_count():
    """Reset the exercise counter."""
    state.reset()
    form_analyzer.__init__()
    return jsonify({'status': 'success'}) 

def cleanup():
    """Clean up resources."""
    global camera, pose
    if camera is not None:
        camera.release()
    if pose is not None:
        pose.close()

# Initialize camera when app starts
with app.app_context():
    initialize_camera()

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        cleanup()