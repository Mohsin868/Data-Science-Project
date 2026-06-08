from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import base64
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this_2024'
app.config['SESSION_TYPE'] = 'filesystem'

plt.switch_backend('Agg')

screen_time_model = None
productivity_model = None
label_encoders = {}
scaler = None
feature_columns = []

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def load_and_train_models():
    global screen_time_model, productivity_model, label_encoders, scaler, feature_columns
    
    df = pd.read_csv('data.csv')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Environment', 'Work Strategy'])
    
    screen_time_map = {
        'Less than 2': 1, '2–4': 2, '4–6': 3, 
        '6–8': 4, '8-10': 5, 'More than 10': 6
    }
    df['Screen_Time_Numeric'] = df['Average Screen Time'].map(screen_time_map)
    
    productivity_map = {
        'Extremely productive, i efficiently complete my tasks': 2,
        'Moderately productive': 1,
        'Unproductive, i might not have completed the task and got carried away': 0
    }
    df['Productivity_Score'] = df['Productivity'].map(productivity_map)
    
    categorical_features = ['Age Group', 'Gender', 'Education Level', 'Occupation', 
                           'Device', 'Screen Activity', 'App Category', 
                           'Screen Time Period', 'Environment', 'Notification Handling']
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].fillna('Unknown').astype(str)
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    feature_columns = [col + '_encoded' for col in categorical_features]
    
    attention_map = {
        'Less than 10 minutes': 1, 
        '10–30 minutes': 2, 
        '30–60 minutes': 3, 
        'More than 1 hour': 4
    }
    df_encoded['Attention_Span_Numeric'] = df['Attention Span'].map(attention_map).fillna(2)
    feature_columns.append('Attention_Span_Numeric')
    
    df_encoded['Age_Attention'] = df_encoded['Age Group_encoded'] * df_encoded['Attention_Span_Numeric']
    feature_columns.append('Age_Attention')
    
    df_encoded['Focus_Factor'] = df_encoded['Notification Handling_encoded'] * df_encoded['Attention_Span_Numeric']
    feature_columns.append('Focus_Factor')
    
    X = df_encoded[feature_columns].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_screen = df_encoded['Screen_Time_Numeric'].fillna(3)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_screen, test_size=0.15, random_state=42)
    
    screen_time_model = ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_split=2, random_state=42)
    screen_time_model.fit(X_train, y_train)
    
    y_productivity = df_encoded['Productivity_Score'].fillna(1)
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_scaled, y_productivity, test_size=0.2, random_state=42)
    
    productivity_model = ExtraTreesClassifier(n_estimators=80, max_depth=10, min_samples_split=4, class_weight='balanced', random_state=42)
    productivity_model.fit(X_train_p, y_train_p)
    
    prod_pred = productivity_model.predict(X_test_p)
    prod_accuracy = accuracy_score(y_test_p, prod_pred)
    
    if prod_accuracy < 0.8:
        productivity_model.fit(X_scaled, y_productivity)

    return screen_time_model, productivity_model

def generate_dynamic_background():
    img = Image.open('background.png')
    return img

def predict_user_input(user_data):
    global screen_time_model, productivity_model, label_encoders, scaler, feature_columns
    
    encoded_features = []
    
    for col in ['Age Group', 'Gender', 'Education Level', 'Occupation', 
                'Device', 'Screen Activity', 'App Category', 
                'Screen Time Period', 'Environment', 'Notification Handling']:
        if col in label_encoders:
            try:
                val = str(user_data.get(col, 'Unknown'))
                encoded_val = label_encoders[col].transform([val])[0]
                encoded_features.append(encoded_val)
            except ValueError:
                encoded_features.append(0)
        else:
            encoded_features.append(0)
    
    attention_map = {
        'Less than 10 minutes': 1, 
        '10–30 minutes': 2, 
        '30–60 minutes': 3, 
        'More than 1 hour': 4
    }
    attention_val = attention_map.get(user_data.get('Attention Span', '10–30 minutes'), 2)
    encoded_features.append(attention_val)
    
    age_encoded = encoded_features[0]
    encoded_features.append(age_encoded * attention_val)
    
    notif_encoded = encoded_features[9]
    encoded_features.append(notif_encoded * attention_val)
    
    features_scaled = scaler.transform([encoded_features])
    
    predicted_screen_numeric = screen_time_model.predict(features_scaled)[0]
    predicted_productivity_class = productivity_model.predict(features_scaled)[0]
    productivity_proba = productivity_model.predict_proba(features_scaled)[0]
    
    screen_time_map_reverse = {
        1: 'Less than 2 hours', 2: '2-4 hours', 3: '4-6 hours',
        4: '6-8 hours', 5: '8-10 hours', 6: 'More than 10 hours'
    }
    
    screen_category_num = int(round(predicted_screen_numeric))
    screen_category_num = max(1, min(6, screen_category_num))
    predicted_screen_time = screen_time_map_reverse[screen_category_num]
    
    productivity_labels = {0: 'Unproductive', 1: 'Moderately Productive', 2: 'Highly Productive'}
    predicted_productivity = productivity_labels[predicted_productivity_class]
    
    confidence = float(max(productivity_proba) * 100)
    
    return {
        'predicted_screen_time': str(predicted_screen_time),
        'predicted_productivity': str(predicted_productivity),
        'confidence': float(round(confidence, 1)),
        'screen_time_numeric': float(round(predicted_screen_numeric, 1)),
        'productivity_score': int(predicted_productivity_class)
    }

def generate_prediction_visualization(user_input, prediction):
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    screen_categories = ['<2h', '2-4h', '4-6h', '6-8h', '8-10h', '>10h']
    current_value = prediction['screen_time_numeric']
    
    colors_gauge = ['#2ecc71', '#2ecc71', '#2ecc71', '#f39c12', '#e74c3c', '#e74c3c']
    bars = ax1.bar(screen_categories, [1]*len(screen_categories), color=colors_gauge, alpha=0.3, edgecolor='white')
    
    bar_index = min(int(current_value)-1, 5)
    if bar_index >= 0:
        bars[bar_index].set_alpha(0.8)
        bars[bar_index].set_color(colors_gauge[bar_index])
    
    ax1.set_ylabel('Predicted Category', color='white')
    ax1.set_title(f'Predicted Screen Time: {prediction["predicted_screen_time"]}', 
                  fontsize=12, fontweight='bold', color='white')
    ax1.set_ylim(0, 1.2)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')
    
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    productivity_levels = ['Unproductive', 'Moderately\nProductive', 'Highly\nProductive']
    current_prod = prediction['productivity_score']
    
    colors_prod = ['#e74c3c', '#f39c12', '#2ecc71']
    
    for i, (level, color) in enumerate(zip(productivity_levels, colors_prod)):
        if i == current_prod:
            ax2.barh(level, 100, color=color, alpha=0.8, edgecolor='white', linewidth=2)
            ax2.text(105, level, f'{prediction["confidence"]:.0f}%', 
                    va='center', fontweight='bold', fontsize=10, color='white')
        else:
            ax2.barh(level, 100, color=color, alpha=0.3, edgecolor='white')
    
    ax2.set_xlim(0, 120)
    ax2.set_xlabel('Confidence (%)', color='white')
    ax2.set_title(f'Predicted Productivity: {prediction["predicted_productivity"]}', 
                  fontsize=12, fontweight='bold', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')
    
    plt.suptitle('Your Personalized Prediction Results', fontsize=14, fontweight='bold', 
                 color='white', y=1.02)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close('all')
    
    return img_base64

screen_time_model, productivity_model = load_and_train_models()

@app.route('/background.png')
def serve_background():
    img = generate_dynamic_background()
    img_io = io.BytesIO()
    img.save(img_io, 'PNG', quality=85)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html', 
                         current_year=datetime.now().year,
                         age_groups=['Below 18', '18–24', '25–34', '35–44', '45 and above'],
                         genders=['Male', 'Female'],
                         education_levels=['High school or below', 'Undergraduate', 'Graduate'],
                         occupations=['Student', 'Professional'],
                         devices=['Smartphone', 'Laptop/PC', 'Tablet', 'Television'],
                         screen_activities=['Entertainment (gaming, streaming, social media, etc.)', 'Academic/Work-related'],
                         app_categories=['Social Media', 'Streaming', 'Productivity', 'Gaming', 'Messaging'],
                         time_periods=['Morning (6 AM–12 PM)', 'Afternoon (12 PM–6 PM)', 'Evening (6 PM–10 PM)', 'Late night (10 PM–6 AM)'],
                         environments=['Quite workplace', 'I can work in any environment', 'Collaborative/team setting', 'Background noise/music'],
                         notification_handlings=['Turn off notifications altogether', 'Ignore them until my task is completed', 'Check them briefly and resume my work', 'Spend time interacting with the notifications'],
                         attention_spans=['Less than 10 minutes', '10–30 minutes', '30–60 minutes', 'More than 1 hour'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            'Age Group': request.form.get('age_group'),
            'Gender': request.form.get('gender'),
            'Education Level': request.form.get('education_level'),
            'Occupation': request.form.get('occupation'),
            'Device': request.form.get('device'),
            'Screen Activity': request.form.get('screen_activity'),
            'App Category': request.form.get('app_category'),
            'Screen Time Period': request.form.get('time_period'),
            'Environment': request.form.get('environment'),
            'Notification Handling': request.form.get('notification_handling'),
            'Attention Span': request.form.get('attention_span')
        }
        
        prediction = predict_user_input(user_input)
        viz_img = generate_prediction_visualization(user_input, prediction)
        
        session_prediction = {
            'predicted_screen_time': prediction['predicted_screen_time'],
            'predicted_productivity': prediction['predicted_productivity'],
            'confidence': prediction['confidence'],
            'screen_time_numeric': prediction['screen_time_numeric'],
            'productivity_score': prediction['productivity_score']
        }
        
        session['prediction'] = session_prediction
        session['viz_img'] = viz_img
        session['user_input'] = user_input
        
        return render_template('result.html', 
                             prediction=prediction,
                             viz_img=viz_img,
                             user_input=user_input)
                             
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        prediction = predict_user_input(data)
        return jsonify({'success': True, 'prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)