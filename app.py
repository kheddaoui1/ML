# File: app.py
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, has_request_context, jsonify
import xgboost as xgb
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'solar_uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Default data file name
DEFAULT_DATA_FILE = 'tiaret_future_weather.csv'

# Load both models once
model_efficiency = xgb.XGBRegressor()
model_efficiency.load_model("model/solar_model.json")

model_egrid = xgb.XGBRegressor()
model_egrid.load_model("model/solar_model_egrid.json")

def load_data_from_folder():
    data_folder = os.path.join(app.root_path, 'data')
    data_files = {}
    
    # Look for CSV files in the data folder
    for filename in os.listdir(data_folder):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(data_folder, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                data_files[filename] = {
                    'path': file_path,
                    'columns': df.columns.tolist(),
                    'preview': df.head().to_dict('records')
                }
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return data_files

def get_default_data_path():
    """Get the path to the default data file in the data directory."""
    data_dir = Path(__file__).parent / 'data'
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        return str(csv_files[0])  # Return the first CSV file found
    return None

@app.route('/')
def index():
    # Load available data files
    data_files = load_data_from_folder()
    default_data_available = get_default_data_path() is not None
    return render_template('index.html', data_files=data_files, default_data_available=default_data_available)

from flask import session
import os
from werkzeug.utils import secure_filename

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Check if we should use the default data file
        use_default = request.args.get('use_default') == 'true' or request.form.get('use_default') == 'true'
        
        if use_default:
            temp_filepath = get_default_data_path()
            if not temp_filepath or not os.path.exists(temp_filepath):
                return render_template('index.html', error="Default data file not found in the data folder")
            prediction_type = request.args.get('prediction_type', 'efficiency')
            session['current_file'] = temp_filepath
            session['prediction_type'] = prediction_type
            file = open(temp_filepath, 'rb')
        elif request.method == 'POST':
            prediction_type = request.form.get('prediction_type', 'efficiency')
            
            # Check if a file from the data folder was selected
            selected_file = request.form.get('selected_file')
            
            # Initialize file variable
            file = None
            
            # If no file selected or uploaded, use the default file
            if not selected_file and ('file' not in request.files or not request.files['file'].filename):
                default_file = os.path.join(app.root_path, 'data', DEFAULT_DATA_FILE)
                if os.path.exists(default_file):
                    selected_file = DEFAULT_DATA_FILE
                    file_path = os.path.join('data', selected_file)
                    session['current_file'] = file_path
                    session['prediction_type'] = prediction_type
                    temp_filepath = os.path.join(app.root_path, file_path)  # Use full path
                    app.logger.info(f"Using default data file: {DEFAULT_DATA_FILE}")
                    file = open(temp_filepath, 'rb')
                else:
                    data_files = load_data_from_folder()
                    return render_template('index.html', 
                                       error=f"Default data file {DEFAULT_DATA_FILE} not found.",
                                       data_files=data_files)
            # If a file from data folder was selected
            elif selected_file:
                file_path = os.path.join('data', selected_file)
                full_path = os.path.join(app.root_path, file_path)
                if not os.path.exists(full_path):
                    data_files = load_data_from_folder()
                    return render_template('index.html', 
                                       error="Selected file not found in data folder.",
                                       data_files=data_files)
                session['current_file'] = file_path
                session['prediction_type'] = prediction_type
                temp_filepath = full_path
                file = open(temp_filepath, 'rb')
            # Handle file upload
            else:
                file = request.files['file']
                if file.filename == '':
                    data_files = load_data_from_folder()
                    return render_template('index.html', 
                                       error="No file selected",
                                       data_files=data_files)
                
                if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
                    data_files = load_data_from_folder()
                    return render_template('index.html', 
                                       error="Invalid file type. Please upload a CSV or Excel file.",
                                       data_files=data_files)
                
                # Save the uploaded file temporarily
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filename = secure_filename(file.filename)
                temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_filepath)
                session['current_file'] = temp_filepath
                session['prediction_type'] = prediction_type
                session['is_temp_file'] = True  # Mark as temporary file for cleanup
                file = open(temp_filepath, 'rb')  # Open the file for processing
            
        else:  # GET request (view toggle)
            prediction_type = request.args.get('prediction_type', 'efficiency')
            
            # Get the file path from session if it exists
            if 'current_file' in session:
                temp_filepath = session['current_file']
                if not os.path.exists(temp_filepath):
                    return redirect(url_for('index'))
                file = open(temp_filepath, 'rb')
                session['prediction_type'] = prediction_type
            else:
                return redirect(url_for('index'))

        if not file:
            return render_template('index.html', error="No file uploaded")
            
        # Ensure prediction_type is valid
        prediction_type = prediction_type.lower()
        if prediction_type not in ['efficiency', 'egrid']:
            prediction_type = 'efficiency'  # Default to efficiency if invalid

        # Read the file based on its extension
        if temp_filepath.lower().endswith('.csv'):
            df = pd.read_csv(temp_filepath)
        else:  # Excel file
            df = pd.read_excel(temp_filepath)
            
        # Check if the file is empty
        if df.empty:
            return render_template('index.html', error="The uploaded file is empty")
            
        # Check for required columns
        required_cols = ["T_Amb", "GlobHor", "WindVel"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return render_template('index.html', error=f"Missing required columns: {', '.join(missing_cols)}")
            
    except pd.errors.EmptyDataError:
        return render_template('index.html', error="The uploaded file is empty or not in the correct format")
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return render_template('index.html', error=f"Error processing file: {str(e)}")
    required_cols = ["T_Amb", "GlobHor", "WindVel"]
    if not all(col in df.columns for col in required_cols):
        return "Missing required columns: T_Amb, GlobHor, WindVel", 400

    # Select the model and set display parameters based on prediction type
    if prediction_type == 'efficiency':
        model = model_efficiency
        y_label = "Efficiency"
        display_type = "Efficiency"
        line_color = "blue"
    else: # prediction_type == 'egrid'
        model = model_egrid
        y_label = "E_Grid"
        display_type = "E_Grid"
        line_color = "red"

    predictions = model.predict(df[required_cols])
    df["Prediction"] = predictions
    
    # Calculate statistics
    max_value = predictions.max()
    min_value = predictions.min()
    mean_value = predictions.mean()
    
    # Add units for E8 grid predictions
    if prediction_type == 'egrid':
        max_value = f"{max_value:.2f} KWh/day"
        min_value = f"{min_value:.2f} KWh/day"
        mean_value = f"{mean_value:.2f} KWh/day"
    
    # Get dates for max and min values
    if "Date" in df.columns:
        max_date = df.loc[predictions.argmax(), "Date"]
        min_date = df.loc[predictions.argmin(), "Date"]
    else:
        max_date = min_date = "N/A (No date information)"

    # Use Date if available, else index for x-axis
    x_axis = df["Date"] if "Date" in df.columns else df.index

    # Create frames for animation
    frames = []
    for i in range(1, len(df) + 1):
        frame = go.Frame(
            data=[go.Scatter(x=x_axis[:i], y=predictions[:i], mode='lines+markers',
                             line=dict(color=line_color, width=2),
                             marker=dict(size=5))]
        )
        frames.append(frame)

    # Initial data - first point
    init_x = x_axis[:1]
    init_y = predictions[:1]

    # Create plot with animation controls
    fig = go.Figure(
        data=[go.Scatter(x=init_x, y=init_y, mode='lines+markers',
                         line=dict(color=line_color, width=2),
                         marker=dict(size=5))],
        layout=go.Layout(
            title=f"Predicted {y_label} Over Time",
            xaxis=dict(title="Date", showgrid=True, gridcolor='#e0e0e0'),
            yaxis=dict(title=y_label, showgrid=True, gridcolor='#e0e0e0'),
            plot_bgcolor='#f9f9f9',
            paper_bgcolor='#ffffff',
            font=dict(family="Arial, sans-serif", size=12, color="#333"),
            margin=dict(l=50, r=50, b=100, t=50, pad=4),
        ),
        frames=frames
    )

    fig.update_layout(
        title=f"{y_label} Over Time",
        xaxis_title="Date",
        yaxis_title=y_label,
        showlegend=True,
        paper_bgcolor='#ffffff',
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        margin=dict(l=50, r=50, b=100, t=50, pad=4),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Fast forward",
                     method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                  "mode": "immediate"}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
            ],
            showactive=False,
            x=0.5,
            y=-0.5,
            xanchor="center",
            yanchor="top",
            direction="left",
            pad=dict(r=10, l=10)
        )]
    )

    graph_html = fig.to_html(full_html=False)
    
    # Convert frames to a serializable format
    serializable_frames = []
    for frame in fig.frames:
        serializable_frames.append({
            'data': [{
                'y': trace.y.tolist() if hasattr(trace, 'y') else None,
                'type': trace.type if hasattr(trace, 'type') else None
            } for trace in frame.data],
            'name': frame.name,
            'layout': frame.layout.to_plotly_json() if hasattr(frame.layout, 'to_plotly_json') else {}
        })

    # If this is a GET request (toggle), we need to preserve the file
    file_param = f"?file={file.filename}" if hasattr(file, 'filename') else ''
    
    return render_template('result.html',
                         plot=fig.to_html(full_html=False, include_plotlyjs='cdn'),
                         prediction_type=display_type,
                         prediction_type_param=prediction_type,  # Lowercase for URL params
                         max_value=max_value,
                         min_value=min_value,
                         mean_value=mean_value,
                         max_date=max_date,
                         min_date=min_date,
                         file_param=file_param,
                         line_color=line_color,
                         frames=serializable_frames,
                         prediction_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.teardown_appcontext
def cleanup_temp_files(exception=None):
    if not has_request_context():
        return
    
    # Close any open file handles
    if 'file_handle' in session:
        try:
            file_handle = session['file_handle']
            if hasattr(file_handle, 'close') and not file_handle.closed:
                file_handle.close()
        except Exception as e:
            app.logger.error(f"Error closing file handle: {e}")
        finally:
            session.pop('file_handle', None)
    
    # Clean up temporary uploaded files
    if 'current_file' in session and session.get('is_temp_file', False):
        try:
            filepath = session['current_file']
            # Only remove files from the temporary upload directory
            if os.path.exists(filepath) and filepath.startswith(app.config['UPLOAD_FOLDER']):
                os.remove(filepath)
                app.logger.info(f"Cleaned up temporary file: {filepath}")
                
                # Remove the directory if it's empty
                upload_dir = os.path.dirname(filepath)
                if (os.path.exists(upload_dir) and 
                    upload_dir.startswith(app.config['UPLOAD_FOLDER']) and 
                    not os.listdir(upload_dir)):
                    os.rmdir(upload_dir)
                    app.logger.info(f"Removed empty directory: {upload_dir}")
        except Exception as e:
            app.logger.error(f"Error removing temp file: {e}")
    
    # Clear session variables
    for key in ['current_file', 'is_temp_file', 'file_handle']:
        if key in session:
            session.pop(key)

if __name__ == "__main__":
    app.run(debug=True)
