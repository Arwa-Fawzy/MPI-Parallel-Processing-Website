from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import subprocess
import time
import uuid
from werkzeug.utils import secure_filename
import sys
from mpi_scripts.linear_regression_mpi import run_linear_regression_mpi



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

TASK_SCRIPTS = {
    'sorting': 'mpi_scripts/sort_mpi.py',
    'file_process': 'mpi_scripts/file_process_mpi.py',
    'image_process': 'mpi_scripts/image_process_mpi.py',
    'ml_training': 'mpi_scripts/linear_regression_mpi.py',
    'text_search': 'mpi_scripts/text_search_mpi.py',
    'stats': 'mpi_scripts/stats_mpi.py',
    'matrix_mult': 'mpi_scripts/matrix_mult_mpi.py'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_task():
    task = request.form.get('task')
    num_procs = int(request.form.get('num_procs', 2))


    if task not in TASK_SCRIPTS:
        return "Invalid task selected", 400

    script_path = TASK_SCRIPTS[task]
    input_args = []
    upload_path = None

    try:
        if task == 'sorting':
            numbers_str = request.form.get('numbers', '').strip()
            if not numbers_str:
                return "Please provide a comma-separated list of numbers", 400

            numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip().isdigit()]
            if not numbers:
                return "No valid numbers found in the input", 400

            input_args = [','.join(map(str, numbers))]

        elif task == 'file_process':
            file = request.files.get('file')
            if not file or not file.filename.endswith('.txt'):
                return "Please upload a valid .txt file", 400
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            input_args = [upload_path]

        elif task == 'image_process':
            file = request.files.get('image')
            filter_type = request.form.get('filter_type')
            if not file or not (file.filename.endswith('.png') or file.filename.endswith('.jpg') or file.filename.endswith('.jpeg')):
                return "Please upload a valid image (.png, .jpg)", 400
            if filter_type not in ['grayscale', 'blur']:
                return "Invalid filter type selected", 400
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            input_args = [upload_path, filter_type]
        
        
        
        elif task == 'ml_training':
            file = request.files.get('csvfile')
            if not file or not file.filename.endswith('.csv'):
                return "Please upload a valid CSV file", 400
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            start_time = time.time()
            try:
                # Run the MPI ML training function directly
                result = run_linear_regression_mpi(upload_path)
                duration = time.time() - start_time
                output = f"Output: Trained Coefficients (including intercept): {result}"
            except Exception as e:
                output = f"Error during ML training: {str(e)}"
                duration = None

            return render_template('result.html', task=task, output=output, duration=duration, num_procs=num_procs)



        elif task == 'text_search':
            file = request.files.get('textfile')
            keyword = request.form.get('keyword')
            if not file or not file.filename.endswith('.txt'):
                return "Please upload a valid text file", 400
            if not keyword:
                return "Please enter a keyword for search", 400
            filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            input_args = [upload_path, keyword]

        elif task == 'stats':
            file = request.files.get('csvfile')
            

            # Generate a safe unique filename to avoid conflicts
            filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Save the uploaded file to the server
            file.save(upload_path)

            # Prepare arguments for the stats MPI task, passing the uploaded CSV path
            input_args = [upload_path]


        elif task == 'matrix_mult':
            matrix_a = request.form.get('matrix_a')
            matrix_b = request.form.get('matrix_b')
            if not matrix_a or not matrix_b:
                return "Please enter both matrices", 400
            matrix_a_file = os.path.join(app.config['UPLOAD_FOLDER'], f'matrix_a_{uuid.uuid4().hex}.csv')
            matrix_b_file = os.path.join(app.config['UPLOAD_FOLDER'], f'matrix_b_{uuid.uuid4().hex}.csv')
            with open(matrix_a_file, 'w') as f:
                f.write(matrix_a.strip())
            with open(matrix_b_file, 'w') as f:
                f.write(matrix_b.strip())
            input_args = [matrix_a_file, matrix_b_file]

        else:
            return "Unsupported task", 400

        command = ['mpiexec', '-n', str(num_procs), 'python', script_path] + input_args
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        output = result.stdout

    except subprocess.CalledProcessError as e:
        output = f"Error during MPI execution:\n{e.stderr}"
        duration = None

    return render_template('result.html', task=task, output=output, duration=duration, num_procs=num_procs)


if __name__ == '__main__':
    app.run(debug=True)
