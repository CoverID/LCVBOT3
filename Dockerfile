# Use Python 3.12.6 from the official Docker Hub image
FROM python:3.12.6

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the app's port
EXPOSE 5000

# Command to run the app
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "5000", "--server.address", "0.0.0.0"]
