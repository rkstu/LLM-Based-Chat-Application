# Use a base image with Python
FROM python:3.12-slim

# Install git and other dependencies required by some Python packages
RUN apt-get update && apt-get install -y git && apt-get clean

# Create a non-root user and set the home directory
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Ensure the application is owned by the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the port that the app will run on (OpenShift default is 8080, we will use the same for flexibility)
EXPOSE 8080

# Set the environment variable to tell OpenShift what port to use
ENV PORT 8080

# Command to run the app
CMD ["python", "app.py"]
