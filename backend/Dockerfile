FROM python:3-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Set the working directory in the container
WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN playwright install --with-deps chromium
# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 3100

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3100", "--workers", "4"]