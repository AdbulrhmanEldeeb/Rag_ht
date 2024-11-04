### attention : this docker file is optimized for build time and pull time 
### this is called multi-stage build to optimize the build time and the final image size. 
# Stop all running containers
# docker stop $(docker ps -aq)

# # Remove all containers
# docker rm $(docker ps -aq)

# # Remove all images
# docker rmi $(docker images -q)

# # Optional: Clean up everything (stopped containers, unused networks, etc.)
# docker system prune -a

# Use a slim version of Python 3.12 as the base image
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt && pip show streamlit

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app /app
COPY . .

# Expose port 8080 for the application
EXPOSE 8080



# Set default command-line arguments for the ENTRYPOINT
CMD ["streamlit", "run", "/app/src/main.py","--server.port=8080", "--server.address=0.0.0.0"]
