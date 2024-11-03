### attention : this docker file is optimized for build time and pull time 
### this is called multi-stage build to optimize the build time and the final image size. 

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

# Set environment variables
ENV GROQ_API_KEY='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'
ENV HUGGINGFACEHUB_API_TOKEN='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'

RUN dir

# Specify the command to run the application
ENTRYPOINT ["streamlit", "run", "/app/src/main.py"]

# Set default command-line arguments for the ENTRYPOINT
CMD ["--server.port=8080", "--server.address=0.0.0.0"]
