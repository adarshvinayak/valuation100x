# Stage 1: The "builder" stage with all build tools
FROM public.ecr.aws/lambda/python:3.11 as builder

# Copy only the requirements file
COPY common/requirements.txt ./

# Upgrade pip and install dependencies using pre-compiled wheels only
# This avoids compilation issues with older GCC in Amazon Linux 2
RUN pip install --upgrade pip && \
    pip install --target ./packages \
        --only-binary=all \
        --prefer-binary \
        -r ./requirements.txt

# ---
# Stage 2: The final, clean production image
FROM public.ecr.aws/lambda/python:3.11

# Copy only the installed packages from the "builder" stage
COPY --from=builder /var/task/packages/ ./

# Copy your application code
COPY common/ ${LAMBDA_TASK_ROOT}/
COPY worker_function/handler.py ${LAMBDA_TASK_ROOT}/

# Set the command to your handler
CMD [ "handler.handler" ]
