# Use an official CUDA runtime as a parent image
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04


# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip

# Copy the local requirements.txt file to the container at /app
COPY ./requirements.txt /tmp

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN adduser whiskey

USER whiskey


WORKDIR /workspace

CMD [ "/bin/bash" ]