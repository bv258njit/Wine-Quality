# Base Java version
ARG OPENJDK_VERSION=17

# Use base image with Java
FROM eclipse-temurin:${OPENJDK_VERSION}-jre

# Configure Spark version
ARG SPARK_VERSION=3.3.1
ARG SPARK_EXTRAS=

# Container labels
LABEL org.opencontainers.image.title="Wine Quality Predictor with PySpark ${SPARK_VERSION}" \
      org.opencontainers.image.version=${SPARK_VERSION}

# Environment variables
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"
ENV JAVA_HOME="/opt/java/openjdk"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install dependencies and configure Miniconda
RUN set -ex && \
    apt-get update && \
    apt-get install -y curl bzip2 procps --no-install-recommends && \
    # Install Miniconda
    curl -s -L --url "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    # Configure conda
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda install pip -y && \
    conda clean -tipy && \
    # Install Python packages
    pip install --no-cache \
        pyspark[${SPARK_EXTRAS}]==${SPARK_VERSION} \
        numpy \
        pandas \
        pathlib \
        typing && \
    # Configure Spark
    SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py) && \
    echo "export SPARK_HOME=$(python3 /opt/miniconda3/bin/find_spark_home.py)" > /etc/profile.d/spark.sh && \
    # Create necessary directories
    mkdir -p /mlprog && \
    mkdir -p /home/ec2-user/models && \
    # Cleanup
    apt-get remove -y curl bzip2 && \
    apt-get autoremove -y && \
    apt-get clean


# Set working directory and filenames
ENV PROG_DIR /mlprog
ENV TRAIN_SCRIPT wine_train.py
ENV TEST_SCRIPT wine_test.py
ENV TRAIN_DATA TrainingDataset.csv
ENV TEST_DATA ValidationDataset.csv

# Set working directory
WORKDIR ${PROG_DIR}

# Copy Python scripts and datasets to the working directory
ADD ${TRAIN_SCRIPT} .
ADD ${TEST_SCRIPT} .
ADD ${TRAIN_DATA} .
ADD ${TEST_DATA} .

# Combine training and testing into a single entrypoint
RUN echo '#!/bin/bash\n' \
         'spark-submit wine_train.py\n' \
         'spark-submit wine_test.py\n' > /mlprog/run.sh && chmod +x /mlprog/run.sh

# Set the startup executable for Docker
ENTRYPOINT ["bash", "/mlprog/run.sh"]
