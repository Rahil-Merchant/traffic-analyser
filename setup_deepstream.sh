#!/usr/bin/env bash
# setup_deepstream.sh
# DeepStream 7.1 Environment Setup for WSL Ubuntu 22.04

set -e

echo "=== Updating system packages ==="
apt-get update -y
apt-get upgrade -y

echo "=== Installing GStreamer & DeepStream dependencies ==="
apt-get install -y --no-install-recommends \
    python3-gi python3-gi-cairo python3-gst-1.0 python3-pip \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav libglib2.0-dev libgirepository1.0-dev \
    libcairo2-dev pkg-config

echo "=== Installing DeepStream 7.1 ==="
# The base Docker image already includes DeepStream; skip .deb install when using Docker.
# If you ever install on host, uncomment next lines:
# wget -q https://developer.nvidia.com/downloads/deepstream-7.1_7.1.0-1_amd64.deb -O /tmp/deepstream.deb
# apt-get install -y /tmp/deepstream.deb
# rm /tmp/deepstream.deb

echo "=== Installing Python bindings & dependencies ==="
# pip3 install --no-cache-dir nvidia-pyindex nvidia-deepstream
pip3 install --no-cache-dir -r requirements_deepstream.txt

echo "=== Setting environment variables ==="
echo 'export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib/' >> ~/.bashrc

echo "=== Creating output directories ==="
mkdir -p output/videos output/reports output/static_previews

echo "=== Setup complete. Run 'source ~/.bashrc' and then start your application. ==="
