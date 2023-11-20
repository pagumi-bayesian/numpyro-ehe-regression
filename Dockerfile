FROM python:3.11

RUN apt update -y \
    && apt upgrade -y

RUN curl -sL https://dev.nodesource.com/setup_12.x |bash - \
    && apt install -y --no-install-recommends \
    wget \
    git \
    vim \
    curl \
    make \
    cmake \
    nodejs \
    fonts-ipaexfont \
    fonts-noto-cjk \
    && apt autoremove -y \
    && apt clean \
    && rm -rf \
        /var/lib/apt/lists/* \
        /var/cache/apt/* \
        /usr/local/src/* \
        /tmp/*

# timezone
ENV TZ Asia/Tokyo

# install python library
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip
COPY matplotlibrc /root/.config/matplotlib/matplotlibrc

EXPOSE 8888

# install jupyter lab
RUN pip3 install --upgrade --no-cache-dir jupyterlab

WORKDIR /home/work/
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
