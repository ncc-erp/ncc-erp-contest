FROM debian:sid

RUN echo deb http://deb.debian.org/debian/ stretch main > /etc/apt/sources.list.d/stretch.list && \
    echo deb http://security.debian.org/debian-security stretch/updates main >> /etc/apt/sources.list.d/stretch.list && \
    echo deb http://deb.debian.org/debian/ experimental main > /etc/apt/sources.list.d/experimental.list && \
    echo 'APT::Default-Release "sid";' > /etc/apt/apt.conf.d/99stretch && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl file gcc g++ python3-pip python3-dev python3-setuptools python3-wheel cython3 libseccomp-dev bzip2 gzip \
        python2 fp-compiler libxtst6 libffi8 tini $([ "$(arch)" = aarch64 ] && echo binutils-arm-linux-gnueabihf) && \
    apt-get install -y -t stretch --no-install-recommends openjdk-8-jdk-headless openjdk-8-jre-headless && \
    apt-get install -y -t experimental --no-install-recommends g++-11 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m judge

ENTRYPOINT ["/usr/bin/tini", "/code/run"]
