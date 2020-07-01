FROM archlinux/base

RUN pacman -Sy
RUN pacman --noconfirm -S base-devel git gcc make llvm clang

RUN pacman --noconfirm -S pyenv
ARG PYTHON_VERSION=3.7.6
RUN CONFIGURE_OPTS='--enable-shared' pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION
RUN pyenv rehash
ENV PATH=/root/.pyenv/shims:$PATH

ARG HALIDE_REPO=https://github.com/halide/Halide.git
ARG HALIDE_COMMIT=95f154af71f1895119b2c57270ab8eaccb9cbef0
ENV HALIDE_PATH=/Halide
RUN git clone $HALIDE_REPO $HALIDE_PATH && cd $HALIDE_PATH && git checkout $HALIDE_COMMIT
RUN cd $HALIDE_PATH && make -j$(nproc)

RUN cd $HALIDE_PATH/python_bindings && pip install -r requirements.txt
RUN cd $HALIDE_PATH/python_bindings && make -j$(nproc)
ENV PYTHONPATH=$HALIDE_PATH/python_bindings/bin:$PYTHONPATH

RUN pacman --noconfirm -S libpng libjpeg-turbo  libtiff
RUN cd $HALIDE_PATH/apps/autoscheduler && make demo -j$(nproc)
ENV LD_LIBRARY_PATH=$HALIDE_PATH/apps/autoscheduler/bin:$LD_LIBRARY_PATH

WORKDIR /work