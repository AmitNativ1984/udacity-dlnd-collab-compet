ARG BASE_IMAGE
FROM $BASE_IMAGE as base

ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN echo ${DEBIAN_FRONTEND}

COPY python /udacity_drlnd/python
RUN pip3 install /udacity_drlnd/python
