FROM postgres:latest
COPY trgm.sql /docker-entrypoint-initdb.d
