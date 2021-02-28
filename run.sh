#!/usr/bin/env bash

# This script runs the frontend and prediction server
FLASK_RUN_PORT=8080 FLASK_APP=arch_elements/model_deployment/app.py flask run --host=0.0.0.0