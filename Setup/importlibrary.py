from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Response
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse, UJSONResponse, FileResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel
from configparser import ConfigParser
import psycopg2

import uvicorn
import bcrypt
import time
import jwt
import pytz
from pytz import timezone
import pandas as pd
import logging
import time
from time import gmtime, strftime,localtime
from datetime import datetime, timedelta
from pathlib import Path
import base64
import os

import ArcFace
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
