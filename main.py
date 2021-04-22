from fastapi import BackgroundTasks, FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse,RedirectResponse
from typing import List
import tempfile
import os
import io
import cv2
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from analysis2 import analyze
import threading
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
    )
def generate_html_response():
    html_content = """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


outputs = {}


def process(file_name,weights=[]):
    if(len(weights) > 2):
        bytes = analyze(file_name,color_weight=weights[0],motion_weight=weights[1],audio_weight=weights[2])
    else:
        bytes = analyze(file_name)
    outputs[file_name] = bytes

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.get("/tmp/{file_name}")
def get_status(file_name: str):
    print(outputs.keys())
    if('/tmp/'+file_name in outputs.keys()):
        res, im_png = cv2.imencode(".png",outputs['/tmp/'+file_name])
        del outputs['/tmp/'+file_name]
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    else:
        return {'message': "not_found"}

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}

@app.post("/uploadfiles/")
async def create_upload_files(background_tasks: BackgroundTasks, parameters: str=Form(...),file: UploadFile = File(...)):
    weights = [float(x)/10 for x in parameters]
    print(weights)
    tmp, file_name = tempfile.mkstemp()
    contents = await file.read()
    os.write(tmp,contents)
    os.close(tmp)
    background_tasks.add_task(process,file_name,weights)
    content = """
            <body>
            <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
            </form>
            </body>
                """
    return {'file_name': file_name}

@app.get("/")
async def main():
    content = """
            <body>
            <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
            </form>
            </body>
                """
    return HTMLResponse(content=content)

@app.get("/get_status")
def status():
    return state.get_state()



