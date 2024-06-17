from time import time_ns
import rtx_api_3_5 as rtx_api
import pyttsx3
import speech_recognition as sr
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Function to listen and recognize speech
def listen_and_recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return "no value"

# Function to listen and detect objects
def ObjectDetection(cap):
    Tot = ""
    sumResults = 0
    while(sumResults == 0):
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()

            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])
                sumResults = sumResults + 1
                Tot = Tot + "," + names[int(cls)]
        Tot =Tot[1:]
        print(Tot)
    return Tot,sumResults

def replace_with_object(text, object_to_insert):
    start = 0
    new_text = []
    index = text.find("this", start)

    # Append everything before "this"
    new_text.append(text[start:index])
    # Insert the object
    new_text.append(object_to_insert)
    print(object_to_insert)
    # Append everything after "this"
    new_text.append(text[index + len("this"):])

    return ''.join(new_text)

if __name__ == '__main__':
    port = 27973
    #message = ""
    #while("Jarvis" not in message):
    #   message = listen_and_recognize()
    #   print("answear: " + message)
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    #engine.say("Hello, Sir. loading virtual environment")
    #engine.runAndWait()
    model = YOLO("yolov8x-seg.pt")  # segmentation model
    names = model.model.names
    cap = cv2.VideoCapture(0)
    engine.say("Hello, Sir. How may I assist you today")
    engine.runAndWait()

    while(True):
        current_time_ns = time_ns()
        start_sec = current_time_ns / 1_000_000_000
        message = listen_and_recognize()
        Tot, sumResults = ObjectDetection(cap)
        if ("this" in message):
            words = Tot.split(",")
            if(words[0]=="person"):
                message=replace_with_object(message, words[1])
            else:
                message = replace_with_object(message, words[0])
        elif(message!="no value"):
            message = "you live in a world with " + str(
                sumResults) + " objects: " + Tot + ". and the preson ask " + message
        if(message!="no value"):
            print("sent: "+message)
            out = rtx_api.send_message(message, port)

            current_time_ns = time_ns()
            end_sec = current_time_ns / 1_000_000_000

            average_characters_in_token = 4
            took = end_sec - start_sec
            tokens_per_second = len(out) / took / average_characters_in_token
            print("tokens/s:", int(tokens_per_second))

            if (len(out)>100):
                out = rtx_api.send_message("Write single sentence sumary of: " + out, port)
            if "<br>" in out:
                parts = out.split("<br>")
                out=parts[0]
            print("answear: "+out)
            engine.say(out)
            engine.runAndWait()
            if 0xFF == ord('q'):
                break
            sumResults = 0

    cap.release()
    cv2.destroyAllWindows()