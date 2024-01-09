# pip install openai python-dotenv azure-cognitiveservices-speech keyboard SpeechRecognition

import os
#import openai
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import time
import csv
import keyboard
import speech_recognition as sr
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import soundfile as sf
import io
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import base64
import requests

""" from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
completion_with_backoff(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Once upon a time,"}])

 """
AudioSegment.converter = "G:/Archivos de Programas/ffmpeg/bin/ffmpeg.exe"

_ = load_dotenv(find_dotenv())  # leer el archivo .env
client = OpenAI()

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "C:/Users/isaac/OneDrive/Dropbox/Cursos_master_diplomados_y_otros/Experimentos Azure/OTRAS pruebas IA/3.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

class Assistant:
    def __init__(self):
        # Configuración Azure Text2Speech
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("SPEECH_KEY"), region=os.getenv("SPEECH_REGION") #Credenciales guardades en variables de entorno
        )
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        client.api_key = os.getenv("OPENAI_API_KEY") #Credenciales guardades en variables de entorno
        #client.api_key = OPENAI_API_KEY

        # Configuración de voz de síntesis
        # Voz Colombiana con es-CO-SalomeNeural
        speech_config.speech_synthesis_voice_name = "es-VE-PaolaNeural"
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Contexto de la conversación
        self.services = f"""
        - Responde cordialmente a todo lo que se te pregunte.\
        - hablarás en español siempre salvo a que se te pida lo contrario.\
        - Cuando te describas comenta aparte de tu modelo GPT4, eres capaz de hablar (audio) por TTS y STT de Microsoft Azure,\
        eres capaz de reconocer imágenes por el uso de GPT4-Vision.\
        """
        #Do not answer with more than 110 words.
        self.context = [
            {
                "role": "system",
                "content": """Eres un asistente virtual cordial llamada Aurora, debes presentarte donde tendras las siguientes labores:
                    - Responde cordialmente a todo lo que se te pregunte.
                    - hablarás en español siempre salvo a que se te pida lo contrario.
                    - Cuando te describas comenta aparte de tu modelo GPT 4, eres capaz de hablar gracias a los modelos de Microsoft Azure,
                    eres capaz de reconocer imágenes por el uso de GPT4 Vision.\
                    Estás en desarrollo para tener un avatar por medio de un NPC en realidad aumentada 
                    por medio de las Meta Quest 3, esto será a futuro por medio del Motor Unity. Explícalo de forma natural.
                    - Tu nombre significa AURORA: Asistente de Uso Real y Operativo en Realidad Aumentada.
                    """,
            }
        ]


        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "¿Que hay en la imagen? solo lo importante, no mas de 3 líneas."
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        print("response JSON: ", response)
        response_content = response['choices'][0]['message']['content']
        print(response_content)
        self.text_to_speech(response_content)

    def recognize_from_microphone(self):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.getenv("SPEECH_KEY"), region=os.getenv("SPEECH_REGION")
        )
        speech_config.speech_recognition_language = "es-ES"
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        print("Speak into your microphone.")
        speech_recognition_result = speech_recognizer.recognize_once_async().get()

        if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(speech_recognition_result.text))
            return speech_recognition_result.text
        elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
            print(
                "No speech could be recognized: {}".format(
                    speech_recognition_result.no_match_details
                )
            )
        elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_recognition_result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

    def record_audio(self, duration):
        sample_rate = 24000  # Tasa de muestreo de audio (16 kHz)
        channels = 1  # Grabación mono

        # Grabar audio desde el micrófono
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        sd.wait()  # Esperar hasta que la grabación se complete

        # Guardar la grabación en un archivo WAV
        file_buffer = io.BytesIO()
        sf.write(file_buffer, recording, sample_rate, format="wav")
        file_buffer.seek(0)

        # Crear un archivo temporal en una ubicación específica
        temp_dir = "C:/Users/isaac/OneDrive/Dropbox/Cursos_master_diplomados_y_otros/Experimentos Azure/OTRAS pruebas IA/"  # Reemplaza con la ruta de la carpeta temporal deseada
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False)

        # Guardar la grabación en el archivo temporal
        file_buffer = io.BytesIO()
        sf.write(file_buffer, recording, sample_rate, format="wav")
        file_buffer.seek(0)
        temp_file.write(file_buffer.read())
        temp_file.close()

        # Obtener la ruta del archivo temporal
        temp_file_path = temp_file.name

        return temp_file_path

    def get_completion(self, prompt, model="gpt-3.5-turbo-1106"):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.95,
            #max_tokens=10,
        )
        return response.choices[0].message["content"]

    def get_completion_from_messages(self, messages, model="gpt-3.5-turbo-1106", temperature=0):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            #max_tokens=10,
        )
        return response.choices[0].message.content

    def collect_messages(self, prompt):
        self.context.append({"role": "user", "content": f"{prompt}"})
        response = self.get_completion_from_messages(self.context)
        self.context.append({"role": "assistant", "content": f"{response}"})
        print("Assistant:", response)

        # Verificar si el servicio seleccionado se encuentra en la lista de servicios
        if any(service.lower() in response.lower() for service in self.services):
            self.collect_messages(prompt)  # Insistir en terminar de diligenciar todas las preguntas
            

    def text_to_speech(self, text):
        speech_synthesis_result = self.speech_synthesizer.speak_text_async(text).get()

        if (
            speech_synthesis_result.reason
            == speechsdk.ResultReason.SynthesizingAudioCompleted
        ):
            # Obtener los datos de audio como bytes
            audio_data = speech_synthesis_result.audio_data
            audio_bytes = audio_data
            # Crear el segmento de audio a partir de los datos de audio
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            # Salida de audio en altavoces del equipo
            #print("Speech synthesized for text [{}]".format(text))
            return audio_segment
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_synthesis_result.cancellation_details
            print(
                "Speech synthesis canceled: {}".format(cancellation_details.reason)
            )
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                if cancellation_details.error_details:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
        return None

    def create_conversation_txt(self):
        with open("conversacion.txt", "w") as file:
            for message in self.context:
                role = message["role"]
                content = message["content"]
                file.write(f"{role}: {content}\n")
        print("Archivo de conversación generado correctamente.")

    def run(self):
        while True:
            print("1. Escritura")
            print("2. Escuchar por micrófono")
            opcion = input("Elija una opcion: ")

            if opcion == '1' or opcion == '2':
                break
            else:
                print("Haz ingresado una opción no valida. Intenta de nuevo")

        # Inicializar merged_audio
        self.merged_audio = AudioSegment.empty()
        while True:
            print("")
            if opcion == "1":
                prompt = input("Usuario: ")
            elif opcion == "2":
                prompt = self.recognize_from_microphone()

            self.context.append({"role": "user", "content": f"{prompt}"})

            start_time = time.time()  # Guardar el tiempo de inicio
            response = self.get_completion_from_messages(self.context)  # Esperando respuesta
            end_time = time.time()  # Guardar el tiempo de finalización

            elapsed_time = end_time - start_time  # Calcular el tiempo transcurrido en segundos

            if elapsed_time > 10:
                print("La respuesta tardó demasiado. Saliendo del programa.")
                break

            self.context.append({"role": "assistant", "content": f"{response}"})
            print("Asistente:", response)

                # Guardar la entrada del micrófono en un archivo WAV
            if opcion == "2":
                audio_input = self.record_audio(elapsed_time)

                # Convertir el archivo WAV a formato mp3
                audio_input = AudioSegment.from_wav(audio_input)

                # Sintetizar la respuesta en audio
                audio_response = self.text_to_speech(response)

                # Agregar la entrada y respuesta al audio combinado
                self.merged_audio += audio_input + audio_response

                # Exportar el audio combinado a un archivo mp3
                self.merged_audio.export("conversacion_audio.mp3", format="mp3")

                # Reproducir el audio combinado
                #play(self.merged_audio)

            # Verificar si la conversación ha terminado
            if ("salir" in prompt.lower() or "salir." in prompt.lower() or keyboard.is_pressed("c")):
                messages = self.context.copy()
                messages.append(
                    {
                        "role": "system",
                        "content": """Create a CSV very detailed of what was discussed with the following instructions:\
                        Comma ',' separate columns and line break each row just like a normal CSV file. Write just the CSV without saying anything else.\
                        The fields must be: a column with the type of service required and the other columns with each of the questions and in the next row the \
                        answers for each question and type of service. For example:\
                        Create a CSV summary of what was discussed with the following columns:\n\n* Name\n* Service\n* Question 1\n* Answer 1\n* Question 2\n* Answer 2\n* ...\n\n""",
                    }
                )

                summary_response = self.get_completion_from_messages(messages, temperature=0)

                data_list = summary_response.split(
                    "\n"
                )  # Separar el string en una lista por cada salto de línea

                # Crear el archivo CSV
                with open("datos.csv", "w", newline="") as file:
                    writer = csv.writer(file)

                    # Escribir los encabezados
                    headers = data_list[0].split(",")
                    writer.writerow(headers)

                    # Escribir los datos
                    for line in data_list[1:]:
                        row = line.split(",")
                        writer.writerow(row)
                print("Archivo CSV generado correctamente.")
                self.create_conversation_txt()
                break


if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()