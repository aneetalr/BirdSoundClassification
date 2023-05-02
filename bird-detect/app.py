from skimage import transform
from PIL import Image
import soundfile as sf
from pydub import AudioSegment
import gc
from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils import secure_filename
import os
import shutil
import librosa
import librosa.display
import numpy as np
from flask import jsonify
import json
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(False)
matplotlib.use('Agg')

UPLOAD_FOLDER = 'static/output'

rslt = -1
file = ""
specd = []
audio = []


app = Flask(__name__)


@app.route('/')
def index():
    output = os.path.join(os.getcwd(), 'static/output')
    if os.path.exists(output):
        shutil.rmtree(output)
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global rslt, file, audio, specd
    if request.method == 'POST':
        output = os.path.join(os.getcwd(), 'static/output')
        if os.path.exists(output):
            shutil.rmtree(output)
        if not os.path.exists(output):
            os.mkdir(output)
        f = request.files['file1']
        filename = secure_filename(f.filename)
        f.save(os.path.join(UPLOAD_FOLDER, filename))

        pt = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename)

        file = filename
        size = {'desired': 10,  # [seconds]
                'minimum':  5,  # [seconds]
                'stride': 1,  # [seconds]
                'name': 5  # [number of letters]
                }
        signal, sr = librosa.load(pt)  # sr = sampling rate
        # length of step between two cuts in seconds
        step = (size['desired']-size['stride'])*sr

        nr = 0
        if not os.path.exists(output+"/spec"):
            os.mkdir(output+"/spec")
        if not os.path.exists(output+"/wav"):
            os.mkdir(output+"/wav")
        if not os.path.exists(output+"/mp3"):
            os.mkdir(output+"/mp3")
        if not os.path.exists(output+"/specpre"):
            os.mkdir(output+"/specpre")
        if (len(signal)//sr) <= 10:
            nr = 1
            melpath = output+"/spec/"+file+"_"+str(nr)+".png"
            dir2 = output+"/wav/"+file+"_"+str(nr)+".wav"
            dir3 = output+"/mp3/"+file+"_"+str(nr)+".mp3"
            saveMel(signal, melpath, dir2, dir3, sr)
        else:
            for start, end in zip(range(0, len(signal), step), range(size['desired']*sr, len(signal), step)):
                # cut file and save each piece
                nr = nr+1
                # save the file if its length is higher than minimum
                if end-start > size['minimum']*sr:
                    melpath = output+"/spec/"+file+"_"+str(nr)+".png"
                    dir2 = output+"/wav/"+file+"_"+str(nr)+".wav"
                    dir3 = output+"/mp3/"+file+"_"+str(nr)+".mp3"
                    saveMel(signal[start:end], melpath, dir2, dir3, sr)

        res_ls = []
        model = keras.models.load_model('model/bird_resnet50.h5')
        for x in os.listdir(os.path.join(os.getcwd(), UPLOAD_FOLDER, "spec")):
            spec = os.path.join(os.getcwd(), UPLOAD_FOLDER, "spec", x)
            specp = os.path.join(os.getcwd(), UPLOAD_FOLDER, "specpre", x)

            np_image = Image.open(spec)
            np_image = np.array(np_image).astype('float32')/255
            np_image = transform.resize(np_image, (224, 224, 3))
            plt.imsave(specp, np_image)
            np_image = np.expand_dims(np_image, axis=0)

            pred = model.predict(np_image)
            res_ls.append(pred)

        avg_res = [sum(x) / len(x) for x in zip(*res_ls)]
        print(avg_res)
        print(res_ls)
        rslt = np.argmax(avg_res)
        print(rslt)
        for x in res_ls:
            print(np.argmax(x))
        melpath = output+"/spec/"
        dir3 = output+"/mp3/"
        audio = os.listdir(dir3)
        specd = os.listdir(melpath)
        specp = os.listdir(output+"/specpre/")
        # json_string = json.dumps([arr.tolist() for arr in res_ls])
        # print(json_string)
        # response2 = jsonify(json.loads(json_string))
        # print(response2)
        return render_template('result.html', rslt=rslt, file=file, spec=specd, specp=specp, audio=audio)
    else:
        return render_template('index.html')
    #     return '''
    #     <script>
    #         alert("Form submitted successfully!");
    #         window.location.href = "/thank-you";
    #     </script>
    # '''


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


def saveMel(signal, directory, directory2, directory3, sr):
    # write the signal to a WAV file
    sf.write(directory2, signal, sr)

    # load the WAV file
    sound = AudioSegment.from_wav(directory2)

    # export the sound to an MP3 file
    sound.export(directory3, format='mp3')

    gc.enable()
    # MK_spectrogram modified
    N_FFT = 1024         #
    HOP_SIZE = 1024
    N_MELS = 128          # Higher
    WIN_SIZE = 1024      #
    WINDOW_TYPE = 'hann'
    FEATURE = 'mel'      #
    FMIN = 1400

    fig = plt.figure(1, frameon=False)
    fig.set_size_inches(6, 6)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    S = librosa.feature.melspectrogram(y=signal, sr=sr,
                                       n_fft=N_FFT,
                                       hop_length=HOP_SIZE,
                                       n_mels=N_MELS,
                                       htk=True,
                                       # higher limit ##high-pass filter freq.
                                       fmin=FMIN,
                                       fmax=sr/2)  # AMPLITUDE
    librosa.display.specshow(librosa.power_to_db(
        S**2, ref=np.max), fmin=FMIN)  # power = S**2

    fig.savefig(directory)
    plt.ioff()
    # plt.show(block=False)
    fig.clf()
    ax.cla()
    plt.clf()
    plt.close('all')


def load(image):
    np_image = Image.open(image)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


if __name__ == '__main__':
    app.run(threaded=False)
