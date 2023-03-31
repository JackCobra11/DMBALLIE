from flask import Flask, render_template, request
from glob import glob
import numpy as np
import scipy
import keras
import os
import Network
import utls
import time
import cv2
import argparse
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    result_folder = request.form['result_folder']
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    input_folder = request.form['input_folder']
    path = glob(input_folder+'/*.*')

    model_name = request.form['model']
    mbllen = Network.build_mbllen((None, None, 3))
    mbllen.load_weights('../models/'+model_name+'.h5')
    opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mbllen.compile(loss='mse', optimizer=opt)

    flag = int(request.form['com'])

    lowpercent = int(request.form['lowpercent'])
    highpercent = int(request.form['highpercent'])
    maxrange = int(request.form['maxrange'])/10.
    hsvgamma = int(request.form['gamma'])/10.#HSV色彩空间 Hue、Saturation、Value（明度=亮度）
    img_base64_list=[]
    for i in range(len(path)):
        img_A_path = path[i]
        img_A = utls.imread_color(img_A_path)
        img_A = img_A[np.newaxis, :]

        starttime = time.clock()
        out_pred = mbllen.predict(img_A)
        endtime = time.clock()
        print('The ' + str(i + 1) + 'th image\'s Time:' + str(endtime - starttime) + 's.')
        fake_B = out_pred[0, :, :, :3]
        fake_B_o = fake_B

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        percent_max = sum(sum(gray_fake_B >= maxrange)) / sum(sum(gray_fake_B <= 1.0))
        # print(percent_max)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100 - highpercent) / 100.:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        fake_B = (fake_B - sub_value) * (1. / (1 - sub_value))

        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        if flag:
            outputs = np.concatenate([img_A[0, :, :, :], fake_B_o, fake_B], axis=1)
        else:
            outputs = fake_B

        filename = os.path.basename(path[i])
        img_name = result_folder+'/' + filename
        # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
        outputs = np.minimum(outputs, 1.0)
        outputs = np.maximum(outputs, 0.0)
        utls.imwrite(img_name, outputs)
        print(img_name)


        # Img file to Base64 on html
        # img_b64 = Image.fromarray(outputs.astype(np.uint8))
        # buffer = io.BytesIO()
        # img_b64.save(buffer, format='PNG')
        # img_str = base64.b64encode(buffer.getvalue()).decode()
        # img_base64_list.append(img_str)
        with open(img_name, "rb") as img_file:
            # 将图像编码为Base64格式
            img_str = base64.b64encode(img_file.read()).decode('utf-8')
            img_base64_list.append(img_str)

    output_paths=[]
    for i in range(len(path)):
        # 其他代码
        img_name1 = path[i]
        output_paths.append(img_base64_list[i])
    return render_template('result.html', output_paths=output_paths)
    #return 'Test completed successfully!'



if __name__ == "__main__":
    app.run(debug=True)
