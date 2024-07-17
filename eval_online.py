import csv
import glob
import os
import re
import shutil
import tempfile
from datetime import datetime
from zipfile import ZipFile

from flask import Flask, request
from werkzeug.utils import secure_filename

from eval import main as eval_fun

app = Flask(__name__)

eval_online_dir = './eval_online_files'
ref_db_dir = '../../../dbs/dfuc24/DFUC2022_train_masks'
split_file_path = './db_0_fold_0_out_of_5_debug_files_list.json'

if os.path.exists(eval_online_dir) and len(os.listdir(eval_online_dir)) > 0:
    dirs = os.listdir(eval_online_dir)
    dirs = list(sorted([int(dir_id) for dir_id in dirs]))
    last_id = dirs[-1]
else:
    last_id = 0


@app.route("/")
def main_page():
    dirs = os.listdir(eval_online_dir)
    test_set_tests = []
    whole_db_tests = []

    for dir in dirs:
        report_path = os.path.join(eval_online_dir, dir, 'eval_report.csv')
        comment_path = os.path.join(eval_online_dir, dir, 'comment.txt')

        with open(report_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line = ''
            for row in csv_reader:
                line = row
            avg_dice = line[-1]

        with open(comment_path, 'r') as f:
            comment = f.read()

        only_testset_path = os.path.join(eval_online_dir, dir, 'only_testset.txt')
        if os.path.exists(only_testset_path):
            test_set_tests.append([avg_dice, comment])
        else:
            whole_db_tests.append([avg_dice, comment])

    test_set_tests = list(sorted(test_set_tests, key=lambda x: -float(x[0])))
    whole_db_tests = list(sorted(whole_db_tests, key=lambda x: -float(x[0])))

    result = 'testy na testsecie (dla porownania z unetem)<br />' + \
              '<table>' + \
              '<tr><td>avg_dice</td><td>comment</td></tr>'
    for row in test_set_tests:
        result += '<tr><td>' + row[0] + '</td><td>' + row[1] + '</td></tr>'
    result += '</table><br />'
    result += 'testy na całym datasecie z 2022 (prawdopodobnie bardziej wiarygodne wyniki)<br />' + \
              '<table>' + \
              '<tr><td>avg_dice</td><td>comment</td></tr>'
    for row in whole_db_tests:
        result += '<tr><td>' + row[0] + '</td><td>' + row[1] + '</td></tr>'
        result += '</table><br />'

    result += '<br /><a href=/test>Dodaj wynik</a>'
    return result


@app.route("/test", methods=['GET', 'POST'])
def test():
    global last_id
    id_for_request = last_id
    last_id += 1
    if request.method == 'GET':
        return ('<form action="/test" method="POST" enctype="multipart/form-data"/>'
                '<input type="file" name="file"/>Zip ze spakowanymi predykcjami masek, nazwa jak plik z obrazem, rozszerzenie png<br /><br />'
                '<input type="checkbox" name="only_testset" value="testset" />tylko test set (dla uneta)'
                '(zaznacz jesli testy maja byc na wydzielonym test secie, w przeciwnym razie na calosci 2k obrazkow)<br /><br />'
                '<input type="text" name="comment" />komentarz (aby latwiej zidentyfikowac wyniki) <br /><br />'
                f'<input type="submit" value="Testuj"><br /><br />'
                '</form>')
    elif request.method == 'POST':

        if 'only_testset' in request.form:
            only_testset = True
        else:
            only_testset = False

        comment = request.form['comment']

        file = request.files['file']
        if not file or not file.filename.rsplit('.', 1)[1] in ['zip']:
            return 'zly plik'
        filename = secure_filename(file.filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_file_path = os.path.join(temp_dir, filename)
            file.save(tmp_file_path)

            unzip_dir = os.path.join(temp_dir, datetime.now().strftime("%H_%M_%S"))
            os.makedirs(unzip_dir)
            with ZipFile(tmp_file_path, 'r') as zip_file:
                zip_file.extractall(path=unzip_dir)

            os.remove(tmp_file_path)

            no_of_mask_files = 0
            pred_masks_dir = ''
            for root, dirs, files in os.walk(unzip_dir):
                pred_masks_files = [os.path.join(root, file) for file in files if re.match(r".+\.png", os.path.basename(file))]
                if len(pred_masks_files) > no_of_mask_files:
                    no_of_mask_files = len(pred_masks_files)
                    pred_masks_dir = root

            config = {
                'split_file_path': split_file_path if only_testset else None,
                'ref_masks_dir': ref_db_dir,
                'pred_masks_dir': pred_masks_dir,
                'eval_report_output': f"{os.path.join(temp_dir, 'eval_report.csv')}"
            }

            eval_fun(config)

            entry_dir = os.path.join(eval_online_dir, str(id_for_request))
            os.makedirs(entry_dir, exist_ok=True)
            report_path = os.path.join(entry_dir, 'eval_report.csv')
            shutil.copy(config['eval_report_output'], report_path)

        comment_path = os.path.join(entry_dir, 'comment.txt')
        with open(comment_path, 'w') as f:
            f.write(comment)

        with open(report_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            line = ''
            for row in csv_reader:
                line = row
            avg_dice = line[-1]

        if only_testset:
            only_testset_path = os.path.join(entry_dir, 'only_testset.txt')
            with open(only_testset_path, 'w') as f:
                pass

        return f'avg dice: {avg_dice}<br /><a href="/">powrot</a>'


if __name__ == "__main__":
    os.makedirs(eval_online_dir, exist_ok=True)
    app.run(debug=True, host='0.0.0.0')