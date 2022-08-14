import subprocess as sp

import os
import sys

i = 0
for model_type in ['t5-small', 't5-base', 't5-large']:
    for number_of_epochs in [1, 2, 3]:
        for learning_rate in [5e-5, 5e-6, 5e-7]:
            print(f'Starting: {model_type}, {number_of_epochs}, {learning_rate}')
            process = sp.Popen([sys.executable, f'{os.path.join(os.getcwd(), "main.py")}',
                                model_type, str(number_of_epochs), str(learning_rate)])
            process.wait()
