import os
import csv
import sys
from typing import Tuple, List

from datasets import load_dataset
from os.path import join as pj

from t5 import T5Transformer, T5EvaluationArgs, T5TrainingArgs, T5GenerateSettings

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_PROJECT'] = 'NLP Course 2022 - IDC'

FOLDER = os.getcwd()


def gen_dataset(csv_path: str, dataset):
    with open(csv_path, 'w', newline='') as csv_handle:
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow(['input', 'target'])
        for item in dataset:
            input_text = f'grammar: {item["sentence"]}'
            for corr in item['corrections']:
                if input_text and corr:
                    csv_writer.writerow([input_text, corr])


def get_tests() -> List[Tuple[str, str]]:
    tests = [
        ("The team plays good in decisive games . ", 'The team plays well in decisive games . '),
        ("I haven't barely started to think about my exam . ", 'I have barely started to think about my exam . '),
        ("The candidate promised not to rise taxes when elected . ",
         'The candidate promised not to raise taxes when elected . '),
        ("Most people not only are lifting weights at the gym, but they also do a cardiovascular workout . ",
         'Most people not only lift weights at the gym, but they also do a cardiovascular workout . '),
        ("Neither of the players in the last game was injured . ",
         'Neither of the players in the last game were injured . '),
        ("When I did my lab experiments, I tried to thoroughly document each of my measurements . ",
         'When I did my lab experiments, I tried to document thoroughly each of my measurements . '),
        ("Neither students nor their teacher are participating in this play . ",
         'Neither students nor their teacher is participating in this play . '),
        ('He told me that he has been working in Spain the previous year . ',
         'He told me that he had been working in Spain the previous year . '),
        ("I'm fed up for doing this excersice . ", "I'm fed up with doing this excersice . "),
        ('I went to the shop for buying some chocolate . ', 'I went to the shop to buy some chocolate . '),
        ('By this time next year, I have taken all my exams . ',
         'By this time next year, I will have taken all my exams . '),
        ("You wouldn't have done the cleaning.I would have done it tonight.",
         "You needn't have done the cleaning. I would have done it tonight . "),
        ('The best way to learn a language is to speaking a little every day . ',
         'The best way to learn a language is by speaking a little every day . ')
    ]

    return tests


def train(model_type: str, number_of_epochs: int, learning_rate: float):
    train_dataset = load_dataset("jfleg", split='validation[:]')
    eval_dataset = load_dataset("jfleg", split='test[:]')

    train_path = pj(FOLDER, 'train.csv')
    eval_path = pj(FOLDER, 'eval.csv')

    gen_dataset(csv_path=train_path, dataset=train_dataset)
    gen_dataset(csv_path=eval_path, dataset=eval_dataset)

    t5 = T5Transformer(model_name=model_type)

    eval_args = T5EvaluationArgs(batch_size=32)
    before_train_loss = t5.eval(input_filepath=eval_path, args=eval_args)

    per_device_train_batch_size = 32
    if model_type == 't5-base':
        per_device_train_batch_size = 16
    elif model_type == 't5-large':
        per_device_train_batch_size = 1
    train_args = T5TrainingArgs(
        num_train_epochs=number_of_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size)
    t5.train(train_path, args=train_args)

    after_train_loss = t5.eval(input_filepath=train_path, args=eval_args)

    beam_settings = T5GenerateSettings(num_beams=5, min_length=1, max_length=20)

    print(f'Model: {model_type}')
    print(f'Hyperparameters: {train_args.__dict__}')
    print(f'Generate settings: {beam_settings.__dict__}')
    print()

    print(f'Before training evaluation score: {before_train_loss}')
    print(f'After training evaluation score: {after_train_loss}')
    print()

    tests = get_tests()

    i = 1
    for test in tests:
        incorrect_sen, correct_sen = test
        _pred = t5.generate_text(f'grammar: {incorrect_sen}', args=beam_settings)

        print(f'########## {i} ##########')
        print(f'Incorrect sentence:\t\t{incorrect_sen}')
        print(f'Correct sentence:\t\t{correct_sen}')
        print(f'Model prediction:\t\t{_pred}')
        print(f'#######################')
        print()

        i += 1


def main():
    model_type = sys.argv[1]
    number_of_epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])

    train(model_type=model_type, number_of_epochs=number_of_epochs, learning_rate=learning_rate)


if __name__ == '__main__':
    main()
