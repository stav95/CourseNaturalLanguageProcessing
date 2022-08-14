import logging
import tempfile
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          T5ForConditionalGeneration, T5TokenizerFast, BatchEncoding)
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer


def detect_cuda_device_number() -> int:
    return torch.cuda.current_device() if torch.cuda.is_available() else -1


@dataclass
class T5GenerateSettings:
    min_length: int = 10
    max_length: int = 50
    do_sample: bool = False
    early_stopping: bool = False
    num_beams: int = 1
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1
    no_repeat_ngram_size: int = 0


@dataclass
class T5TrainingArgs:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_input_length: int = 1024
    max_output_length: int = 1024
    preprocessing_processes: int = 1

    def get_seq2seq_dict(self) -> Dict[str, float]:
        d = self.__dict__
        d.pop('preprocessing_processes')
        d.pop('max_input_length')
        d.pop('max_output_length')
        return d


@dataclass
class T5EvaluationArgs:
    batch_size: int = 1
    preprocessing_processes: int = 1
    max_input_length: int = 1024
    max_output_length: int = 1024


class T5Trainer:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5TokenizerFast,
                 device: torch.device, logger: logging.Logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

        self.max_input_length = 1024
        self.max_output_length = 1024

    def train(self, input_filepath: str, args: T5TrainingArgs = None):
        if args is None:
            args = T5TrainingArgs()

        self.logger.info("Preprocessing training data...")

        dataset = load_dataset("csv", data_files={"train": input_filepath}, delimiter=",")

        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=args.preprocessing_processes,
            remove_columns=["input", "target"],
        )

        self.logger.info("Training...")

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # noinspection PyTypeChecker
            training_args = Seq2SeqTrainingArguments(
                tmp_dir_name,
                do_train=True,
                do_eval=False,
                report_to=["wandb"],
                logging_steps=1,
                save_strategy="no",
                **args.get_seq2seq_dict()
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            trainer.train()

    def eval(self, input_filepath: str, args: T5EvaluationArgs = None) -> float:
        if args is None:
            args = T5EvaluationArgs()

        self.logger.info("Preprocessing evaluating data...")
        dataset = load_dataset("csv", data_files={"eval": input_filepath}, delimiter=",")

        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["input", "target"],
            num_proc=args.preprocessing_processes
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = Seq2SeqTrainingArguments(
                tmp_dir_name,
                do_train=False,
                do_eval=True,
                seed=42,
                report_to=["none"],
                per_device_eval_batch_size=args.batch_size,
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=eval_args,
                eval_dataset=tokenized_dataset['eval'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            result = trainer.evaluate()
            return float(result["eval_loss"])

    def preprocess_function(self, examples) -> BatchEncoding:
        model_inputs = self.tokenizer(examples["input"], max_length=self.max_input_length, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples["target"], max_length=self.max_output_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class T5Transformer:
    def __init__(self, model_name: str = "t5-base", load_path: str = ""):
        if load_path != "":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=load_path)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=load_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        self.model.eval()
        self.logger = logging.getLogger(__name__)

        handler = logging.StreamHandler()
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._device == 'cuda':
            self.model.to(self._device)
        self.logger.info("Using model: %s", self._device)

        device_number = detect_cuda_device_number()

        self.t2t_pipeline = Text2TextGenerationPipeline(model=self.model,
                                                        tokenizer=self.tokenizer, device=device_number)

        self.trainer = T5Trainer(model=self.model, tokenizer=self.tokenizer, device=self._device, logger=self.logger)

    def generate_text(self, text: str, args: T5GenerateSettings = None) -> str:
        if args is None:
            args = T5GenerateSettings()

        output = self.t2t_pipeline(text, **args.__dict__)

        result = output[0]['generated_text']
        return result

    def train(self, input_filepath: str, args: T5TrainingArgs = None):
        if args is None:
            args = T5TrainingArgs()

        self.trainer.train(input_filepath=input_filepath, args=args)

    def eval(self, input_filepath: str, args: T5EvaluationArgs = None) -> float:
        if args is None:
            args = T5EvaluationArgs()

        result = self.trainer.eval(input_filepath=input_filepath, args=args)
        return result

    def save(self, path: str):
        self.model.save_pretrained(save_directory=path)
        self.tokenizer.save_pretrained(save_directory=path)
