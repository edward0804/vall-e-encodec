import wandb
from datasets import load_dataset
from jiwer import wer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from encodec_model.nar_bart_model import NARBartForConditionalGeneration
from encodec_model.nar_encodec_bart_model import NARBartEncodecForConditionalGeneration
name = 'libri960_nar_wr0.08_lr1e-4'
wandb.init(name)

# Load dataset and tokenizer
# train_dataset = load_dataset("voidful/librispeech_encodec", split="trainclean100+trainclean360+trainother500")
train_dataset = load_dataset("lca0503/soxdata_small_encodec")
valid_dataset = load_dataset("voidful/librispeech_encodec", split="validationclean")
tokenizer = AutoTokenizer.from_pretrained("voidful/bart-base-unit")
# model = NARBartForConditionalGeneration.from_pretrained("voidful/bart-base-unit")
model = NARBartEncodecForConditionalGeneration.from_pretrained("voidful/bart-base-unit")
print(train_dataset)

# Set training parameters
training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.08,
    weight_decay=1e-4,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=10000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=10000,
    # fp16=True,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
)

# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def pad_sequences(sequences, max_length, padding_value):
    return [sequence + [padding_value] * (max_length - len(sequence)) for sequence in sequences]

def pad_sequences_for_input_id(sequences, max_length, padding_value):
    return sequences + [padding_value] * (max_length - len(sequences))

def pad_sequences_for_attn_mask(sequences, max_length):
    return sequences + [0] * (max_length - len(sequences)) 

# Define training and validation functions
def process_data_to_model_inputs(batch):

    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []
    
    src_prompt_ids = []
    src_prompt_mask = []

    max_length = 1023  # You can set this to a suitable value based on your dataset
    input_datas = []
    for b in range(len(batch["instruction"])):
        
        # [Instruction Prompt] </s> [Transcription Prompt].
        batch_data = batch["instruction"][b] + " </s> " + batch["transcription"][b]

        # Tokenize prompt data.
        prompt_data = tokenizer(batch_data, padding='max_length', truncation=True, max_length=max_length)

        for i in range(1, 8):
            src_prompt_ids.append(prompt_data["input_ids"])
            src_prompt_mask.append(prompt_data["attention_mask"])

        # source encodec input, which have 8 group units.
        encodec_input = []

        for i in range(0, 8):
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                ([f"v_tok_{u + i * 1000}" for u in batch[f'src_encodec_{i}'][b]]))
            
            encodec_input.append(decoder_input_id)

        for i in range(1, 8):
            input_datas.append(encodec_input)
        
        # decoder input, which is target encodec units.
        for i in range(1, 8):
            decoder_input_id = tokenizer.convert_tokens_to_ids(
                [f"v_tok_{u + (i - 1) * 1000}" for u in batch[f'tgt_encodec_{i - 1}'][b]])
            
            label = tokenizer.convert_tokens_to_ids([f"v_tok_{u + i * 1000}" for u in batch[f'tgt_encodec_{i}'][b]])

            # print(f"decoder_input_id: {len(decoder_input_id)}")

            decoder_input_ids.append(decoder_input_id)
            labels.append(label)
    
    # Pad the input data sequences (source data) and create attention masks
    padded_input_datas = []
    attention_masks = []
    for input_data in input_datas:
        padded_input_data = []
        for seq in input_data:
            seq_len = len(seq)
            padded_seq = seq + [tokenizer.pad_token_id] * (max_length - seq_len)
            mask = [1] * seq_len + [0] * (max_length - seq_len)
            padded_input_data.append(padded_seq)
        padded_input_datas.append(padded_input_data)
        attention_masks.append(mask)
       
    batch["input_ids"] = padded_input_datas
    batch["attention_mask"] = attention_masks

    # Pad decoder_input_ids and labels
    decoder_input_ids = pad_sequences(decoder_input_ids, max_length=max_length, padding_value=tokenizer.pad_token_id)
    labels = pad_sequences(labels, max_length=max_length, padding_value=-100)

    return {
        "input_ids": padded_input_datas,
        "attention_mask": attention_masks,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "src_prompt_ids": src_prompt_ids,
        "src_prompt_mask": src_prompt_mask
    }

def filter_examples(example):
     return len(example[f"src_encodec_0"]) <= 1000 and len(example[f"tgt_encodec_0"]) <= 1000

train_dataset = train_dataset.filter(filter_examples)
# valid_dataset = valid_dataset.filter(filter_examples)

train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    remove_columns=["src_encodec_2", "src_encodec_3", "src_encodec_4", "src_encodec_5", "src_encodec_6", "src_encodec_7",
    "src_encodec_1", "tgt_encodec_0", "tgt_encodec_1", "tgt_encodec_2", "tgt_encodec_3", "tgt_encodec_4", "tgt_encodec_5",
    "tgt_encodec_6", "tgt_encodec_7", "file_id", "instruction", "transcription", "src_encodec_0"],
    batched=True,
    batch_size=training_args.per_device_train_batch_size
)
print(train_dataset)
# valid_dataset = valid_dataset.map(process_data_to_model_inputs,
#                                   remove_columns=valid_dataset.column_names,
#                                   batched=True,
#                                   batch_size=training_args.per_device_eval_batch_size
#                                   )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.max(logits, axis=-1).indicies
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute WER
    wer_value = wer([" ".join(filter(None, i.split("v_tok_"))) for i in decoded_labels],
                    [" ".join(filter(None, i.split("v_tok_"))) for i in decoded_preds])
    print("pred_result")
    print("=================================")
    for i in range(10):
        print("target:", labels[i])
        print("pred:", predictions[i])
        print("-----------------")
    print("=================================")
    return {"wer": wer_value}

# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    # eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics,
)

# Start training
trainer.train()
# trainer.evaluate()
