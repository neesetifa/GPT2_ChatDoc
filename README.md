### GPT2 Chat Doctor<br>
Fine-tuning GPT-2 for Conversational AI

#### Overview
GPT2_ChatDoc is a project aimed at fine-tuning the GPT-2 model to enhance its conversational abilities. This project follows a methodology similar to Alpaca's fine-tuning approach, further improving it with domain-specific datasets. The goal is to create a chatbot capable of providing meaningful, context-aware responses while maintaining coherence across interactions.

#### Features
- Fine-tuning GPT-2 with high-quality conversational data<br>
  Pretrained model using GTP-2 small (https://huggingface.co/openai-community/gpt2/tree/main)<br>
  The model is first finetuned on Alpaca dataset to acquire multi-turn conversation and instruction-following ability(Following Stanford Alpaca https://github.com/tatsu-lab/stanford_alpaca).
- Support for domain-specific adaptation (medical)<br>
  Then finetuned on Chatdoctor200K (https://huggingface.co/datasets/LinhDuong/chatdoctor-200k) to get domain knowledge 
- Efficient training on consumer-grade GPUs(Using RTX4080)

#### Usage
To fine-tune GPT-2 on your dataset:
```rb
python train.py --dataset <your_dataset_path> --epochs 3 --batch_size 4
```
To run inference:
```rb
python demo.py --model_path <path_to_trained_model>
```

#### Progress
- [x] LoRA with larger GPT-2 model
- [ ] RLHF
  - [x] SFT on Alpaca Dataset and Medical Dataset(Instruction Following)
  - [x] No need for reward model
  - [ ] GRPO
- [ ] Retrieval Augmented Generation
- [ ] Improve the chatbot's long-term memory retention
