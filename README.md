### GPT2 Chat Doctor<br>
This repo implements a Doctor Chatbot.<br>
- Pretrained model using GTP-2 small (https://huggingface.co/openai-community/gpt2/tree/main)<br>
- The model is first finetuned on Alpaca dataset to acquire multi-turn conversation and instruction-following ability(Following Stanford Alpaca https://github.com/tatsu-lab/stanford_alpaca).<br>
  Then finetuned on Chatdoctor200K (https://huggingface.co/datasets/LinhDuong/chatdoctor-200k)<br>
- [ ] LoRA
- [ ] Retrieval Augmented Generation
- [ ] RLHF
