from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
import pdb

def main(model_path, mode = 'doctor'):
    assert mode in ['chat', 'doctor']
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    text_generator = TextGenerationPipeline(model, tokenizer)
    
    print('## 蛋饼🥚: Hello, I\'m Egg-Cookie, what do you want to know today?')
    prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
    if mode == 'chat':
        human_instruct = '### Instruction:\n'
        human_input = ''  # During chitchat, human use instruction, no input is used
    elif mode == 'doctor':
        human_instruct = '### Instruction:\nIf you are a doctor, please answer the medical questions based on the patient\'s description.\n\n'
        human_input = '### Input:\n'
    bot = '### Response:\n'
    
    history = []
    while True:
        human_question = input('** 饼干🍪: ')
        history.append(human_instruct + human_input + human_question)
        
        full_text = prompt + '\n\n'.join(history) + bot
        result = text_generator(full_text, max_length=500, do_sample=True,
                                pad_token_id=text_generator.tokenizer.eos_token_id)
        result_text = result[0]['generated_text']
        answer_without_prompt = result_text[len(full_text):]
    
        # encode_input = tokenizer(text, return_tensors='pt')
        # output = model(**encode_input)
        # output = model.generate(encode_input['input_ids'], max_length=50)
        # text = tokenizer.decode(output[0])
        # pdb.set_trace()

        print('## 蛋饼🥚: '+answer_without_prompt)
        history.append(bot+answer_without_prompt)



if __name__ == "__main__":
    model_path = './exp2/checkpoint-30000/'
    main(model_path)
# Tell me something about Thanksgiving.
# Why do we eat turkey on Thanksgiving?
# What do people usually do after having Thanksgiving dinner?
# I caught a cold yesterday, now I'm sneezing and coughing, what should I do?
