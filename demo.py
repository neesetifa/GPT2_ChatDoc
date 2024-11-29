from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline
import pdb

def main(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    text_generator = TextGenerationPipeline(model, tokenizer)
    
    print('## 蛋饼🥚: Hello, I\'m Egg-Cookie, what do you want to know today?')
    instruction = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### '
    human = 'Instruction:\n'
    bot = '### Response:'
    history = []

    while True:
        human_question = input('** 饼干🍪: ')
        # human_question = 'Give three tips for staying healthy.'
        history.append(human+human_question)
        
        full_text = instruction + '\n\n'.join(history) + bot
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
    model_path = './exp/checkpoint-9700/'
    main(model_path)
# Tell me something about Thanksgiving.
# Why do we eat turkey on Thanksgiving?
# What do people usually do after having Thanksgiving dinner?
